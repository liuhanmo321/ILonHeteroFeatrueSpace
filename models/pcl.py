import torch
from torch import nn
# from saint.ours_model import SAINT
from sklearn.metrics import roc_auc_score
from data import DataSetCatCon, sub_data_prep
# import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_ours
from augmentations import embed_data_cont
import copy

from prettytable import PrettyTable
import time
import os
import numpy as np
import torch.autograd as autograd

def classification_scores(model, dloader, device, task, data_id, alpha):
    model.eval()
    feat_idx = 0 if model.extractor_type == 'transformer' else -1
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

            shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,feat_idx,:]
            pred_ps = [class_classifier(shared_feature) for class_classifier in model.class_classifier[data_id]]
            y_outs = torch.cat(pred_ps, dim=1)
         
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob, m(y_outs)[:,-1].float()],dim=0) # modified the possibility
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc


def calc_gradient_penalty(model, real_data, fake_data, device):
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1)    
    alpha = alpha.expand(real_data.size()).to(device)
    # alpha = alpha.cuda()
    # alpha = 0.5

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = real_data

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = model(interpolates)
    # disc_interpolates = disc_interpolates[:,0]
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(
                                disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1))**8).mean()
    # gradient_penalty = (torch.max(gradients.norm(2, dim=1), threshold)**2).mean()
    return gradient_penalty

def parameter_l2_penalty(model, avg_model):
    
    weight = (model.layers[0].weight.data - avg_model.weight1).norm()
    weight = weight + (model.layers[2].weight.data - avg_model.weight2).norm()
    
    return weight

class Model_Maintain():
    def __init__(self):
        self.weight1 = None
        self.weight2 = None
    
    def update(self, models):
        for i, model in enumerate(models):
            if i == 0:
                temp_weight1 = model.layers[0].weight.data.clone().detach()
                temp_weight2 = model.layers[2].weight.data.clone().detach()
            else:
                temp_weight1 += model.layers[0].weight.data.clone().detach()
                temp_weight2 += model.layers[2].weight.data.clone().detach()
        
        temp_weight1 = temp_weight1 / len(models)
        temp_weight2 = temp_weight2 / len(models)

        self.weight1 = temp_weight1
        self.weight2 = temp_weight2
    

def pcl(opt):
    from saint.pcl_model import SAINT

    # save_path = './results/' + '_'.join(saving_list) + '.csv'
    save_path = opt.result_path
    
    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,opt.data_name,opt.run_name)

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2], single_class=False)
    _, _, cls_trainloaders, _, _, _ = sub_data_prep(opt, datasplit=[.65, .15, .2], single_class=True)
    # torch.manual_seed(opt.set_seed)
    os.makedirs(modelsave_path, exist_ok=True)

    avg_model = Model_Maintain()
    # Data Set Related

    # cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)
    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2])

    # Model Related

    ce = nn.CrossEntropyLoss().to(device)
    nll = nn.NLLLoss().to(device)
    feat_idx = 0 if opt.extractor_type == 'transformer' else -1

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    global_index = 0

    total_time = 0

    for data_id in range(opt.num_tasks):

        if data_id == 0:
            model = SAINT(
                categories = tuple(cat_dims_group[0]), 
                num_continuous = len(con_idxs_group[0]),                
                dim = opt.embedding_size,                           
                depth = opt.transformer_depth,                       
                heads = opt.attention_heads,                         
                attn_dropout = opt.attention_dropout,             
                ff_dropout = opt.ff_dropout,                  
                y_dim = y_dims[0],
                extractor_type = opt.extractor_type
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        # Class Level Training
        for cls in range(y_dims[data_id]):
            if data_id + cls > 0:
                model.set_parameters(data_id, cls, avg_model)

            optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
            
            lr = opt.lr
            best_loss = np.inf
            stop_count = 0
            print('Training begins now.')
            
            for epoch in range(opt.epochs):
                start_time = time.time()
                model.train()
                running_loss = 0.0
                for i, data in enumerate(cls_trainloaders[data_id][cls], 0):
                    optimizer.zero_grad() 

                    x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                    
                    # print(torch.unique(y_gts))
                    shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,feat_idx,:]
                    class_output = model.class_classifier[data_id][cls](shared_feature)

                    loss = -torch.log(torch.sigmoid(class_output) + 1e-2).mean()

                    loss_pen = calc_gradient_penalty(model.class_classifier[data_id][cls], shared_feature, shared_feature, device)

                    loss = loss + 10000000 * loss_pen * opt.alpha
                    
                    loss_l2_transfer = 0
                    if global_index > 0:
                        loss_l2_transfer = parameter_l2_penalty(model.class_classifier[data_id][cls], avg_model)
                        loss += 100000 * loss_l2_transfer * opt.beta

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                end_time = time.time()
                total_time += end_time - start_time
                
                if epoch%1==0:
                    print('[EPOCH %d] Running Loss: %.3f \t Loss Pen: %.6f \t Loss l2 %.6f' % (epoch + 1, running_loss, 10000000*loss_pen, 100000*loss_l2_transfer))
                    if running_loss < best_loss:
                        best_loss = running_loss             
                        stop_count = 0
                    else:
                        stop_count += 1

                    if stop_count == opt.patience:
                        print('patience reached')
                        lr = lr / 10
                        stop_count = 0
                        if lr < opt.lr_lower_bound:
                            break
                        optimizer = optim.AdamW(model.parameters(),lr=lr)
                
            avg_model.update(model.get_model_list(cls))


        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores(model, testloaders[temp_data_id], device,  opt.task, temp_data_id, opt.alpha)
            result_matrix[temp_data_id, data_id] = temp_test_auroc

        
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))

    embedding_parameters = count_parameters(model.embeds) + count_parameters(model.simple_MLP)

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'batch_size', 'distill_frac', 'alpha', 'beta', 'gamma', 'params', 'emb params'])
    table.add_row(['%.2f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, opt.batchsize, opt.distill_frac, opt.alpha, opt.beta, opt.gamma, total_parameters, embedding_parameters])
    print(table)

    print('===========================================================================')

    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(result_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()       
    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])