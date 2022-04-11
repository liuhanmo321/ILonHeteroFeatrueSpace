import torch
from torch import nn
# from saint.ours_model import SAINT

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_acl
from augmentations import embed_data_cont
from augmentations import add_noise
import copy

from prettytable import PrettyTable
import time
import os
import numpy as np

sp_weight = 0.7

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True)

def ours_cat(opt):

    from saint.ours_cat_model import SAINT

    # save_path = './results/' + '_'.join(saving_list) + '.csv'
    save_path = opt.result_path

    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,opt.data_name,opt.run_name)
    if opt.task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'

    alpha = opt.alpha
    beta = opt.beta
    gamma = opt.gamma

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)
    os.makedirs(modelsave_path, exist_ok=True)

    if opt.active_log:
        import wandb        
        if opt.task=='multiclass':
            wandb.init(project="saint_v2_all_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.data_name)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'{opt.method}_{opt.task}_{str(opt.attentiontype)}_{str(opt.data_name)}_{str(opt.set_seed)}')

    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related

    if y_dims[0] == 2 and opt.task == 'binary':
        # opt.task = 'binary'
        criterion = nn.NLLLoss().to(device)
    elif y_dims[0] > 2 and  opt.task == 'multiclass':
        # opt.task = 'multiclass'
        criterion = nn.NLLLoss().to(device)
    else:
        raise'case not written yet'
    CE_loss = nn.CrossEntropyLoss().to(device)
    

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    shared_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    specific_matrix = np.zeros((opt.num_tasks, opt.num_tasks))

    total_time = 0

    for data_id in range(opt.num_tasks):

        if data_id == 0:
            model = SAINT(
                categories = tuple(cat_dims_group[0]), 
                num_continuous = len(con_idxs_group[0]),                
                dim = opt.embedding_size,                           
                dim_out = 1,                       
                depth = opt.transformer_depth,                       
                heads = opt.attention_heads,                         
                attn_dropout = opt.attention_dropout,             
                ff_dropout = opt.ff_dropout,                  
                mlp_hidden_mults = (4, 2),       
                cont_embeddings = opt.cont_embeddings,
                attentiontype = opt.attentiontype,
                final_mlp_style = opt.final_mlp_style,
                y_dim = y_dims[0]
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        ## Choosing the optimizer

        if opt.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                                momentum=0.9, weight_decay=5e-4)
            from utils import get_scheduler
            scheduler = get_scheduler(opt, optimizer)
        elif opt.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=opt.lr)
        elif opt.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

        ## Prepare past model
        old_shared_extractor = copy.deepcopy(model.shared_extractor).to(device)
        old_specific_classifier = copy.deepcopy(model.specific_classifier).to(device)

        lr = opt.lr
        best_loss = np.inf
        best_valid_auroc = 0
        best_valid_accuracy = 0
        stop_count = 0
        print('Training begins now.')
        
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad() 

                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                    
                specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
                shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                feature = torch.cat([shared_feature, specific_feature], dim=1)

                output = model.specific_classifier[data_id](feature)                
                
                loss = CE_loss(output, y_gts.squeeze())

                with torch.no_grad():
                    old_specific_features = [model.specific_extractor[temp_id](x_categ_enc, x_cont_enc)[:,0,:] for temp_id in range(data_id)]
                    if not opt.no_distill:
                        old_shared_feature = old_shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                
                if data_id > 0 and not opt.no_discrim:
                    temp_shared_feature = Variable(shared_feature.data, requires_grad=False)
                    disc_features = [torch.cat([temp_shared_feature, old_specific_features[i]], dim=1) for i in range(data_id)]
                    disc_outputs = [model.specific_classifier[data_id](disc_features[i]) for i in range(data_id)]
                    disc_outputs.append(output)
                    
                    specific_p = [torch.softmax(output, dim=1) for output in disc_outputs]
                    label_p = [-nn.NLLLoss(reduction='none')(p, y_gts.squeeze()) for p in specific_p]
                    max_label_p = torch.max(torch.stack(label_p[:-1], 1), 1).values
                    temp_dis_score = label_p[-1] - max_label_p

                    temp_dis_score = torch.exp(-temp_dis_score * opt.gamma)
                    temp_dis_score = F.normalize(temp_dis_score, dim=0)
                    dis_score = torch.reshape(temp_dis_score, (y_gts.shape[0], 1))

                    dis_output = dis_score * torch.log_softmax(output, dim=1)
                    dis_loss = criterion(dis_output, y_gts.squeeze())
                    loss += opt.beta * dis_loss            
                
                if not opt.no_distill:
                    temp_categ_enc, temp_cont_enc = Variable(x_categ_enc.data, requires_grad=False), Variable(x_cont_enc.data, requires_grad=False)
                    shared_feature =  model.shared_extractor(temp_categ_enc, temp_cont_enc)[:, 0, :]

                    for temp_data_id in range(data_id):
                        old_dist_feature = torch.cat([old_shared_feature, old_specific_features[temp_data_id]], dim=1)
                        dist_feature = torch.cat([shared_feature, old_specific_features[temp_data_id]], dim=1)
                        old_y_outs = old_specific_classifier[temp_data_id](old_dist_feature)
                        y_outs = model.specific_classifier[temp_data_id](dist_feature)

                        loss += MultiClassCrossEntropy(y_outs, old_y_outs, T=2) / data_id

                loss.backward()
                optimizer.step()
                if opt.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()
            end_time = time.time()
            total_time += end_time - start_time
            
            if epoch%1==0:
                print('[EPOCH %d] Running Loss: %.3f' % (epoch + 1, running_loss))
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

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_acl(model, testloaders[temp_data_id], device,  opt.task, temp_data_id, alpha)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
        
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'batch_size', 'lr', 'alpha', 'beta', 'gamma', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.batchsize, opt.lr, opt.alpha, opt.beta, opt.gamma, total_parameters])
    print(table)
    print('===========================================================================')
    if not opt.hyper_search:
        with open(save_path, 'a+') as f:
            f.write(table.get_string())
            f.write('\n')
            f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
            f.write(str(result_matrix))
            f.write('\n')
            f.write('====================================================================\n\n')
            f.close()       
    else:
        return  np.mean(result_matrix[:, -1])

