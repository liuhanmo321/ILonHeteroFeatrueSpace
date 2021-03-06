import torch
from torch import nn
# from saint.acl_model import SAINT

from data import DataSetCatCon, sub_data_prep
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from utils.discriminator import Discriminator
from tools import count_parameters, classification_scores_acl
from augmentations import embed_data_cont
import copy

from prettytable import PrettyTable
import time
import os
import numpy as np

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True)

def baseline_acl(opt):

    from saint.acl_model import SAINT
    save_path = opt.result_path

    alpha = opt.alpha

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    
    # Data Set Related

    # cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)
    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2])

    # Model Related

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    total_time = 0

    for data_id in range(opt.num_tasks):

        if data_id == 0:
            model = SAINT(
                categories = tuple(cat_dims_group[0]), 
                num_continuous = len(con_idxs_group[0]),                
                dim = opt.embedding_size,                           
                # dim_out = 1,                       
                depth = opt.transformer_depth,                       
                heads = opt.attention_heads,                         
                attn_dropout = opt.attention_dropout,             
                ff_dropout = opt.ff_dropout,                  
                # mlp_hidden_mults = (4, 2),       
                # cont_embeddings = opt.cont_embeddings,
                # attentiontype = opt.attentiontype,
                # final_mlp_style = opt.final_mlp_style,
                y_dim = y_dims[0]
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        discriminator_params = {'num_shared_features': opt.embedding_size, 'ntasks': data_id+1}
        discriminator = Discriminator(discriminator_params, data_id).to(device)

        ## Choosing the optimizer

        if opt.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                                momentum=0.9, weight_decay=5e-4)
            from utils import get_scheduler
            scheduler = get_scheduler(opt, optimizer)
        elif opt.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),lr=opt.lr)
            discriminator_optimizer = optim.Adam(discriminator.parameters(),lr=opt.lr)
        elif opt.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
            discriminator_optimizer = optim.AdamW(discriminator.parameters(),lr=opt.lr)

        ## Prepare past model
        lr = opt.lr
        best_loss = np.inf
        stop_count = 0
        print('Training begins now.')
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloaders[data_id], 0):
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                t_real_D = (data_id + 1) * torch.ones_like(y_gts.squeeze()).to(device)
                t_fake_D = torch.zeros_like(y_gts.squeeze()).to(device)

                for _ in range(2):

                    optimizer.zero_grad()
                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                    
                    shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                    specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]

                    feature = torch.cat([shared_feature, specific_feature], dim=1)
                    
                    y_outs = model.specific_classifier[data_id](feature)

                    dis_output = discriminator.forward(shared_feature, t_real_D, data_id)
                    adv_loss= nn.CrossEntropyLoss()(dis_output, t_real_D)

                    diff_loss= DiffLoss()(shared_feature, specific_feature)

                    norm_loss = nn.CrossEntropyLoss()(y_outs, y_gts.squeeze())
                    running_loss += norm_loss.item() + 0.05 * adv_loss.item() + 0.1 * diff_loss.item()
                    loss = norm_loss +  0.05 * adv_loss + 0.1 * diff_loss            

                    loss.backward()
                    optimizer.step()
                    if opt.optimizer == 'SGD':
                        scheduler.step()
                for _ in range(1):
                    discriminator_optimizer.zero_grad()

                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

                    shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]

                    dis_real_output = discriminator.forward(shared_feature.detach(), t_real_D, data_id)
                    adv_real_loss= nn.CrossEntropyLoss()(dis_real_output, t_real_D)
                    adv_real_loss.backward(retain_graph=True)

                    z_fake = torch.as_tensor(np.random.normal(0.0, 1.0, (y_gts.squeeze().shape[0], discriminator_params['num_shared_features'])),dtype=torch.float32, device=device)
                    dis_fake_output = discriminator.forward(z_fake, t_real_D, data_id)
                    dis_fake_loss = nn.CrossEntropyLoss()(dis_fake_output, t_fake_D)
                    dis_fake_loss.backward(retain_graph=True)

                    discriminator_optimizer.step()
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
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))

    
    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_frog', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, total_parameters])
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

class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))