import torch
from torch import nn

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_cont, classification_scores_specific_only, valid_loss_ours
from augmentations import embed_data_cont
import copy

from prettytable import PrettyTable
import time

import os
import numpy as np

def fisher_matrix_diag(data_id, dataloader, model, old_model, fisher, device):
    # Init
    temp_fisher={}
    for n,p in model.shared_extractor.named_parameters():
        temp_fisher[n]=0*p.data
    # Compute
    model.train()
    feat_idx = 0 if model.extractor_type == 'transformer' else -1
    for i, data in enumerate(dataloader, 0):
        x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
        # print(x_categ.shape)
        _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id) 
        model.zero_grad()
        shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,feat_idx,:]
        shared_output = model.shared_classifier[data_id](shared_feature)
        loss = nn.CrossEntropyLoss().to(device)(shared_output, y_gts.squeeze())
        if data_id > 0:
            for (name, param),(_, param_old) in zip(model.shared_extractor.named_parameters(),old_model.shared_extractor.named_parameters()):
                loss+=torch.sum(fisher[name]*(param_old-param).pow(2))/2
        loss.backward()
        # Get gradients
        for n,p in model.shared_extractor.named_parameters():
            if p.grad is not None:
                temp_fisher[n]+=x_categ.shape[0] * p.grad.data.pow(2)
    # Mean
    for n,_ in model.shared_extractor.named_parameters():
        temp_fisher[n]=temp_fisher[n]/len(dataloader.dataset)
        temp_fisher[n]=Variable(temp_fisher[n],requires_grad=False)
    return temp_fisher

def baseline_shared_only_ewc(opt):   
    if opt.shrink:
        from saint.ours_model import SAINT
    else:
        from saint.base_model import SAINT
    # from saint.base_model import SAINT

    save_path = opt.result_path

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)
    # Data Set Related

    # cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)
    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2])

    # Model Related

    ce = nn.CrossEntropyLoss().to(device)

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    total_time = 0
    feat_idx = 0 if opt.extractor_type == 'transformer' else -1
    fisher = {}

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
                condition = 'shared_only',
                extractor_type = opt.extractor_type
            )
            old_model = None
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])
            
            old_model = copy.deepcopy(model)
            old_model.eval()
            for params in old_model.parameters():
                params.requires_grad = False
            old_model.to(device)

        model.to(device)

        ## Choosing the optimizer        
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        
        lr = opt.lr
        best_loss = np.inf
        stop_count = 0
        print('Training begins now.')
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            running_fisher_loss = 0.0
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad()
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                shared_output = model.shared_extractor(x_categ_enc, x_cont_enc)
                shared_feature = shared_output[:,feat_idx,:]
                
                y_outs = model.shared_classifier[data_id](shared_feature)
                loss = ce(y_outs,y_gts.squeeze())

                fisher_loss = 0.0
                if not opt.no_distill and data_id > 0:
                    for (name, param),(_, param_old) in zip(model.shared_extractor.named_parameters(),old_model.shared_extractor.named_parameters()):
                        fisher_loss+=torch.sum(fisher[name]*(param_old-param).pow(2))/2
                
                loss = loss + 5000 * fisher_loss * opt.distill_frac
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if data_id > 0 and not opt.no_distill:
                    running_fisher_loss += 5000 * fisher_loss.item()
            end_time = time.time()
            total_time += end_time - start_time
            
            if epoch%1==0:
                model.eval()
                print('[EPOCH %d] Running Loss: %.3f Fisher Loss: %.3f' % (epoch + 1, running_loss, running_fisher_loss))
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
                model.train()
        
        # Fisher ops
        if not opt.no_distill:
            if data_id>0:
                fisher_old={}
                for n,_ in model.shared_extractor.named_parameters():
                    fisher_old[n]=fisher[n].clone()
            
            fisher = fisher_matrix_diag(data_id, trainloaders[data_id], model, old_model, fisher, device)
            
            if data_id>0:
                for n,_ in model.shared_extractor.named_parameters():
                    fisher[n]=(fisher[n]+fisher_old[n]*data_id)/(data_id+1)

        # print(fisher)       

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device, opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc

    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))


    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'parameters'])
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