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


def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True)

def baseline_shared_only(opt):
    if opt.shrink:
        from saint.ours_model import SAINT
    else:
        from saint.base_model import SAINT
    save_path = opt.result_path

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    
    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed, opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related

    ce = nn.CrossEntropyLoss().to(device)

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
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
                condition = 'shared_only'
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        ## Choosing the optimizer
        
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        ## Prepare past model
        old_shared_extractor = copy.deepcopy(model.shared_extractor).to(device)
        old_shared_classifier = copy.deepcopy(model.shared_classifier).to(device)

        for params in old_shared_classifier.parameters():
            params.requires_grad = False
        for params in old_shared_extractor.parameters():
            params.requires_grad = False
        
        lr = opt.lr
        best_loss = np.inf

        stop_count = 0
        print('Training begins now.')
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            dist_loss = 0
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad()
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                shared_output = model.shared_extractor(x_categ_enc, x_cont_enc)
                shared_feature = shared_output[:,0,:]
                
                y_outs = model.shared_classifier[data_id](shared_feature)
                loss = ce(y_outs,y_gts.squeeze())
                
                if not opt.no_distill:
                    with torch.no_grad():
                        old_shared_feature = old_shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                        old_temp_outs = [old_shared_classifier[temp_data_id](old_shared_feature) for temp_data_id in range(data_id)]
                    
                    for temp_data_id in range(data_id):
                        temp_outs = model.shared_classifier[temp_data_id](shared_feature)

                        loss += opt.distill_frac * MultiClassCrossEntropy(temp_outs, old_temp_outs[temp_data_id], T=opt.T) / data_id
                        dist_loss += (opt.distill_frac * MultiClassCrossEntropy(temp_outs, old_temp_outs[temp_data_id], T=opt.T) / data_id).item()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            end_time = time.time()
            total_time += end_time - start_time
            
            if epoch%1==0:
                model.eval()
                print('[EPOCH %d] Running Loss: %.3f Dist Loss: %.3f' % (epoch + 1, running_loss, dist_loss))
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

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device, opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc

    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'T', 'dist_frac', 'parameters'])
    table.add_row(['%.4f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, opt.T, opt.distill_frac, total_parameters])
    print(table)
    print('===========================================================================')

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])
    else:
        with open(save_path, 'a+') as f:
            f.write(table.get_string())
            f.write('\n')
            f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
            f.write(str(result_matrix))
            f.write('\n')
            f.write('====================================================================\n\n')
            f.close()


def baseline_specific_only(opt):
    # if opt.shrink:
    from saint.ours_model import SAINT
    # else:
    #     from saint.base_model import SAINT

    save_path = opt.result_path


    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)
    
    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related

    ce = nn.CrossEntropyLoss().to(device)
    nll = nn.NLLLoss().to(device)

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    dis_score_list = []
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
                condition = 'specific_only'
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

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
            total_dis_loss = 0.0
            total_dis_score = 0.0

            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad()

                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                specific_output = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)
                specific_feature = specific_output[:,0,:]
                
                y_outs = model.specific_classifier[data_id](specific_feature)
                loss = ce(y_outs,y_gts.squeeze())
                specific_p = torch.softmax(y_outs, dim=1)

                if data_id > 0:
                    with torch.no_grad():
                        specific_features = [model.specific_extractor[temp_id](x_categ_enc, x_cont_enc)[:,0,:] for temp_id in range(data_id + 1)]
                        specific_outputs = [model.specific_classifier[data_id](specific_features[temp_id]) for temp_id in range(data_id +1)]
                        specific_p = [torch.softmax(output, dim=1) for output in specific_outputs]
                        label_p = [-nn.NLLLoss(reduction='none')(p, y_gts.squeeze()) for p in specific_p]

                        # max calculation
                        max_label_p = torch.max(torch.stack(label_p[:-1], 1), 1).values
                        temp_dis_score = label_p[-1] - max_label_p

                        total_dis_score += torch.sum(temp_dis_score)
                        temp_dis_score = torch.exp(-temp_dis_score * opt.gamma)
                        # print("after exp temp_dis_score", temp_dis_score)
                        temp_dis_score = F.normalize(temp_dis_score, dim=0)
                        # print("after norm temp_dis_score", temp_dis_score)
                        dis_score = torch.reshape(temp_dis_score, (y_gts.shape[0], 1))         

                    dis_output = dis_score * torch.log_softmax(y_outs, dim=1)
                    dis_loss = nll(dis_output, y_gts.squeeze())
                    
                    total_dis_loss += dis_loss.item()
                    if not opt.no_discrim:
                        loss += opt.beta * dis_loss            

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            end_time = time.time()
            total_time += end_time - start_time

            if epoch%1==0:
                print('[EPOCH %d] Running Loss: %.3f, total_dis_score %f' % (epoch + 1, running_loss, total_dis_score))
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

        dis_score_list.append(total_dis_score)
        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_specific_only(model, testloaders[temp_data_id], device, opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc

    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'lr', 'beta', 'gamma', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, opt.lr, opt.beta, opt.gamma, total_parameters])
    print(table)
    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(result_matrix))
        f.write('\n')
        f.write('dis_score\n')
        f.write(str(dis_score_list))
        f.write('\n====================================================================\n\n')
        f.close()

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])
