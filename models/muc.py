import torch
from torch import nn
# from saint.ours_model import SAINT

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_muc
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

def muc(opt):
    from saint.muc_model import SAINT

    # save_path = './results/' + '_'.join(saving_list) + '.csv'
    save_path = opt.result_path

    num_side_classifier = opt.side_classifier

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)
    # Data Set Related

    # cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)
    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2])

    lengths = [len(loader.dataset) for loader in trainloaders]
    class temp_opt:
        data_name = 'aps'
        dset_seed = opt.dset_seed
        dtask = opt.dtask
        num_tasks = opt.num_tasks
        class_inc = False
        shuffle = opt.shuffle
        order = opt.order
        num_workers = opt.num_workers

    # aps_cat_dims_group, aps_con_idxs_group, aps_trainloaders, _, _, _ = sub_data_prep('aps', opt.dset_seed,opt.dtask, datasplit=[1., 0., 0.], num_tasks=opt.num_tasks, class_inc=False, length=lengths)
    aps_cat_dims_group, aps_con_idxs_group, aps_trainloaders, _, _, _ = sub_data_prep(temp_opt(), datasplit=[1., 0., 0.], length=lengths)

    print(len(cat_dims_group))
    print(len(aps_cat_dims_group))
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
                num_side_classifier = num_side_classifier,
                extractor_type = opt.extractor_type
            )
            model.add_unlabeled_task(tuple(aps_cat_dims_group[data_id]), len(aps_con_idxs_group[data_id]), y_dims[data_id])
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])
            model.add_unlabeled_task(tuple(aps_cat_dims_group[data_id]), len(aps_con_idxs_group[data_id]), y_dims[data_id])
        
        model.to(device)

        ## Choosing the optimizer

        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        ## Prepare past model
        old_model = copy.deepcopy(model).to(device)
        old_model.eval()
        for params in old_model.parameters():
            params.requires_grad = False
        
        lr = opt.lr
        best_loss = np.inf
        stop_count = 0
        print('Training begins now.')

        for params in model.parameters():
            params.requires_grad = True
        
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad() 

                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                    
                shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                shared_output = model.shared_classifier[data_id](shared_feature)
                
                loss = ce(shared_output, y_gts.squeeze())

                distill_loss = 0
                if not opt.no_distill:
                    with torch.no_grad():
                        # temp_categ_enc, temp_cont_enc = x_categ_enc.detach(), Variable(x_cont_enc.data, requires_grad=False)
                        temp_categ_enc, temp_cont_enc = x_categ_enc.detach(), x_cont_enc.detach()
                        old_shared_feature = old_model.shared_extractor(temp_categ_enc, temp_cont_enc)[:,0,:]
                    shared_feature =  model.shared_extractor(temp_categ_enc, temp_cont_enc)[:, 0, :]
                    for temp_data_id in range(data_id):
                        old_y_outs = old_model.shared_classifier[temp_data_id](old_shared_feature)
                        y_outs = model.shared_classifier[temp_data_id](shared_feature)

                        distill_loss += opt.distill_frac * MultiClassCrossEntropy(y_outs, old_y_outs, T=opt.T) / data_id

                        for side_id in range(num_side_classifier):
                            old_y_outs = old_model.side_classifier[temp_data_id][side_id](old_shared_feature)
                            y_outs = model.side_classifier[temp_data_id][side_id](shared_feature)

                            distill_loss += opt.distill_frac * MultiClassCrossEntropy(y_outs, old_y_outs, T=opt.T) / data_id / num_side_classifier
                        
                    loss += distill_loss
                
                loss.backward()
                optimizer.step()
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
        
        model.eval()
        model.side_classifier.train()

        for params in model.parameters():
            params.requires_grad = False
        
        for params in model.side_classifier.parameters():
            params.requires_grad = True
            
        best_loss, lr = np.inf, opt.lr

        optimizer = optim.AdamW(model.side_classifier.parameters(),lr=lr)
        
        for epoch in range(opt.epochs):
            start_time = time.time()
            running_loss = 0
            for ((i, data), (_, unlabeled_data)) in zip(
                        enumerate(trainloaders[data_id]), enumerate(aps_trainloaders[data_id])):
                loss = 0
                optimizer.zero_grad() 

                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

                unlabeled_x_categ, unlabeled_x_cont = unlabeled_data[0].to(device), unlabeled_data[1].to(device)
                _ , unlabeled_x_categ_enc, unlabeled_x_cont_enc = embed_data_cont(unlabeled_x_categ, unlabeled_x_cont, model, data_id, unlabeled=True)           
                    
                shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                side_outputs = [model.side_classifier[data_id][i](shared_feature) for i in range(num_side_classifier)]
                
                for i in range(num_side_classifier):
                    loss += ce(side_outputs[i], y_gts.squeeze())
                
                unlabeled_shared_feature = model.shared_extractor(unlabeled_x_categ_enc, unlabeled_x_cont_enc)[:,0,:]
                unlabeled_side_outputs = [model.side_classifier[data_id][i](unlabeled_shared_feature) for i in range(num_side_classifier)]
                unlabeled_side_outputs = [torch.softmax(output, dim=1) for output in unlabeled_side_outputs]
                loss_discrepancy = 0
                for i in range(num_side_classifier):
                    for j in range(i+1, num_side_classifier):
                        loss_discrepancy += torch.mean(torch.mean(torch.abs(unlabeled_side_outputs[i] - unlabeled_side_outputs[j]), 1))
                
                loss = loss - opt.discrepancy * loss_discrepancy

                loss.backward()
                optimizer.step()

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
                    optimizer = optim.AdamW(model.side_classifier.parameters(),lr=lr)


        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_muc(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
        
    
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))


    emb_parameters = count_parameters(model.embeds) + count_parameters(model.simple_MLP)
    unlabeled_emb_parameters = count_parameters(model.unlabeled_embeds) + count_parameters(model.unlabeled_simple_MLP)

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'dist_frac', 'side_classifier', 'discrepancy', 'parameters', 'emb params', 'ulb params'])
    table.add_row(['%.2f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, opt.distill_frac, num_side_classifier, opt.discrepancy, total_parameters, emb_parameters, unlabeled_emb_parameters])
    print(table)

    print('===========================================================================')
    # if not opt.hyper_search:
    #     with open(save_path, 'a+') as f:
    #         f.write(table.get_string())
    #         f.write('\n')
    #         f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
    #         f.write(str(result_matrix))
    #         f.write('\n')
    #         f.write('====================================================================\n\n')
    #         f.close()       
    # else:
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(result_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close() 
        # return  np.mean(result_matrix[:, -1])

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])

