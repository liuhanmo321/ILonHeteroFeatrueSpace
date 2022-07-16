import torch
from torch import nn
# from saint.ours_model import SAINT

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_cont
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

def dmc(opt):
    from saint.dmc_model import SAINT

    save_path = opt.result_path

    num_side_classifier = opt.side_classifier

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt, datasplit=[.65, .15, .2])

    lengths = [len(loader.dataset) for loader in trainloaders]
    class temp_opt:
        data_name = opt.data_name
        dset_seed = 100
        dtask = 100
        num_tasks = opt.num_tasks
        class_inc = False
        shuffle = opt.shuffle
        order = opt.order
        num_workers = opt.num_workers

    aps_cat_dims_group, aps_con_idxs_group, aps_trainloaders, _, _, _ = sub_data_prep(temp_opt(), datasplit=[1., 0., 0.], length=lengths)

    print(len(cat_dims_group))
    print(len(aps_cat_dims_group))
    # Model Related

    feat_idx = 0 if opt.extractor_type == 'transformer' else -1

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

        ## Prepare past model
        old_model = copy.deepcopy(model).to(device)
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False
        
        if data_id > 0:
            for layer in model.modules():
                if hasattr(layer, 'reset_parameters'):
                    # print(layer)
                    layer.reset_parameters()
            # for m in model.modules():
            #     m.reset_parameters()
        new_model = copy.deepcopy(model).to(device)

        ## Choosing the optimizer

        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=opt.lr)

        lr = opt.lr
        best_loss = np.inf
        stop_count = 0
        print('Training begins now.')

        for param in model.parameters():
            param.requires_grad = True

        if data_id > 0:
            for param in new_model.parameters():
                param.requires_grad = True            
            for epoch in range(opt.epochs):
                start_time = time.time()
                new_model.train()
                running_loss = 0

                for i, data in enumerate(trainloaders[data_id], 0):
                    new_optimizer.zero_grad()
                    x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, new_model, data_id)

                    shared_feature = new_model.shared_extractor(x_categ_enc, x_cont_enc)[:,feat_idx,:]
                    shared_output = new_model.shared_classifier[data_id](shared_feature)

                    loss = ce(shared_output, y_gts.squeeze())

                    loss.backward()
                    new_optimizer.step()

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
                        new_optimizer = optim.AdamW(new_model.parameters(),lr=lr)                  
                
            new_model.eval()
            for param in new_model.parameters():
                param.requires_grad = False

            lr = opt.lr
            best_loss = np.inf
            stop_count = 0
            for epoch in range(opt.epochs):
                running_loss = 0
                start_time = time.time()

                model.train()
                for _, unlabeled_data in enumerate(aps_trainloaders[data_id], 0):
                # for _, unlabeled_data in enumerate(trainloaders[data_id], 0):
                    loss = 0
                    optimizer.zero_grad()

                    unlabeled_x_categ, unlabeled_x_cont = unlabeled_data[0].to(device), unlabeled_data[1].to(device)
                    _ , unlabeled_x_categ_enc, unlabeled_x_cont_enc = embed_data_cont(unlabeled_x_categ, unlabeled_x_cont, model, data_id, unlabeled=True)
                    # _ , unlabeled_x_categ_enc, unlabeled_x_cont_enc = embed_data_cont(unlabeled_x_categ, unlabeled_x_cont, model, data_id)           
                        
                    with torch.no_grad():
                        old_shared_feature = old_model.shared_extractor(unlabeled_x_categ_enc, unlabeled_x_cont_enc)[:, feat_idx, :]
                        old_outputs = [old_model.shared_classifier[temp_id](old_shared_feature) for temp_id in range(data_id)]
                        new_shared_feature = new_model.shared_extractor(unlabeled_x_categ_enc, unlabeled_x_cont_enc)[:, feat_idx, :]
                        new_output = new_model.shared_classifier[data_id](new_shared_feature)
                        
                        old_outputs = [old_output - old_output.mean(0) for old_output in old_outputs]
                        old_outputs = torch.cat(old_outputs, dim=1)
                        new_output -= new_output.mean(0)
                        temp_outputs = torch.cat([old_outputs, new_output], dim=1)
                        # temp_outputs -= temp_outputs.mean(0)                
                    
                    shared_feature = model.shared_extractor(unlabeled_x_categ_enc, unlabeled_x_cont_enc)[:, feat_idx, :]
                    outputs = [model.shared_classifier[temp_id](shared_feature) for temp_id in range(data_id + 1)]                
                    outputs = [output - output.mean(0) for output in outputs]
                    outputs = torch.cat(outputs, dim=1)
                    

                    # loss = MultiClassCrossEntropy(outputs, temp_outputs, T=2)
                    loss = nn.MSELoss()(outputs, temp_outputs)

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
                        lr = lr / 100
                        stop_count = 0
                        if lr < opt.lr_lower_bound:
                            break
                        optimizer = optim.AdamW(model.parameters(),lr=lr)     

            for temp_id in range(data_id):
                model.embeds[temp_id] = copy.deepcopy(old_model.embeds[temp_id])
                model.simple_MLP[temp_id] = copy.deepcopy(old_model.simple_MLP[temp_id])
            model.embeds[-1] = copy.deepcopy(new_model.embeds[-1])
            model.simple_MLP[-1] = copy.deepcopy(new_model.simple_MLP[-1])
        
        else:
            for epoch in range(opt.epochs):
                start_time = time.time()
                model.train()
                running_loss = 0

                for i, data in enumerate(trainloaders[data_id], 0):
                    optimizer.zero_grad()
                    x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

                    shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,feat_idx,:]
                    shared_output = model.shared_classifier[data_id](shared_feature)

                    loss = ce(shared_output, y_gts.squeeze())

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

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
        
    
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    avg_forgetting = np.mean(np.array([result_matrix[temp_id, temp_id] - result_matrix[temp_id, opt.num_tasks-1] for temp_id in range(opt.num_tasks)]))


    emb_parameters = count_parameters(model.embeds) + count_parameters(model.simple_MLP)
    unlabeled_emb_parameters = count_parameters(model.unlabeled_embeds) + count_parameters(model.unlabeled_simple_MLP)

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_auc', 'avg_forg', 'dist_frac', 'parameters', 'emb params', 'ulb params'])
    table.add_row(['%.2f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %avg_forgetting, opt.distill_frac, total_parameters, emb_parameters, unlabeled_emb_parameters])
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
        # return  np.mean(result_matrix[:, -1])

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])

