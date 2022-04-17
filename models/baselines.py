import torch
from torch import nn
# from saint import SAINT

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_ours, classification_scores_cont, classification_scores_specific_only
from augmentations import embed_data_cont
import copy

import os
import numpy as np

from prettytable import PrettyTable
import time

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True)

def baseline_finetune(opt):

    from saint.ours_model import SAINT

    save_path = './results/' + '_'.join([opt.method, opt.data_name]) + '.csv'
    save_path = opt.result_path

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)    
    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related

    ce = nn.CrossEntropyLoss().to(device)

    total_time = 0
    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    shared_matrix = np.zeros((opt.num_tasks, opt.num_tasks))
    specific_matrix = np.zeros((opt.num_tasks, opt.num_tasks))

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
                y_dim = y_dims[0]
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
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad()
                
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                # We are converting the data to embeddings in the next step
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

                shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                shared_output = model.shared_classifier[data_id](shared_feature)
                # shared_p = torch.softmax(shared_output, dim=1)

                specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
                specific_output = model.specific_classifier[data_id](specific_feature)
                # specific_p = torch.softmax(specific_output, dim=1)
                
                loss = (ce(specific_output, y_gts.squeeze()) + ce(shared_output, y_gts.squeeze())) / 2

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # print(running_loss)
            end_time = time.time()
            total_time += end_time - start_time

            if epoch%1==0:
                model.eval()
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
                model.train()

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_ours(model, testloaders[temp_data_id], device,  opt.task, temp_data_id, opt.alpha)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            shared_matrix[temp_data_id, data_id] = temp_test_auroc
            temp_test_accuracy, temp_test_auroc = classification_scores_specific_only(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            specific_matrix[temp_data_id, data_id] = temp_test_auroc


    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'alpha', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.alpha, total_parameters])

    result_table = PrettyTable(['cmb auc', 'shared auc', 'specific auc'])
    result_table.add_row(['%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %np.mean(shared_matrix[:, -1]), '%.4f' %np.mean(specific_matrix[:, -1])])
    print(table)
    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write(result_table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(result_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close() 

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])



def baseline_joint(opt):

    from saint.ours_model import SAINT

    save_path = './results/' + '_'.join([opt.method, opt.data_name]) + '.csv'
    save_path = opt.result_path

    alpha = opt.alpha


    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)    
    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related
    ce = nn.CrossEntropyLoss().to(device)

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
                depth = opt.transformer_depth,                       
                heads = opt.attention_heads,                         
                attn_dropout = opt.attention_dropout,             
                ff_dropout = opt.ff_dropout,                  
                y_dim = y_dims[0]
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        ## Choosing the optimizer
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

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
                optimizer.zero_grad()
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

                shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                shared_output = model.shared_classifier[data_id](shared_feature)
                shared_p = torch.softmax(shared_output, dim=1)

                specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
                specific_output = model.specific_classifier[data_id](specific_feature)
                specific_p = torch.softmax(specific_output, dim=1)
                
                loss = (ce(specific_output, y_gts.squeeze()) + ce(shared_output, y_gts.squeeze()))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                torch.cuda.empty_cache()
            
            for temp_id in range(data_id):
                for i, data in enumerate(trainloaders[temp_id], 0):
                    optimizer.zero_grad()
                    x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, temp_id)           
                    shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
                    shared_output = model.shared_classifier[temp_id](shared_feature)
                    shared_p = torch.softmax(shared_output, dim=1)

                    specific_feature = model.specific_extractor[temp_id](x_categ_enc, x_cont_enc)[:,0,:]
                    specific_output = model.specific_classifier[temp_id](specific_feature)
                    specific_p = torch.softmax(specific_output, dim=1)
                    
                    loss = (ce(specific_output, y_gts.squeeze()) + ce(shared_output, y_gts.squeeze()))

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    torch.cuda.empty_cache()
            # print(running_loss)
            end_time = time.time()
            total_time += end_time - start_time

            if epoch%1==0:
                model.eval()
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
                model.train()

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_ours(model, testloaders[temp_data_id], device,  opt.task, temp_data_id, opt.alpha)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            shared_matrix[temp_data_id, data_id] = temp_test_auroc
            temp_test_accuracy, temp_test_auroc = classification_scores_specific_only(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            specific_matrix[temp_data_id, data_id] = temp_test_auroc

        
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    
    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'lr', 'alpha', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.lr, opt.alpha, total_parameters])

    result_table = PrettyTable(['cmb auc', 'shared auc', 'specific auc'])
    result_table.add_row(['%.4f' %np.mean(result_matrix[:, -1]), '%.4f' %np.mean(shared_matrix[:, -1]), '%.4f' %np.mean(specific_matrix[:, -1])])
    print(table)
    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write(result_table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(result_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close() 

    if opt.hyper_search:
        return  np.mean(result_matrix[:, -1])



def baseline_ord_joint(opt):

    from saint.ours_model import SAINT
    
    save_path = './results/' + '_'.join([opt.method, opt.data_name]) + '.csv'
    save_path = opt.result_path

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)
    
    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

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
        lr = opt.lr
        best_loss = np.inf
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
                shared_output = model.shared_extractor(x_categ_enc, x_cont_enc)
                shared_feature = shared_output[:,0,:]
                
                y_outs = model.shared_classifier[data_id](shared_feature)
                loss = ce(y_outs,y_gts.squeeze())

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                torch.cuda.empty_cache()

            for temp_id in range(data_id):
                for i, data in enumerate(trainloaders[temp_id], 0):
                    optimizer.zero_grad()
                    x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)

                    _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, temp_id)           
                    shared_output = model.shared_extractor(x_categ_enc, x_cont_enc)
                    shared_feature = shared_output[:,0,:]
                    
                    y_outs = model.shared_classifier[temp_id](shared_feature)
                    loss = ce(y_outs,y_gts.squeeze())

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    torch.cuda.empty_cache()

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
            # if epoch%5==0:
            #     print('running_loss is:', running_loss)
            #     model.eval()
            #     with torch.no_grad():
            #         sum_acc, sum_auroc = 0, 0
            #         for temp_data_id in range(data_id + 1):
            #             accuracy, auroc = classification_scores_cont(model, validloaders[temp_data_id], device, opt.task, temp_data_id)
            #             sum_acc += accuracy
            #             sum_auroc += auroc

            #         accuracy = sum_acc / (data_id + 1)
            #         auroc = sum_auroc / (data_id + 1)
            #         print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
            #             (epoch + 1, accuracy,auroc ))

            #         if opt.active_log:
            #             wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })       
            #         if opt.task =='multiclass':
            #             if accuracy > best_valid_accuracy:
            #                 best_valid_accuracy = accuracy
            #                 torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            #         else:
            #             # if accuracy > best_valid_accuracy:
            #             #     best_valid_accuracy = accuracy
            #             # if auroc > best_valid_auroc:
            #             #     best_valid_auroc = auroc   
            #             if running_loss < best_loss:
            #                 best_loss = running_loss          
            #                 # torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            #                 stop_count = 0
            #             else:
            #                 stop_count += 1
            #     model.train()

            #     if stop_count == opt.earlystop:
            #         break

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_cont(model, testloaders[temp_data_id], device,  opt.task, temp_data_id)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
        
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'lr', 'parameters'])
    table.add_row([total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.lr, total_parameters])
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

