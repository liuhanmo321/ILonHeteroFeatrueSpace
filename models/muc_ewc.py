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
from augmentations import add_noise
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
    num_side_classifier = len(model.side_classifier[0])
    for i, data in enumerate(dataloader, 0):
        x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
        _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id) 
        
        model.zero_grad()
        shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
        shared_output = model.shared_classifier[data_id](shared_feature)
        loss = nn.CrossEntropyLoss().to(device)(shared_output, y_gts.squeeze())
        side_outputs = [model.side_classifier[data_id][i](shared_feature) for i in range(num_side_classifier)]
                
        for i in range(num_side_classifier):
            loss += nn.CrossEntropyLoss().to(device)(side_outputs[i], y_gts.squeeze())
        
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

def muc_ewc(opt):
    from saint.muc_model import SAINT
        # opt.transformer_depth = 3
        # opt.attention_heads = 4

    # save_path = './results/' + '_'.join(saving_list) + '.csv'
    save_path = opt.result_path

    num_side_classifier = opt.side_classifier

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # torch.manual_seed(opt.set_seed)

    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    lengths = [len(loader.dataset) for loader in trainloaders]
    aps_cat_dims_group, aps_con_idxs_group, aps_trainloaders, _, _, _ = sub_data_prep('aps', opt.dset_seed,opt.dtask, datasplit=[1., 0., 0.], num_tasks=opt.num_tasks, class_inc=False, length=lengths)

    # Model Related

    if y_dims[0] == 2 and opt.task == 'binary':
        # opt.task = 'binary'
        criterion = nn.NLLLoss().to(device)
    elif y_dims[0] > 2 and  opt.task == 'multiclass':
        # opt.task = 'multiclass'
        criterion = nn.NLLLoss().to(device)
    else:
        raise'case not written yet'
    ce = nn.CrossEntropyLoss().to(device)
    

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))

    total_time = 0
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
                num_side_classifier = num_side_classifier
            )
            model.add_unlabeled_task(tuple(aps_cat_dims_group[data_id]), len(aps_con_idxs_group[data_id]), y_dims[data_id])
            old_model = None
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])
            model.add_unlabeled_task(tuple(aps_cat_dims_group[data_id]), len(aps_con_idxs_group[data_id]), y_dims[data_id])
            
            old_model = copy.deepcopy(model)
            old_model.eval()
            for params in old_model.parameters():
                params.requires_grad = False
            old_model.to(device)
        
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

                fisher_loss = 0.0
                if not opt.no_distill and data_id > 0:
                    for (name, param),(_, param_old) in zip(model.shared_extractor.named_parameters(),old_model.shared_extractor.named_parameters()):
                        fisher_loss+=torch.sum(fisher[name]*(param_old-param).pow(2))/2
                
                loss = loss + 5000 * fisher_loss
                
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

        if not opt.no_distill:
            if data_id>0:
                fisher_old={}
                for n,_ in model.shared_extractor.named_parameters():
                    fisher_old[n]=fisher[n].clone()
            
            fisher = fisher_matrix_diag(data_id, trainloaders[data_id], model, old_model, fisher, device)
            
            if data_id>0:
                for n,_ in model.shared_extractor.named_parameters():
                    fisher[n]=(fisher[n]+fisher_old[n]*data_id)/(data_id+1)

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

    emb_parameters = count_parameters(model.embeds) + count_parameters(model.simple_MLP)
    unlabeled_emb_parameters = count_parameters(model.unlabeled_embeds) + count_parameters(model.unlabeled_simple_MLP)
    
    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'dist_frac', 'side_classifier', 'discrepancy', 'parameters', 'emb params', 'ulb params'])
    table.add_row(['%.2f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.distill_frac, num_side_classifier, opt.discrepancy, total_parameters, emb_parameters, unlabeled_emb_parameters])
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

