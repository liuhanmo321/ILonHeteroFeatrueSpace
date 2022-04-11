import torch
from torch import nn
# from saint.ours_model import SAINT

from data import DataSetCatCon, sub_data_prep
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from tools import count_parameters, classification_scores_hat
from augmentations import embed_data_cont
from augmentations import add_noise
import copy

from prettytable import PrettyTable
import time
import os
import numpy as np

# frac = 1

def hat(opt):
    def criterion(outputs,targets,masks, mask_pre=None):
        reg=0
        count=0
        if mask_pre is not None:
            for m,mp in zip(masks, mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return ce(outputs,targets) + opt.distill_frac*reg, opt.distill_frac*reg

    from saint.hat_model import SAINT

    save_path = opt.result_path

    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,opt.data_name,opt.run_name)
    if opt.task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'

    device = torch.device('cuda:' + opt.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    os.makedirs(modelsave_path, exist_ok=True)

    # Data Set Related

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = sub_data_prep(opt.data_name, opt.dset_seed,opt.dtask, datasplit=[.65, .15, .2], num_tasks=opt.num_tasks, class_inc=opt.class_inc)

    # Model Related

    if y_dims[0] == 2 and opt.task == 'binary':
        nll = nn.NLLLoss().to(device)
    elif y_dims[0] > 2 and  opt.task == 'multiclass':
        nll = nn.NLLLoss().to(device)
    else:
        raise'case not written yet'
    ce = nn.CrossEntropyLoss().to(device)
    

    result_matrix = np.zeros((opt.num_tasks, opt.num_tasks))

    total_time = 0

    mask_pre = None
    mask_back = None
    thres_cosh= 6
    thres_emb = 6
    smax = 1000

    for data_id in range(opt.num_tasks):

        if data_id == 0:
            model = SAINT(
                categories = tuple(cat_dims_group[0]), 
                num_continuous = len(con_idxs_group[0]),                
                dim = opt.embedding_size,                           
                depth = opt.transformer_depth,                       
                heads = opt.attention_heads,
                num_tasks = opt.num_tasks,                         
                attn_dropout = opt.attention_dropout,             
                ff_dropout = opt.ff_dropout,                  
                y_dim = y_dims[0],
                device = device
            )
        else:
            model.cpu()
            model.add_task(tuple(cat_dims_group[data_id]), len(con_idxs_group[data_id]), y_dims[data_id])

        model.to(device)

        # for n, params in model.named_parameters():
        #     print(n)
        # break
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
        
        lr = opt.lr
        best_loss = np.inf
        stop_count = 0
        print('Training begins now.')
        
        for epoch in range(opt.epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            reg_loss = 0.0
            for i, data in enumerate(trainloaders[data_id], 0):
                optimizer.zero_grad() 
                # if i > 0: break
                x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
                
                s = (smax-1/smax)*i/len(trainloaders[data_id].dataset) + 1/smax

                output, masks = model.forward(x_categ_enc, x_cont_enc, data_id, s=s)

                loss, reg = criterion(output, y_gts.squeeze(), masks, mask_pre)
                loss.backward()

                if data_id > 0:
                    for n,p in model.named_parameters():
                        if n in mask_back and p.grad is not None:
                            # print(n)
                            p.grad.data*=mask_back[n]
                
                for n, p in model.task_embeds.named_parameters():
                    # num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                    num=torch.cosh(s*p.data)+1
                    den=torch.cosh(p.data)+1
                    p.grad.data *= smax/s*num/den

                optimizer.step()

                # for n, p in model.task_embeds.named_parameters():
                #     p.data=torch.clamp(p.data,-thres_emb,thres_emb)

                if opt.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()
                reg_loss += reg.item()

            end_time = time.time()
            total_time += end_time - start_time
            
            if epoch%1==0:
                print('[EPOCH %d] Running Loss: %.3f, Reg Loss: %.3f' % (epoch + 1, running_loss, reg_loss))
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

        mask=model.mask(data_id, s=smax)
        for i in range(len(mask)):
            mask[i] = Variable(mask[i].data.clone(),requires_grad=False)
        
        if data_id==0:
            mask_pre=mask
        else:
            for i in range(len(mask_pre)):
                mask_pre[i]=torch.max(mask_pre[i],mask[i])

        # Weights mask
        mask_back={}
        for n,_ in model.named_parameters():
            # print(n)
            vals=model.get_view_for(n,mask_pre)
            if vals is not None:
                mask_back[n]=1-vals
            # else:
            #     print(n)

        for temp_data_id in range(data_id + 1):
            temp_test_accuracy, temp_test_auroc = classification_scores_hat(model, testloaders[temp_data_id], device,  opt.task, temp_data_id, smax)
            result_matrix[temp_data_id, data_id] = temp_test_auroc
        
    print(result_matrix)
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'batch_size', 'smax', 'distill_frac', 'parameters'])
    table.add_row(['%.2f' %total_time, '%.4f' %np.mean(result_matrix[:, -1]), opt.batchsize, smax, opt.distill_frac, total_parameters])
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

