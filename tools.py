import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from augmentations import embed_data_mask, embed_data_cont
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont = data[0].to(device), data[1].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, model)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def classification_scores_cont(model, dloader, device, task, data_id):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
            reps = model.shared_extractor(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.shared_classifier[data_id](y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc


def classification_scores_specific_only(model, dloader, device, task, data_id):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
            reps = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.specific_classifier[data_id](y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

# def classification_scores_ours(model, dloader, device, task, data_id, alpha):
#     model.eval()
#     m = nn.Softmax(dim=1)
#     y_test = torch.empty(0).to(device)
#     y_pred = torch.empty(0).to(device)
#     prob = torch.empty(0).to(device)
#     with torch.no_grad():
#         for i, data in enumerate(dloader, 0):
#             x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
#             _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

#             shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
#             shared_output = model.shared_classifier[data_id](shared_feature)
#             shared_p = torch.softmax(shared_output, dim=1)

#             specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
#             specific_output = model.specific_classifier[data_id](specific_feature)
#             specific_p = torch.softmax(specific_output, dim=1)
            
#             y_outs = alpha * shared_p + (1-alpha) * specific_p
#             # import ipdb; ipdb.set_trace()   
#             y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
#             y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
#             if task == 'binary':
#                 prob = torch.cat([prob, m(y_outs)[:,-1].float()],dim=0)
     
#     correct_results_sum = (y_pred == y_test).sum().float()
#     acc = correct_results_sum/y_test.shape[0]*100
#     auc = 0
#     if task == 'binary':
#         auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
#     return acc.cpu().numpy(), auc

def classification_scores_ours(model, dloader, device, task, data_id, alpha):
    model.eval()
    # m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

            shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
            shared_output = model.shared_classifier[data_id](shared_feature)
            shared_p = torch.softmax(shared_output, dim=1)

            specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
            specific_output = model.specific_classifier[data_id](specific_feature)
            specific_p = torch.softmax(specific_output, dim=1)
            
            y_outs = alpha * shared_p + (1-alpha) * specific_p
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob, y_outs[:,-1].float()],dim=0) # modified the possibility
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def classification_scores_acl(model, dloader, device, task, data_id, alpha):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

            shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]

            specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]

            feature = torch.cat([shared_feature, specific_feature], dim=1)
            
            y_outs = model.specific_classifier[data_id](feature)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def valid_loss_ours(model, dloader, device, data_id, opt, train_type=None, old_shared_extractor=None):
    nll = nn.NLLLoss().to(device)
    ce = nn.CrossEntropyLoss().to(device)
    loss = 0
    for i, data in enumerate(dloader, 0):
        x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
        # y_gts = y_gts.squeeze()
        _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)

        if train_type != 'specific_only':
            shared_feature = model.shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]
            shared_output = model.shared_classifier[data_id](shared_feature)
            loss += ce(shared_output, y_gts.squeeze()) / 2

        if train_type != 'shared_only':
            specific_feature = model.specific_extractor[data_id](x_categ_enc, x_cont_enc)[:,0,:]
            specific_output = model.specific_classifier[data_id](specific_feature)
            specific_p = torch.softmax(specific_output, dim=1)
            loss += ce(specific_output, y_gts.squeeze()) / 2

        if data_id > 0 and not opt.no_discrim and train_type != 'shared_only':
            with torch.no_grad():
                specific_features = [model.specific_extractor[temp_id](x_categ_enc, x_cont_enc)[:,0,:] for temp_id in range(data_id + 1)]
                specific_outputs = [model.specific_classifier[data_id](specific_features[temp_id]) for temp_id in range(data_id +1)]
                specific_p = [torch.softmax(output, dim=1) for output in specific_outputs]
                label_p = [-nn.NLLLoss(reduction='none')(p, y_gts.squeeze()) for p in specific_p]

                max_label_p = torch.max(torch.stack(label_p[:-1], 1), 1).values
                temp_dis_score = label_p[-1] - max_label_p

                temp_dis_score = torch.exp(-temp_dis_score * opt.gamma)
                temp_dis_score = F.normalize(temp_dis_score, dim=0)
                dis_score = torch.reshape(temp_dis_score, (y_gts.shape[0], 1))

            dis_output = dis_score * torch.log_softmax(specific_output, dim=1)
            dis_loss = nll(dis_output, y_gts.squeeze())
            loss += opt.beta * dis_loss            
        
        if not opt.no_distill and train_type != 'specific_only':
            with torch.no_grad():
                old_shared_feature = old_shared_extractor(x_categ_enc, x_cont_enc)[:,0,:]                    
            for temp_data_id in range(data_id):
                old_y_outs = model.shared_classifier[temp_data_id](old_shared_feature)
                y_outs = model.shared_classifier[temp_data_id](shared_feature)

                loss += MultiClassCrossEntropy(y_outs, old_y_outs, T=2) / data_id
    return loss.item()

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True)

def classification_scores_muc(model, dloader, device, task, data_id):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
            reps = model.shared_extractor(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            side_y_outs = [classifier(y_reps) for classifier in model.side_classifier[data_id]]            
            y_outs = 0
            for temp_out in side_y_outs:
                y_outs += temp_out
            # y_outs /= len(side_y_outs)
            # y_outs = model.shared_classifier[data_id](y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def classification_scores_pnn(model, dloader, device, task, data_id):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
            y_outs = model.forward(x_categ_enc, x_cont_enc, data_id)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def classification_scores_hat(model, dloader, device, task, data_id, s=400):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device),data[2].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_cont(x_categ, x_cont, model, data_id)           
            y_outs, _ = model.forward(x_categ_enc, x_cont_enc, data_id, s=s)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc