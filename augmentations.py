import torch
import numpy as np

def embed_data_cont(x_categ, x_cont, model, data_id, unlabeled=False):
    device = x_cont.device

    if not unlabeled:
        x_categ = x_categ + model.categories_offset[data_id].type_as(x_categ)
        x_categ_enc = model.embeds[data_id](x_categ)
        n1,n2 = x_cont.shape
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous[data_id]):
            x_cont_enc[:,i,:] = model.simple_MLP[data_id][i](x_cont[:,i])

        x_cont_enc = x_cont_enc.to(device)    
    else:
        x_categ = x_categ + model.unlabeled_categories_offset[data_id].type_as(x_categ)
        x_categ_enc = model.unlabeled_embeds[data_id](x_categ)
        n1,n2 = x_cont.shape

        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.unlabeled_num_continuous[data_id]):
            x_cont_enc[:,i,:] = model.unlabeled_simple_MLP[data_id][i](x_cont[:,i])

        x_cont_enc = x_cont_enc.to(device)    

    return x_categ, x_categ_enc, x_cont_enc