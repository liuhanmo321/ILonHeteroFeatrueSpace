import torch
from torch import nn
from saint import SAINT
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

from models.ablations import baseline_shared_only, baseline_specific_only
from models.ablations_ewc import baseline_shared_only_ewc
from models.ours import ours
from models.baselines import baseline_ord_joint, baseline_joint, baseline_finetune
from models.acl import baseline_acl
# from models.test import ours_test/
from models.ours_ewc import ours_ewc
from models.muc import muc
from models.pnn import pnn
from models.muc_ewc import muc_ewc

from prettytable import PrettyTable

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('-data_name', default=None, type=str)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', default='binary', type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=8, type=int)
parser.add_argument('--transformer_depth', default=4, type=int)
parser.add_argument('--attention_heads', default=2, type=int) 
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)

parser.add_argument('-method', default='shared_only', type=str)
parser.add_argument('-num_tasks', default=3, type=int)
parser.add_argument('-earlystop', default=1, type=int)
parser.add_argument('-no_distill', action = 'store_true')
parser.add_argument('-no_discrim', action = 'store_true')
parser.add_argument('-hyper_search', action = 'store_true')
parser.add_argument('-shrink', action = 'store_true')
parser.add_argument('-class_inc', action = 'store_true')

parser.add_argument('-lr_lower_bound', default=0.00001, type=float)
parser.add_argument('-patience', default=5, type=int)

parser.add_argument('-comment', default='', type=str)

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])

parser.add_argument('-lr', default=0.0001, type=float)
parser.add_argument('-epochs', default=300, type=int)
parser.add_argument('-batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('-set_seed', default= 1 , type=int)
parser.add_argument('-dset_seed', default= 5 , type=int)
parser.add_argument('-shuffle', action = 'store_true')

parser.add_argument('-alpha', default=0.2, type=float)
parser.add_argument('-beta', default=0.1, type=float)
parser.add_argument('-gamma', default=5, type=float)
parser.add_argument('-sp_frac', default=0.5, type=float)
parser.add_argument('-distill_frac', default=1, type=float)
parser.add_argument('-T', default=2, type=float)

parser.add_argument('-result_path', default='', type=str)
parser.add_argument('-debug', action='store_true')
parser.add_argument('-save_model', action='store_true')
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])
parser.add_argument('-dtask', default='clf', type=str)

parser.add_argument('-side_classifier', default=4, type=int)
parser.add_argument('-discrepancy', default=1, type=int)

opt = parser.parse_args()

seed_dict = {'aps': 1, 'bank': 1, 'blast_char': 3, 'income': 2, 'shoppers': 4, 'shrutime': 2, 'volkert': 3, 'mix': 1}
# seed_dict = {'aps': 1, 'bank': 2, 'blast_char': 2, 'income': 3, 'shoppers': 3, 'shrutime': 3, 'volkert': 2, 'mix': 2}
# seed_dict = {'aps': 1, 'bank': 3, 'blast_char': 1, 'income': 1, 'shoppers': 2, 'shrutime': 1, 'volkert': 1, 'mix': 3}

if opt.data_name == 'volkert' and opt.class_inc:
    opt.num_tasks = 5

if opt.debug:
    opt.epochs = 1

if __name__ == '__main__':

    # torch.manual_seed(opt.set_seed)
    torch.manual_seed(1)

    saving_list = [opt.method]
    if opt.no_distill:
        saving_list.append('nodistill')
    if opt.no_discrim:
        saving_list.append('nodiscrim')
    if opt.shrink:
        saving_list.append('shrink')
    if opt.hyper_search:
        saving_list.append('hs')
    if opt.class_inc:
        saving_list.append('cls')
    if opt.comment != '':   
        saving_list.append(opt.comment)

    if 'joint' in opt.method:
        opt.lr_lower_bound = opt.lr / 10

    # if opt.hyper_search:
    #     opt.set_seed = seed_dict[opt.data_name]
    #     opt.dset_seed = 6 - seed_dict[opt.data_name]
        
    #     opt.result_path = './results/' + opt.data_name + '/' + '_'.join(saving_list) + '.csv'
    #     os.makedirs('./results/' + opt.data_name + '/', exist_ok = True)

    #     space = {
    #             "T": hp.choice('T', [2]),
    #             "alpha": hp.choice("alpha", [0.1, 0.2, 0.3, 0.4, 0.5]),
    #             "dist_frac": hp.choice("dist_frac", [0.005, 0.1, 0.2, 0.5, 1, 2]),
    #             "beta": hp.choice("beta", [0.1, 0.5, 1, 2, 5]),
    #             "gamma": hp.choice("gamma", [5, 10, 15, 20, 25, 30])
    #     }

    #     if opt.method == 'joint' or opt.method == 'ord_joint':
    #         space['lr'] = hp.choice("lr", [0.0001, 0.0005])
        
    #     if opt.method == 'lwf' or opt.method == 'muc_lwf' or opt.method == 'ours_lwf':
    #         space['T'] = hp.choice("T", [0.5, 1, 2, 4])

    #     def f(params):
    #         torch.manual_seed(1)

    #         if opt.method == 'joint' or opt.method == 'ord_joint':
    #             opt.lr = params['lr']

    #         opt.T = params['T']
    #         opt.alpha = params['alpha']
    #         opt.distill_frac = params['dist_frac']
    #         opt.beta = params['beta']
    #         opt.gamma = params['gamma']

    #         acc_mean = 0
    #         def run():
    #             if opt.method == 'lwf':
    #                 acc = baseline_shared_only(opt)
    #             if opt.method == 'specific_only':
    #                 acc = baseline_specific_only(opt)
    #             if opt.method == 'ours_lwf':
    #                 acc = ours(opt)
    #             if opt.method == 'joint':
    #                 acc = baseline_joint(opt)
    #             if opt.method == 'finetune':
    #                 acc = baseline_finetune(opt)
    #             if opt.method == 'ord_joint':
    #                 acc = baseline_ord_joint(opt)
    #             if opt.method == 'acl':
    #                 acc = baseline_acl(opt)           
    #             if opt.method == 'ewc':
    #                 acc = baseline_shared_only_ewc(opt)
    #             if opt.method == 'ours_ewc':
    #                 acc = ours_ewc(opt)
    #             if opt.method == 'muc_ewc':
    #                 acc = muc_ewc(opt)
    #             if opt.method == 'muc_lwf':
    #                 acc = muc(opt)
    #             if opt.method == 'pnn':
    #                 acc = pnn(opt) 
    #             return acc

    #         for _ in range(4):
    #             acc_mean += run()
    #         acc_mean = acc_mean / 4
            
    #         table = PrettyTable(['avg_acc', 'batch_size', 'T', 'dist_frac', 'alpha', 'beta', 'gamma'])
    #         table.add_row(['%.4f' %acc_mean, opt.batchsize, opt.T, opt.distill_frac, opt.alpha, opt.beta, opt.gamma])
    #         with open(opt.result_path, 'a+') as f:
    #             f.write(table.get_string())
    #             f.write('\n\n')
    #         return {'loss': -acc_mean, 'status': STATUS_OK}
            
        
    #     trials = Trials()
    #     best = fmin(f, space, algo=tpe.suggest, max_evals=25, trials=trials)
        
    #     print('best performance:', best) 
    # else:
    if opt.data_name == None:
        data_list = ['bank', 'blast_char', 'income', 'shoppers', 'shrutime']
        for name in data_list:
            torch.manual_seed(1)
            opt.data_name = name
            opt.set_seed = seed_dict[opt.data_name]
            opt.dset_seed = 6 - seed_dict[opt.data_name]

            opt.result_path = './results_avg_loss/' + opt.data_name + '/' + '_'.join(saving_list) + '.csv'
            os.makedirs('./results_avg_loss/' + opt.data_name + '/', exist_ok = True) 
            for i in range(4):
            # opt.set_seed = i + 1
            # opt.dset_seed = 5 - i
                if opt.method == 'lwf':
                    baseline_shared_only(opt)
                if opt.method == 'specific_only':
                    baseline_specific_only(opt)
                if opt.method == 'ours_lwf':
                    ours(opt)
                if opt.method == 'ord_joint':
                    baseline_ord_joint(opt)
                if opt.method == 'joint':
                    baseline_joint(opt)
                if opt.method == 'finetune':
                    baseline_finetune(opt)
                if opt.method == 'acl':
                    baseline_acl(opt)
                if opt.method == 'ewc':
                    baseline_shared_only_ewc(opt)
                if opt.method == 'ours_ewc':
                    ours_ewc(opt)
                if opt.method == 'muc_lwf':
                    muc(opt)
                if opt.method == 'pnn':
                    pnn(opt)
                if opt.method == 'muc_ewc':
                    muc_ewc(opt)

    else:
        opt.set_seed = seed_dict[opt.data_name]
        opt.dset_seed = 6 - seed_dict[opt.data_name]
        opt.result_path = './results/' + opt.data_name + '/' + '_'.join(saving_list) + '.csv'
        os.makedirs('./results/' + opt.data_name + '/', exist_ok = True)

        table = PrettyTable(['shuffle', 'set_seed', 'dset_seed'])
        table.add_row([opt.shuffle, '%d' %opt.set_seed, '%d' %opt.dset_seed])

        with open(opt.result_path, 'a+') as file:
            file.write(table.get_string())
            file.write('\n')
            file.close()

        times = 3
        # if opt.debug:
        #     times = 1
        
        if not opt.hyper_search:
            for i in range(times): 
                if opt.method == 'lwf':
                    baseline_shared_only(opt)
                if opt.method == 'specific_only':
                    baseline_specific_only(opt)
                if opt.method == 'ours_lwf':
                    ours(opt)
                if opt.method == 'ord_joint':
                    baseline_ord_joint(opt)
                if opt.method == 'joint':
                    baseline_joint(opt)
                if opt.method == 'finetune':
                    baseline_finetune(opt)
                if opt.method == 'acl':
                    baseline_acl(opt)
                if opt.method == 'ewc':
                    baseline_shared_only_ewc(opt)
                if opt.method == 'ours_ewc':
                    ours_ewc(opt)
                if opt.method == 'muc_lwf':
                    muc(opt)
                if opt.method == 'pnn':
                    pnn(opt)
                if opt.method == 'muc_ewc':
                    muc_ewc(opt)         
        else:
            acc_list = []
            for i in range(times):
                if opt.method == 'ours_lwf':
                    acc = ours(opt)
                if opt.method == 'ours_ewc':
                    acc = ours_ewc(opt)
                acc_list.append(acc)
            avg_acc = np.mean(acc_list)
            std = np.std(acc_list)

            table = PrettyTable(['avg_acc', 'std', 'T', 'dist_frac', 'alpha', 'beta', 'gamma'])
            table.add_row(['%.4f' %avg_acc, '%.7f' %std, opt.T, opt.distill_frac, opt.alpha, opt.beta, opt.gamma])
            with open(opt.result_path, 'a+') as f:
                f.write(table.get_string())
                f.write('\n\n')
            
