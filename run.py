import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-method', default=None, type=str)
parser.add_argument('-gpu', default=1, type=int)
# parser.add_argument('-method', default=None, type=str)

opt = parser.parse_args()

if opt.method == 'lwf':
    # os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=1 -T=4 -data_name=bank")
    os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=0.2 -T=2 -data_name=blast_char")
    # os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=1 -T=2 -data_name=income")
    # os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=1 -T=4 -data_name=shoppers")
    # os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=0.2 -T=2 -data_name=shrutime")
    # os.system("python main.py -method=shared_only -gpu=2 -comment=fg -distill_frac=1 -T=2 -data_name=volkert -class_inc")

if opt.method == 'ewc':
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=1 -data_name=bank")
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=2 -data_name=blast_char")
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=1 -data_name=income")
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=1 -data_name=shoppers")
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=1 -data_name=shrutime")
    os.system("python main.py -method=shared_only_ewc -gpu=3 -comment=fg -distill_frac=1 -data_name=volkert -class_inc")

if opt.method == 'muc_lwf':
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=2 -distill_frac=1 -data_name=bank")
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=4 -distill_frac=1 -data_name=blast_char")
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=4 -distill_frac=0.005 -data_name=income")
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=4 -distill_frac=0.1 -data_name=shoppers")
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=2 -distill_frac=0.5 -data_name=shrutime")
    os.system("python main.py -method=muc -gpu=4 -comment=fg -T=2 -distill_frac=1 -data_name=volkert -class_inc")

if opt.method == 'muc_ewc':
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=bank")
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=blast_char")
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=income")
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=shoppers")
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=shrutime")
    os.system("python main.py -method=muc_ewc -gpu=5 -comment=fg -data_name=volkert -class_inc")
    
if opt.method == 'ours_lwf':
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=30 -data_name=bank")
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=1 -alpha=0.2 -beta=0.1 -gamma=5 -data_name=blast_char")
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=15 -data_name=income")
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=1 -alpha=0.2 -beta=0.1 -gamma=25 -data_name=shoppers")
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=0.2 -alpha=0.2 -beta=2 -gamma=30 -data_name=shrutime")
    os.system("python main.py -method=ours -gpu=1 -comment=fg -distill_frac=0.1 -alpha=0.2 -beta=0.1 -gamma=5 -data_name=volkert -class_inc")

if opt.method == 'ours_ewc':
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=0.5 -alpha=0.2 -beta=0.1 -gamma=15 -data_name=bank")
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=0.4 -alpha=0.3 -beta=0.5 -gamma=10 -data_name=blast_char")
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=1 -alpha=0.1 -beta=0.5 -gamma=25 -data_name=income")
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=0.6 -alpha=0.4 -beta=2 -gamma=25 -data_name=shoppers")
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=1 -alpha=0.2 -beta=1 -gamma=15 -data_name=shrutime")
    os.system("python main.py -method=ours_ewc -gpu=6 -comment=fg -distill_frac=1 -alpha=0.2 -beta=1 -gamma=10 -data_name=volkert -class_inc")

if opt.method == 'pnn':
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=bank")
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=blast_char")
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=income")
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=shoppers")
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=shrutime")
    os.system("python main.py -method=pnn -gpu=1 -comment=fg -data_name=volkert -class_inc")

if opt.method == 'acl':
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=bank")
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=blast_char")
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=income")
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=shoppers")
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=shrutime")
    os.system("python main.py -method=acl -gpu=1 -comment=fg -data_name=volkert -class_inc")

if opt.method == 'joint':
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=bank")
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=blast_char")
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=income")
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=shoppers")
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=shrutime")
    os.system("python main.py -method=joint -gpu=1 -comment=fg -data_name=volkert -class_inc -lr=0.0005")

if opt.method == 'ord_joint':
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=bank")
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=blast_char")
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=income")
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=shoppers")
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=shrutime")
    os.system("python main.py -method=ord_joint -gpu=1 -comment=fg -data_name=volkert -class_inc")