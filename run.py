import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-method', default=None, type=str)
parser.add_argument('-gpu', default=1, type=int)
parser.add_argument('-ext_type', default='transformer', type=str)

opt = parser.parse_args()

gpu = str(opt.gpu)

if opt.ext_type == 'transformer':
    if opt.method == 'lwf':
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=1 -T=4 -data_name=bank")
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=0.2 -T=2 -data_name=blast_char")
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=1 -T=2 -data_name=income")
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=1 -T=4 -data_name=shoppers")
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=0.2 -T=2 -data_name=shrutime")
        os.system("python main.py -method=lwf -gpu="+ gpu + " -hyper_search -distill_frac=1 -T=2 -data_name=volkert -class_inc")

    if opt.method == 'ewc':
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -data_name=bank")
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=2 -data_name=blast_char")
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -data_name=income")
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -data_name=shoppers")
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -data_name=shrutime")
        os.system("python main.py -method=ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -data_name=volkert -class_inc")

    if opt.method == 'muc_lwf':
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=2 -distill_frac=1 -data_name=bank")
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=4 -distill_frac=1 -data_name=blast_char")
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=4 -distill_frac=0.005 -data_name=income")
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=4 -distill_frac=0.1 -data_name=shoppers")
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=2 -distill_frac=0.5 -data_name=shrutime")
        os.system("python main.py -method=muc_lwf -gpu="+ gpu + " -hyper_search -T=2 -distill_frac=1 -data_name=volkert -class_inc")

    if opt.method == 'muc_ewc':
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=bank")
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=blast_char")
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=shrutime")        
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=income")
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=shoppers")
        os.system("python main.py -method=muc_ewc -gpu="+ gpu + "  -hyper_search  -data_name=volkert -class_inc")

    if opt.method == 'ours_lwf':
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=30 -data_name=bank")
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=1 -alpha=0.4 -beta=2 -gamma=5 -data_name=blast_char")
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=15 -data_name=income")
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=1 -alpha=0.2 -beta=0.5 -gamma=5 -data_name=shoppers")
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=0.2 -alpha=0.2 -beta=2 -gamma=30 -data_name=shrutime")
        os.system("python main.py -method=ours_lwf -gpu="+ gpu + "  -hyper_search -distill_frac=0.1 -alpha=0.2 -beta=0.1 -gamma=5 -data_name=volkert -class_inc")

    if opt.method == 'ours_ewc':
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=0.5 -alpha=0.2 -beta=0.1 -gamma=15 -data_name=bank")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=0.5 -alpha=0.3 -beta=0.5 -gamma=10 -data_name=blast_char")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=0.1 -beta=0.5 -gamma=25 -data_name=income")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=0.5 -alpha=0.4 -beta=2 -gamma=25 -data_name=shoppers")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=15 -data_name=shrutime")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=10 -data_name=volkert -class_inc")

    if opt.method == 'pnn':
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=bank")
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=blast_char")
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=income")
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=shoppers")
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=shrutime")
        os.system("python main.py -method=pnn -gpu="+ gpu + " -hyper_search -data_name=volkert -class_inc")

    if opt.method == 'acl':
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search  -data_name=bank")
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search -data_name=blast_char")
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search -data_name=income")
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search -data_name=shoppers")
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search -data_name=shrutime")
        os.system("python main.py -method=acl -gpu="+ gpu + " -order=3 -hyper_search -data_name=volkert -class_inc")

    if opt.method == 'joint':
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=bank")
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=blast_char")
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=income")
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=shoppers")
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=shrutime")
        os.system("python main.py -method=joint -gpu="+ gpu + " -hyper_search -comment=order2 -data_name=volkert -class_inc -lr=0.0005")

    if opt.method == 'ord_joint':
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=bank")
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=blast_char")
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=income")
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=shoppers")
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=shrutime")
        os.system("python main.py -method=ord_joint -gpu="+ gpu + " -hyper_search -data_name=volkert -class_inc")
    
    if opt.method == 'afec':
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=bank")
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=blast_char")
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=income")
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=shoppers")
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=shrutime")
        os.system("python main.py -method=afec -gpu="+ gpu + " -hyper_search -distill_frac=1 -alpha=1 -data_name=volkert -class_inc")

    if opt.method == 'pcl':
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=bank")
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=blast_char")
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=income")
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=shoppers")
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=shrutime")
        os.system("python main.py -method=pcl -gpu="+ gpu + " -hyper_search -beta=1 -alpha=1 -data_name=volkert -class_inc")

    if opt.method == 'dmc':
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=dmc -gpu="+ gpu + " -hyper_search -data_name=volkert -class_inc")

elif opt.ext_type == 'rnn' :
# else :
    if opt.method == 'muc_ewc':
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=bank")
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=blast_char")
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=income")
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shoppers")
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shrutime")
        os.system("python main.py -num_workers=0 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=volkert -class_inc")

    if opt.method == 'ours_lwf':
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=30 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.4 -beta=2 -gamma=5 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=15 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.2 -beta=0.5 -gamma=5 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.2 -beta=2 -gamma=30 -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.1 -alpha=0.2 -beta=0.1 -gamma=5 -data_name=volkert -class_inc")

    if opt.method == 'ours_ewc':
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.2 -beta=0.1 -gamma=15 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.3 -beta=0.5 -gamma=10 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.1 -beta=0.5 -gamma=25 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.4 -beta=2 -gamma=25 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=15 -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=10 -data_name=volkert -class_inc")

    if opt.method == 'pnn':
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc")

    if opt.method == 'joint':
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc -lr=0.0005")

elif opt.ext_type == 'gru' :

    if opt.method == 'muc_ewc':
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=bank")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=income")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=volkert -class_inc")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
    
    if opt.method == 'ours_lwf':
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=30 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.4 -beta=2 -gamma=5 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=15 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.2 -beta=0.5 -gamma=5 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.2 -beta=2 -gamma=30 -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.1 -alpha=0.9 -beta=0.1 -gamma=5 -data_name=volkert -class_inc")

    if opt.method == 'ours_ewc':
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.2 -beta=0.1 -gamma=15 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.3 -beta=0.5 -gamma=10 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.1 -beta=0.5 -gamma=25 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.4 -beta=2 -gamma=25 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=15 -data_name=shrutime")
        os.system("python main.py -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.9 -beta=1 -gamma=10 -data_name=volkert -class_inc")

    if opt.method == 'pnn':
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc")

    if opt.method == 'joint':
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc -lr=0.0005")

if opt.ext_type == 'mlp':
    if opt.method == 'joint':
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=joint -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc -lr=0.0005")
    
    if opt.method == 'ours_lwf':
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=30 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.4 -beta=2 -gamma=5 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.1 -beta=0.1 -gamma=15 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=1 -alpha=0.2 -beta=0.5 -gamma=5 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.2 -alpha=0.2 -beta=2 -gamma=30 -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=ours_lwf -gpu="+ gpu + f" -extractor_type={opt.ext_type}  -hyper_search -distill_frac=0.1 -alpha=0.9 -beta=0.1 -gamma=5 -data_name=volkert -class_inc")

    if opt.method == 'ours_ewc':
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.2 -beta=0.1 -gamma=15 -data_name=bank")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.3 -beta=0.5 -gamma=10 -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.1 -beta=0.5 -gamma=25 -data_name=income")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=0.5 -alpha=0.4 -beta=2 -gamma=25 -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.2 -beta=1 -gamma=15 -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=ours_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -distill_frac=1 -alpha=0.9 -beta=1 -gamma=10 -data_name=volkert -class_inc")
    
    if opt.method == 'muc_ewc':
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=bank")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=income")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=muc_ewc -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search  -data_name=volkert -class_inc")

    if opt.method == 'pnn':
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=bank")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=blast_char")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=income")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shoppers")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=shrutime")
        os.system("python main.py -num_workers=1 -method=pnn -gpu="+ gpu + f" -extractor_type={opt.ext_type} -hyper_search -data_name=volkert -class_inc")