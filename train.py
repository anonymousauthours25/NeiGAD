# %% [markdown]
# ### 主赛道，谱分析对异常检测的作用
# #### 谱截断，图切分，采样
# #### 1) 最新的baseline加进来对比
# #### 2) 将现有baseline调到最佳
# #### GAAN实现 batch_size 
# #### CONAD OK

# %%
# 初始化参数
import os
import argparse
import tqdm
import time
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.seed import seed_everything
from torch_geometric.utils import  add_self_loops, remove_self_loops
from torch_geometric.data import Data

from utils import str2bool, dense_2_edge_index, normalize_adj_sym,  torch_load, torch_save, \
    edge_index_2_sparse_mx, adjacency_positional_encoding, laplacian_positional_encoding, normalize_features_full
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import gc


from pygod.detector import CONAD, CoLA, ANOMALOUS, DOMINANT, AnomalyDAE, GADNR, GAE, GAAN

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['TORCH_USE_CUDA_DSA']='1'

parser = argparse.ArgumentParser()
# general data ['cora', 'citeseer', 'pubmed', 'acm', 'blogcatalog', 'books', 'enron', 'dblp']
parser.add_argument('--dataset', default='cora', help='dataset name: Flickr/ACM/BlogCatalog')
parser.add_argument('--datapath', type=str, default='./data', help='Random seed.') # pokec_z
parser.add_argument('--feat_norm', type=str, default='none',choices=['none','row','column'], help="view of norm") # sgd adam adamw adadelta adagrad
parser.add_argument('--feat_norm_type', type=str, default='minmax',choices=['sum','minmax'], help="type of norm") # sgd adam adamw adadelta adagrad


# {MLPAE GCNAE DOMINANT GAAN AnomalyDAE COLA CONAD GADNR}
parser.add_argument('--model',default='GCNAE', help='GOD model from {MLPAE GCNAE DOMINANT GAAN AnomalyDAE COLA CONAD GADNR}')

parser.add_argument('--nhidden', type=int, default=16, help='dimension of hidden embedding (default: 64)')
parser.add_argument('--nlayer', type=int, default=2, help='dimension of hidden embedding (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='Training epoch') # 100
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=0, help='Random seed.') # 0 for all 
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay.') # 1e-3

parser.add_argument('--verbose', default=3, type=int, help='Verbosity') #[0,1,2,3]
parser.add_argument('--eval_mode', type=str2bool, default=True)

parser.add_argument('--gpu', type=int, default=1, help='dimension of hidden embedding (default: 64)')
parser.add_argument('--seed', type=int, default=20, help='Random seed.') # 20 22 23 25
parser.add_argument('--self_loop', type=str2bool, default=True)

parser.add_argument('--pe_method', type=str, default='adj', choices=['adj','lap','none'], help='spectral num') # 2
parser.add_argument('--pe_dim', type=int, default=2, help='spectral num') # 2
parser.add_argument('--only_k', type=str2bool, default=False)

args = parser.parse_args()
# args = parser.parse_args([])


seed_everything(args.seed)

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    
def load_anomaly_detection_dataset(dataset, datadir='data'):
    
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    # adj = data_mat['Network'] # A
    # feat = data_mat['Attributes'] # X
    try:
        adj = data_mat['Network']
    except KeyError:
        adj = data_mat['A']
    print('original edge num:',adj.sum())
    # try load feat, else load 'X'
    try:
        feat = data_mat['Attributes']
    except KeyError:
        feat = data_mat['X']
        
    truth = data_mat['Label'] 
    truth = truth.flatten()

    # adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = normalize_adj_sym(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    # adj = adj + sp.eye(adj.shape[0])
    try:
        adj = adj.toarray()
        feat = feat.toarray()
    except:
        pass
    
    return adj_norm, feat, truth, adj



data = None

if args.dataset in ['flickr', 'acm', 'blogcatalog', 'cora', 'citeseer', 
                    'pubmed', 'dblp']:
    # data = torch.load(f"./data/reddit.pt")
    adj_norm, feat, truth, adj = load_anomaly_detection_dataset(args.dataset, args.datapath)
    adj_norm = torch.FloatTensor(adj_norm)
    adj = torch.FloatTensor(adj)
    print('adj.max():',adj.max())
    print('adj.min():',adj.min())
    edge_index = dense_2_edge_index(adj)
    feature = torch.FloatTensor(feat)
    y = torch.LongTensor(truth)
    contamination = y.sum().item()/len(y)
    sen = None
    
    # 根据 args.self_loop 处理 edge_index
    edge_index, _ = remove_self_loops(edge_index)
    if args.self_loop:
        edge_index, _ = add_self_loops(edge_index)
    # else:
    #     edge_index, _ = remove_self_loops(edge_index)
        
    feature = normalize_features_full(feature, args.feat_norm, args.feat_norm_type)
    
    data = Data(
        x=feature,         # node feat
        edge_index=edge_index,  # edge list
        y=y,               # label
        sensitive=sen,
        contamination=contamination
    )

elif args.dataset in ['books', 'disney', 'enron', 'reddit', 'weibo']:
    data=torch_load(args.datapath,args.dataset+'.pt')
    data.x = data.x.float()
    data.y = data.y.long()

    data.contamination = data.y.sum().item()/len(data.y)
    print('original edge num:',data.edge_index.shape[1])
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if args.self_loop:
        data.edge_index, _ = add_self_loops(data.edge_index)

    data.x = normalize_features_full(data.x, args.feat_norm, args.feat_norm_type)
    

print('dataset:',args.dataset)
print('gpu:',args.gpu)
print('seed:',args.seed)

print('feature.shape:',data.x.shape)
print('y.shape:',data.y.shape)
print('y.unique:',data.y.unique(return_counts=True))
print('edge_index.shape:',data.edge_index.shape)
print('data.contamination:',data.contamination)

nfeat = data.x.shape[1]
nnode = data.x.shape[0]
nhidden = args.nhidden 
num_layers = args.nlayer
dropout = args.dropout
lr = args.lr
sigmoid_s = False
epochs = args.epochs


batch_size = args.batch_size
gpu = args.gpu
weight_decay = args.weight_decay
contamination = data.contamination
verbose = args.verbose
eignvalue, eignvector = None, None
model = None

'''
position encoding for each method
'''
file_path = './pe_files/'+args.dataset+'/'
file_name = args.dataset+'_'+args.pe_method+'_'+str(args.pe_dim)+'_'+str(args.seed)+'_'+str(args.self_loop)+'_pe.pt'

if args.pe_method == 'none':
    pass
elif args.pe_method == "adj":
    print('adjancency position encoding!')      
    # eignvalue, eignvector = laplacian_positional_encoding_spec(g, lm=args.e_dim)
    if os.path.exists(file_path+file_name):
        print('file exist:',file_name, 'load data')
        load_data = torch_load(file_path,file_name)
        eignvalue, eignvector=load_data
    else:
        sp_adj = edge_index_2_sparse_mx(data.edge_index)
        eignvalue, eignvector = adjacency_positional_encoding(sp_adj, args.pe_dim)
        torch_save(file_path,file_name,[eignvalue, eignvector])
        
    if args.only_k:
        feature = torch.cat((data.x, eignvector[:,1:2]), dim=1)
        nfeat = nfeat+1
    else:
        feature = torch.cat((data.x, eignvector), dim=1)
        nfeat = nfeat+args.pe_dim
    data.x = feature

elif args.pe_method=="lap":
    print('laplacian position encoding!')      
    # eignvalue, eignvector = laplacian_positional_encoding_spec(g, lm=args.e_dim)
    if os.path.exists(file_path+file_name):
        print('file exist:',file_name, 'load data')
        load_data = torch_load(file_path,file_name)
        eignvalue, eignvector=load_data
    else:
        sp_adj = edge_index_2_sparse_mx(data.edge_index)
        eignvalue, eignvector = laplacian_positional_encoding(sp_adj, args.pe_dim)
        torch_save(file_path,file_name,[eignvalue, eignvector])       
        
    if args.only_k:
        feature = torch.cat((data.x, eignvector[:,1:2]), dim=1)
        nfeat = nfeat+1
    else:
        feature = torch.cat((data.x, eignvector), dim=1)
        nfeat = nfeat+args.pe_dim
    data.x = feature



all_idx = torch.arange(nnode)

if args.model == 'GCNAE': # 2016
    model = GAE(nhidden,num_layers, dropout,
                weight_decay=weight_decay, 
                gpu=gpu, 
                contamination=contamination,
                lr=lr, epoch=epochs,
                verbose=verbose,
                batch_size=batch_size,
                eval_mode=args.eval_mode)
if args.model == 'MLPAE': # 2016
    model = GAE(nhidden,num_layers, dropout,
                weight_decay=weight_decay, 
                gpu=gpu, 
                contamination=contamination,
                lr=lr, epoch=epochs,
                verbose=verbose,
                backbone='MLP',
                batch_size=batch_size,
                eval_mode=args.eval_mode)

elif args.model == 'DOMINANT': # 2019
    model = DOMINANT(nhidden, num_layers, dropout, 
                     weight_decay=weight_decay, 
                     gpu=gpu, 
                     contamination=contamination, 
                     lr=lr, epoch=epochs, 
                     verbose=verbose,
                     batch_size=batch_size,
                     eval_mode=args.eval_mode)
elif args.model == 'GAAN': # 2020
    if args.dataset in ['pubmed','enron','acm']:
        # batch_size=1024 # pubmed enron acm reddit
        args.batch_size=1024 # pubmed enron acm reddit
        batch_size=args.batch_size
        
    elif args.dataset in ['reddit','weibo','blogcatalog']:
        args.batch_size=128 # pubmed enron acm reddit
        batch_size=args.batch_size
        
    model = GAAN(nhidden,nhidden,num_layers,dropout,
                weight_decay=weight_decay, 
                gpu=gpu, 
                contamination=contamination,
                lr=lr, epoch=epochs,
                verbose=verbose,
                batch_size=batch_size,
                eval_mode=args.eval_mode)
elif args.model == 'AnomalyDAE': # 2020
    model = AnomalyDAE(nhidden,nhidden,num_layers,dropout,
                       weight_decay=weight_decay, 
                       gpu=gpu, 
                       contamination=contamination,
                       lr=lr, epoch=epochs,
                       verbose=verbose,
                       batch_size=batch_size,
                       eval_mode=args.eval_mode)
elif args.model == 'COLA': # 2021 
    model = CoLA(nhidden,num_layers,dropout, 
                 weight_decay=weight_decay, 
                 gpu=gpu, 
                 contamination=contamination, 
                 lr=lr, epoch=epochs, 
                 verbose=verbose,
                 batch_size=batch_size,
                 eval_mode=args.eval_mode) 
elif args.model == 'CONAD': # 2022
    model = CONAD(nhidden,num_layers,dropout,
                  weight_decay=weight_decay, 
                  gpu=gpu, 
                  contamination=contamination,
                  lr=lr, epoch=epochs,
                  verbose=verbose,
                  batch_size=batch_size,
                  eval_mode=args.eval_mode)
elif args.model =='GADNR': # 2024
    model = GADNR(nhidden,num_layers,dropout=dropout,
                  weight_decay=weight_decay, 
                  gpu=gpu, 
                  contamination=contamination,
                  lr=lr, epoch=epochs,
                  verbose=verbose,
                  batch_size=batch_size,
                  eval_mode=args.eval_mode)

print(model)

# ANOMALOUS DOMINANT ONE ADONE DONE AnomalyDAE COLA CONAD GADNR CARD

# %%
data = data.to(device)
# model = model.to(device)

train_start = time.time()
print('='*10,"Start Training: ",args.dataset,'='*10)
model.fit(data)
train_end = time.time()
train_time = (train_end-train_start)
print('success train data, time is:{:.3f}'.format(train_time))

gc.collect()
torch.cuda.empty_cache()

# original score
score = model.decision_score_ # most of them are loss function .detach()
# project to probability
# outlier_prob = model.predict(data, return_pred=False, return_prob=True) # from score_
# decision by threshold
prediction = model.predict(data, return_pred=True) # based on the threshold/contamination

# print('prediction.unique():', prediction.unique(return_counts=True))
max_memory_cached = torch.cuda.max_memory_cached(device=device) / 1024 ** 2 
max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
print("Max memory cached:", max_memory_cached, "MB")
print("Max memory allocated:", max_memory_allocated, "MB")

# %%
y_cpu = data.y.cpu().numpy()

roc_auc = roc_auc_score(y_cpu, score.numpy())
# precision, recall, thresholds = precision_recall_curve(data.y.cpu().numpy(), score.numpy())
# pr_auc = auc(recall, precision)
pr_auc = average_precision_score(y_cpu, score) # PR 曲线下的面积

f1 = f1_score(y_cpu, prediction) # F1 score

# roc_auc2 = roc_auc_score(data.y.cpu().numpy(), outlier_prob)
# print('pr_auc:', pr_auc)
print('dataset:',args.dataset)
print('method :',args.model)
print('roc_auc: {:.2f}'.format(roc_auc*100))
print('pr_auc : {:.2f}'.format(pr_auc*100))
print('f1score: {:.2f}'.format(f1*100))

best_epoch = model.best_epoch
best_decision_score_ = model.best_decision_score_
best_roc_auc = model.roc_auc
best_pr_auc = average_precision_score(data.y.cpu().numpy(), best_decision_score_)
print('best_epoch:',best_epoch)

print('best_roc_auc : {:.2f}'.format(best_roc_auc*100))
print('best_pr_auc  : {:.2f}'.format(best_pr_auc*100))

# %%

# %%


log_data = {
    'method': [args.model],
    'dataset': [args.dataset],
    'anomaly_ratio(%)':[data.contamination*100],
    'batch_size':[args.batch_size],
    'feat_norm':[args.feat_norm],
    'feat_norm_type':[args.feat_norm_type],
    'pe_method': [args.pe_method],
    'pe_dim': [args.pe_dim],
    'nlayer': [args.nlayer],
    'epochs': [args.epochs],
    'num_hidden': [args.nhidden],
    'dropout': [args.dropout],
    'lr': [args.lr],
    'seed': [args.seed],
    'max_memory_cached': [max_memory_cached],
    'max_memory_allocated': [max_memory_allocated],
    'weight_decay': [args.weight_decay],
    'gpu': [args.gpu],
    'train_time(s)': [train_time],
    
    'final_roc_auc': [roc_auc*100],
    'final_pr_auc': [pr_auc*100],
    'final_F1':[f1*100],
    
    'best_epoch':[best_epoch],
    'best_roc_auc':[best_roc_auc*100],
    'best_pr_auc':[best_pr_auc*100],
    
    'args':[args]
}

train_logs = pd.DataFrame(log_data)

import os
logs_path = './logs/'
train_log_save_file=logs_path+'pygod'+'_gpu_'+str(args.gpu)+'_train_log.csv'

if os.path.exists(train_log_save_file): # add
    train_logs.to_csv(train_log_save_file, mode='a', index=False, header=0)
else: # create
    train_logs.to_csv(train_log_save_file, index=False)

print('='*10,"End Log",'='*10)

# %%



