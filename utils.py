import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
import torch.nn as nn
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
from community import community_louvain
import numpy as np
# import scipy.sparse as sp
import igraph as ig
import leidenalg
import sys
import torch as th
from tqdm import tqdm
import random
import dgl
import time
import pandas as pd
import argparse
from torch_geometric.data import Data
# from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.utils import degree,  to_dense_adj, get_self_loop_attr, scatter, to_edge_index, to_torch_coo_tensor, to_networkx\
    , to_scipy_sparse_matrix, to_undirected,add_remaining_self_loops
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
# import metis
import heapq
import csv




def check_nan(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if (value != value).any():  # 检查张量中是否有nan
                print(f"发现nan值在键 {key} 对应的张量中")
                return True
        elif isinstance(value, dict):
            if check_nan(value):  # 递归检查嵌套的字典
                return True
    return False


# get data size
def get_size_mb(obj):  
    size_bytes = sys.getsizeof(obj)  
    size_mb = size_bytes / (1024 * 1024)  # 1MB = 1024KB = 1024*1024B  
    return size_mb

# def str2bool(v):
#   return v.lower() in ['true', 't']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# 根据目录和文件名读取数据
def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

# 根据目录和文件名存储数据
def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True) # dir name
    fpath = os.path.join(base_dir, filename) # dirname + filename 
    torch.save(data, fpath)

# 待改，增加划分的比例控制
def load_dataset(args):
    datapath = args.datapath
    dataname = args.dataset +'/'
    
    if args.dataset=='nba': # feature_normalize
        # edge_df = pd.read_csv('../data/nba/' + 'nba_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt(datapath + dataname + 'nba_relationship.txt').astype('int')
        # node_df = pd.read_csv(os.path.join('../dataset/nba/', 'nba.csv'))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'nba.csv'))
        print('load edge data')
        predict_attr = 'SALARY'
        labels = idx_features_labels[predict_attr].values
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        sens_attr = "country"
        # labels = y
        adj_start = time.time()
        # feature = node_df[node_df.columns[2:]]
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
            
        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        label_number = 200
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index # 不包含label [0,1(大于1的转成1)]以外的值的id

    elif args.dataset=='pokec_z': # sens (gender, age, education, region)  feature_normalize
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region' # gender AGE
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])

        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='pokec_n': # feature_normalize
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_2_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region'
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='credit': # feature_normalize
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'credit.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'credit_edges.txt').astype('int')
        sens_attr = "Age"
        predict_attr="NoDefaultNextMonth"
        print('Loading {} dataset'.format(args.dataset))
        # header = list(idx_features_labels.columns)
        header = list(idx_features_labels.columns)
        header.remove('Single')
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        label_number = 6000
        # label_number = 20000 # for fedgnn full
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='income': # feature_normalize
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'income.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'income_edges.txt').astype('int')
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset'.format(args.dataset))
        header = list(idx_features_labels.columns) #list将括号里的内容变为数组
        header.remove(predict_attr) #header.remove删除括号内的东西
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
            
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='german': # feature_normalize
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'german.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'german_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="Gender"
        predict_attr="GoodCustomer"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        feature = sp.csr_matrix(feature, dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature.todense())
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        label_number = 100
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='bail': # feature_normalize
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'bail.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'bail_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="WHITE"
        predict_attr="RECID"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index

    elif args.dataset=='facebook': # feature_normalize
        # datapath + dataname
        sens_attr="gender"
        # predict_attr="RECID"
        edges_file = open(os.path.join(datapath, dataname, "107.edges"))
        edges = []
        for line in edges_file:
            edges.append([int(one) for one in line.strip("\n").split(" ")])

        feat_file = open(os.path.join(datapath, dataname, "107.feat"))
        feats = []
        for line in feat_file:
            feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name_file = open(os.path.join(datapath, dataname, "107.featnames"))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip("\n").split(" "))
            
        names = {}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]] = name[1]

        feats = np.array(feats)

        # node_mapping = {}
        idx_map = {}
        for j in range(feats.shape[0]): # 
            idx_map[feats[j][0]] = j

        feats = feats[:, 1:]

        sens = feats[:, 264] # gender
        labels = feats[:, 220]
        # column_index = feature.columns.get_loc(sens_attr)
        feats = np.concatenate([feats[:, :264], feats[:, 266:]], -1) # 0-263||266-
        feats = np.concatenate([feats[:, :220], feats[:, 221:]], -1) # 0-263||266- []

        edges = np.array(edges)
        # node_num = feats.shape[0]

        idx = np.arange(feats.shape[0])
        # idx_map = {feats[j][0]: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges.flatten())),dtype=int).reshape(edges.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        print('original edge num',adj.sum())
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            print('add self-loop')
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()

        sens = sens.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feats)
        feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)

        if args.sens_idex:
            print('remove sensitive from node attribute')
            # feature = feature.drop(columns = ["WHITE"])
            # feature = feature.drop(columns = [sens_attr])
        else:
            feature = torch.cat([feature, sens.unsqueeze(-1)], -1)
        column_index = feature.shape[1]-1
        labels = torch.LongTensor(labels) 

        print('dataset:',dataname)
        print('edges:',adj.sum())
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        # return adj, feature, labels, sens, idx_train, idx_val, idx_test
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index


# 待改，增加划分的比例控制
def load_fair_dataset(args):
    datapath = args.datapath
    dataname = args.dataset +'/'
    label_number = args.label_number
    
    print('feat_norm:',args.feat_norm, '\nlabel_number:',label_number,'\nsens_idex:',args.sens_idex,'\nself_loop:',args.self_loop)
    if args.dataset=='nba':
        # edge_df = pd.read_csv('../data/nba/' + 'nba_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt(datapath + dataname + 'nba_relationship.txt').astype('int')
        # node_df = pd.read_csv(os.path.join('../dataset/nba/', 'nba.csv'))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'nba.csv'))
        print('load edge data')
        predict_attr = 'SALARY'
        labels = idx_features_labels[predict_attr].values
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        sens_attr = "country"
        # labels = y
        adj_start = time.time()
        # feature = node_df[node_df.columns[2:]]
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
            
        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        
        labels = torch.LongTensor(labels) 
        labels[labels >1] = 1

        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 200
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index # 不包含label [0,1(大于1的转成1)]以外的值的id

    elif args.dataset=='pokec_z': # sens (gender, age, education, region)
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region' # gender AGE
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])

        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='pokec_n':
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_2_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region'
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='credit':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'credit.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'credit_edges.txt').astype('int')
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        print('Loading {} dataset'.format(args.dataset))
        # header = list(idx_features_labels.columns)
        header = list(idx_features_labels.columns)
        header.remove('Single')
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        # label_number = 6000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 6000')
            label_number = 6000
        # label_number = 20000 # for fedgnn full
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='income':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'income.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'income_edges.txt').astype('int')
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset'.format(args.dataset))
        header = list(idx_features_labels.columns) #list将括号里的内容变为数组
        header.remove(predict_attr) #header.remove删除括号内的东西
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
            
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        # feature = feature_normalize(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='german':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'german.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'german_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="Gender"
        predict_attr="GoodCustomer"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        feature = sp.csr_matrix(feature, dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature.todense())
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 100')
            label_number = 100
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='bail':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'bail.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'bail_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="WHITE" # "MALE"
        if args.sens_attr in ['WHITE','MALE']:
            sens_attr = args.sens_attr
        predict_attr="RECID"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        feature[:,column_index]=sens
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index

    elif args.dataset=='facebook':
        # datapath + dataname
        sens_attr="gender"
        # predict_attr="RECID"
        edges_file = open(os.path.join(datapath, dataname, "107.edges"))
        edges = []
        for line in edges_file:
            edges.append([int(one) for one in line.strip("\n").split(" ")])

        feat_file = open(os.path.join(datapath, dataname, "107.feat"))
        feats = []
        for line in feat_file:
            feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name_file = open(os.path.join(datapath, dataname, "107.featnames"))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip("\n").split(" "))
            
        names = {}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]] = name[1]

        feats = np.array(feats)

        # node_mapping = {}
        idx_map = {}
        for j in range(feats.shape[0]): # 
            idx_map[feats[j][0]] = j

        feats = feats[:, 1:]

        sens = feats[:, 264] # gender
        labels = feats[:, 220]
        # column_index = feature.columns.get_loc(sens_attr)
        feats = np.concatenate([feats[:, :264], feats[:, 266:]], -1) # 0-263||266-
        feats = np.concatenate([feats[:, :220], feats[:, 221:]], -1) # 0-263||266- []

        edges = np.array(edges)
        # node_num = feats.shape[0]

        idx = np.arange(feats.shape[0])
        # idx_map = {feats[j][0]: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges.flatten())),dtype=int).reshape(edges.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        print('original edge num',adj.sum())
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            print('add self-loop')
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()

        sens = sens.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feats)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
            
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)

        if args.sens_idex:
            print('remove sensitive from node attribute')
            # feature = feature.drop(columns = ["WHITE"])
            # feature = feature.drop(columns = [sens_attr])
        else:
            feature = torch.cat([feature, sens.unsqueeze(-1)], -1)
        column_index = feature.shape[1]-1
        labels = torch.LongTensor(labels) 

        print('dataset:',dataname)
        print('edges:',adj.sum())
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        # label_number = 1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        # return adj, feature, labels, sens, idx_train, idx_val, idx_test
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index

    elif args.dataset=='aminer_s': # multi sens and multi label(label have binarilized)
        sens_attr = 'continent'
        name = 'Small'
        edgelist = csv.reader(open(os.path.join(datapath, dataname, "edgelist_{}.txt".format(name))))

        edges = []
        for line in edgelist:
            edge = line[0].split("\t")
            edges.append([int(one) for one in edge])

        edges = np.array(edges)

        labels_file = csv.reader(open(os.path.join(datapath, dataname, "labels_{}.txt".format(name))))
        labels = []
        for line in labels_file:
            labels.append(float(line[0].split("\t")[1]))
        labels = np.array(labels)

        sens_file = csv.reader(open(os.path.join(datapath, dataname, "sens_{}.txt".format(name))))
        sens = []
        for line in sens_file:
            sens.append([float(line[0].split("\t")[1])])
        sens = np.array(sens)
        sens = torch.FloatTensor(sens)
        
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        # adj = adj + sp.eye(adj.shape[0])
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')

        feature = np.load(os.path.join(datapath, dataname, "X_{}.npz".format(name)))
        feature = sp.coo_matrix(
                        (feature["data"], (feature["row"], feature["col"])),
                        shape=(labels.shape[0], np.max(feature["col"]) + 1),
                        dtype=np.float32,
                    ).todense()
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        
        feature = torch.FloatTensor(feature)
        feature = torch.cat([feature, sens], -1)
        sens = sens.squeeze()
        column_index = feature.shape[1]-1
        node_num = labels.shape[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        labels = (labels == most_common_label).astype(int)
        
        labels = torch.LongTensor(labels) 
        # label_number=1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='aminer_l': # multi sens and multi label(label have binarilized)
        sens_attr = 'continent'
        name = 'LCC'
        edgelist = csv.reader(open(os.path.join(datapath, dataname, "edgelist_{}.txt".format(name))))

        edges = []
        for line in edgelist:
            edge = line[0].split("\t")
            edges.append([int(one) for one in edge])

        edges = np.array(edges)

        labels_file = csv.reader(open(os.path.join(datapath, dataname, "labels_{}.txt".format(name))))
        labels = []
        for line in labels_file:
            labels.append(float(line[0].split("\t")[1]))
        labels = np.array(labels)

        sens_file = csv.reader(open(os.path.join(datapath, dataname, "sens_{}.txt".format(name))))
        sens = []
        for line in sens_file:
            sens.append([float(line[0].split("\t")[1])])
        sens = np.array(sens)
        sens = torch.FloatTensor(sens)
        
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # feature = normalize(feature)
        # adj = adj + sp.eye(adj.shape[0])
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')

        feature = np.load(os.path.join(datapath, dataname, "X_{}.npz".format(name)))
        feature = sp.coo_matrix(
                        (feature["data"], (feature["row"], feature["col"])),
                        shape=(labels.shape[0], np.max(feature["col"]) + 1),
                        dtype=np.float32,
                    ).todense()
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        
        feature = torch.FloatTensor(feature)
        feature = torch.cat([feature, sens], -1)
        sens = sens.squeeze()
        column_index = feature.shape[1]-1
        node_num = labels.shape[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        labels = (labels == most_common_label).astype(int)
        
        labels = torch.LongTensor(labels) 
        # label_number=1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
 


def get_general_dataset(path, name, self_loop=True):
    # path = os.path.join(data_path, name)
    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, name) # split
    elif name in ["photo", "computers"]:
        dataset = Amazon(path, name)
    elif name in ["cs", "physics"]:
        dataset = Coauthor(path, name)
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(path, name)
    elif name in ['chameleon']:
        dataset = WikipediaNetwork(path, name)
    elif name in ['squirrel']:
        dataset = WikipediaNetwork(path, name)
    elif name in ['actor']:
        dataset = Actor(path+name)
    elif name in ['wikics']:
        dataset = WikiCS(path+name)
    else:
        raise Exception("Unknown dataset.")
    
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    # dataset.data.edge_index = remove_self_loops(dataset.data.edge_index)
    # dataset.data.edge_index = add_self_loops(dataset.data.edge_index)
    if self_loop:
        dataset.data.edge_index = add_remaining_self_loops(dataset.data.edge_index)
    data = dataset[0]
    
    return data

# 多次划分
# 类采样的划分，其中 train 和 val 都是 per_class，test为剩余数据划分： train val perclass : test rest 
def get_masks_per_class(y, train_per_class=20, val_per_class=30, num_splits=10, seed=20, return_type='mask'):

    if seed is not None:
        torch.manual_seed(seed)
    
    num_classes = int(y.max()) + 1
    num_nodes = y.size(0)
    
    train_mask = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    train_idx_list,val_idx_list,test_idx_list = [],[],[]

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack([torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        val_idx = idx[train_per_class:train_per_class + val_per_class]

        train_mask.scatter_(0, train_idx, True)
        val_mask.scatter_(0, val_idx, True)
        # print(train_idx.shape)
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)

    test_mask = ~(train_mask | val_mask)  # 剩下的作为测试集

    for split in range(num_splits):
        test_idx_list.append((test_mask[:, split].nonzero(as_tuple=False).view(-1)))

    if return_type == 'idx':
        train_idx = torch.concatenate([train_idx_list[i] for i in range(num_classes)], dim=0)
        val_idx = torch.concatenate([val_idx_list[i] for i in range(num_classes)], dim=0)
        test_idx = torch.concatenate([test_idx_list[i] for i in range(num_classes)], dim=0)
        return train_idx, val_idx, test_idx
    # train_mask, val_mask, test_mask = get_masks_per_class(data.y,seed=20, return_type='idx')
    return train_mask, val_mask, test_mask

# 多次划分
# 只有 train 是 perclass, 剩余的数量是固定的，val 500, test 1000 专为 [cora citeseer pubmed]设计 train perclass, val and test is fixed
def get_class_rand_splits(label, label_num_per_class, num_splits=10, seed=20, return_type='mask'):
    num_nodes = label.shape[0]
    train_masks, valid_masks, test_masks = [], [], []
    train_idxs, valid_idxs, test_idxs = [], [], []
    
    for split in range(num_splits):
        if seed is not None:
            torch.manual_seed(seed + split)  # 每次划分使用不同的随机种子
        
        train_idx, non_train_idx = [], []
        idx = torch.arange(num_nodes)
        class_list = label.squeeze().unique()
        valid_num, test_num = 500, 1000
        
        for i in range(class_list.shape[0]):
            c_i = class_list[i]
            idx_i = idx[label.squeeze() == c_i]
            n_i = idx_i.shape[0]
            rand_idx = idx_i[torch.randperm(n_i)]
            train_idx += rand_idx[:label_num_per_class].tolist()
            non_train_idx += rand_idx[label_num_per_class:].tolist()
        
        train_idx = torch.as_tensor(train_idx)
        non_train_idx = torch.as_tensor(non_train_idx)
        non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
        valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]
        
        if return_type == 'mask':
            # 创建与节点数相同大小的布尔张量，初始化为False
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # 将对应的索引位置设为True
            train_mask[train_idx] = True
            valid_mask[valid_idx] = True
            test_mask[test_idx] = True
            
            # 将每次划分的掩码添加到列表中
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
        else:
            # 将索引添加到列表中
            train_idxs.append(train_idx)
            valid_idxs.append(valid_idx)
            test_idxs.append(test_idx)
    if return_type == 'mask':
        # 将列表中的张量拼接成 [num_nodes, num_splits] 的形状
        train_masks = torch.stack(train_masks, dim=1)
        valid_masks = torch.stack(valid_masks, dim=1)
        test_masks = torch.stack(test_masks, dim=1)
        return train_masks, valid_masks, test_masks
    else:
        # 返回索引列表形式
        train_idxs = torch.stack(train_idxs, dim=1)
        valid_idxs = torch.stack(valid_idxs, dim=1)
        test_idxs = torch.stack(test_idxs, dim=1)
        return train_idxs, valid_idxs, test_idxs


# 多次划分
# 按比例随机划分
def get_mask_random_split(num_samples, train_ratio=0.1, test_ratio=0.8, num_splits=10, seed=20, return_type='mask'):
    if seed is not None:
        torch.manual_seed(seed)
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        if return_type == 'mask':
            train_mask = torch.zeros(num_samples, dtype=torch.bool)
            train_mask.fill_(False)
            train_mask[indices[:train_size]] = True

            test_mask = torch.zeros(num_samples, dtype=torch.bool)
            test_mask.fill_(False)
            test_mask[indices[train_size: test_size + train_size]] = True

            val_mask = torch.zeros(num_samples, dtype=torch.bool)
            val_mask.fill_(False)
            val_mask[indices[test_size + train_size:]] = True

            trains.append(train_mask.unsqueeze(1))
            vals.append(val_mask.unsqueeze(1))
            tests.append(test_mask.unsqueeze(1))

        elif return_type == 'idx':
            train_idx = indices[:train_size]
            test_idx = indices[train_size: test_size + train_size]
            val_idx = indices[test_size + train_size:]

            trains.append(train_idx)
            vals.append(val_idx)
            tests.append(test_idx)
    
    if return_type == 'mask':
        train_mask_all = torch.cat(trains, 1)
        val_mask_all = torch.cat(vals, 1)
        test_mask_all = torch.cat(tests, 1)
        return train_mask_all, val_mask_all, test_mask_all
    
    elif return_type == 'idx':
        trains = torch.stack(trains, dim=1)
        vals = torch.stack(vals, dim=1)
        tests = torch.stack(tests, dim=1)
        return trains, vals, tests
    


# 获取原文的划分 'cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin'
def load_fixed_splits(data_dir, dataset, name, protocol='semi'):
    # splits_lst = []
    train_mask = None
    val_mask = None 
    test_mask = None
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi': # 原本的数据划分
        train_mask = torch.as_tensor(dataset.train_mask).unsqueeze(1)
        val_mask = torch.as_tensor(dataset.val_mask).unsqueeze(1)
        test_mask = torch.as_tensor(dataset.test_mask).unsqueeze(1)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']: # geom-gcn的划分（10个）
        train_masks = []
        val_masks = []
        test_masks = []
        for i in range(10):
            splits_file_path = '{}{}/geom-gcn/raw/{}'.format(data_dir, name,name) + '_split_0.6_0.2_' + str(i) + '.npz'
            # splits = {}
            with np.load(splits_file_path) as splits_file:
                train_masks.append(torch.BoolTensor(splits_file['train_mask']).unsqueeze(1))
                val_masks.append(torch.BoolTensor(splits_file['val_mask']).unsqueeze(1))
                test_masks.append(torch.BoolTensor(splits_file['test_mask']).unsqueeze(1))
        
        train_mask = torch.cat(train_masks, dim=1)
        val_mask = torch.cat(val_masks, dim=1)
        test_mask = torch.cat(test_masks, dim=1)
    else:
        raise NotImplementedError
    # train_mask, val_mask, test_mask = load_fixed_splits(data_path,data,'citeseer','none') # none
    return train_mask, val_mask, test_mask



# 只对可用的node id进行选取切分
def train_val_test_split(labels,train_ratio=0.5,val_ratio=0.25,seed=20,label_number=1000):
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]  # 只要label为0和1的
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)
    position1 = train_ratio
    position2 = train_ratio + val_ratio
    idx_train = np.append(label_idx_0[:min(int(position1 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(position1 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(position1 * len(label_idx_0)):int(position2 * len(label_idx_0))], 
                        label_idx_1[int(position1 * len(label_idx_1)):int(position2 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(position2 * len(label_idx_0)):],
                         label_idx_1[int(position2 * len(label_idx_1)):])
    print('train,val,test:',len(idx_train),len(idx_val),len(idx_test))
    return idx_train, idx_val, idx_test



def generate_uniform_idx(idx, missing_rate):

    # 生成与 idx 相同形状的随机数
    node_mask = torch.rand(size=(idx.size(0),))  
    # 根据 missing_rate 划分 drop_idx 和 keep_idx
    drop_mask = node_mask <= missing_rate
    keep_mask = ~drop_mask
    
    # 根据掩码获取对应的索引
    drop_idx = idx[drop_mask]
    keep_idx = idx[keep_mask]
    # drop_idx, keep_idx = generate_uniform_idx(idx, 0.3)
    return drop_idx, keep_idx

# 生成丢弃和保留的idx
def get_masked_idx(node_nums, drop_feat_rate, idx_train, idx_val, idx_test, is_full=True):
    if drop_feat_rate == 0:
        return torch.tensor([]),torch.arange(node_nums)
    
    all_idx = torch.arange(node_nums)
    remaining_idx=None
    remaining_keep_idx=None
    remaining_drop_idx=None
    # 找出不在 idx_train, idx_val, idx_test 中的剩余节点
    used_num = len(idx_train)+len(idx_val)+len(idx_test)

    
    train_drop_idx, train_keep_idx = generate_uniform_idx(idx_train, drop_feat_rate)
    val_drop_idx, val_keep_idx = generate_uniform_idx(idx_val, drop_feat_rate)
    test_drop_idx, test_keep_idx = generate_uniform_idx(idx_test, drop_feat_rate)
    # 将 train, val, test, 和 remaining 结合起来处理
    
    if used_num < node_nums:
        remaining_idx = torch.tensor([i for i in all_idx if i not in torch.cat([idx_train, idx_val, idx_test])])
        remaining_drop_idx, remaining_keep_idx = generate_uniform_idx(remaining_idx, drop_feat_rate)
    else:
        remaining_drop_idx = torch.tensor([],dtype=torch.long)
        remaining_keep_idx = torch.tensor([],dtype=torch.long)
    if is_full:
        all_drop_idx = torch.cat([train_drop_idx, val_drop_idx, test_drop_idx, remaining_drop_idx])
        all_keep_idx = torch.cat([train_keep_idx, val_keep_idx, test_keep_idx, remaining_keep_idx])
        return all_drop_idx, all_keep_idx
    else:
        return [train_drop_idx, val_drop_idx, test_drop_idx, remaining_drop_idx], [train_keep_idx, val_keep_idx, test_keep_idx, remaining_keep_idx]


# 生成掩码后的属性（mean）
def get_masked_feature(feature, edge_index, idx_drop, idx_keep, imput_method='mean',droptype='row',sens_index=None):
    if idx_drop.shape[0] == 0:
        return feature

    processed_feature = feature.clone()
    
    # imput_method 控制填充策略
    if imput_method == 'mean':
        print("mean process")
        # 对 train_val_drop_idx、test_drop_idx 和 remaining_drop_idx 分别进行均值填充
        if droptype=='row':
            processed_feature[idx_drop, :] = feature[idx_keep, :].mean(0)
        else:
            processed_feature[idx_drop, sens_index] = feature[idx_keep, sens_index].mean(0)
    
    elif imput_method == 'zero':
        print("zero process")
        if droptype=='row':
            processed_feature[idx_drop, :] = 0.0
        else:
            processed_feature[idx_drop, sens_index] = 0.0
            
    elif imput_method == 'ones':
        print("ones process")
        if droptype=='row':
            processed_feature[idx_drop, :] = 1.0
        else:
            processed_feature[idx_drop, sens_index] = 1.0
    elif imput_method == 'agg_avg':
        print("aggregate by avg")
        # 创建邻接矩阵
        adj = to_dense_adj(edge_index, max_num_nodes=feature.shape[0]).squeeze(0)
        adj.diagonal(dim1=0, dim2=1)[:] = 0 
        # 对 train_idx 节点进行处理，忽略 test_idx
        adj[idx_drop] = 0  # 删除 test_idx 的影响
        adj[:, idx_drop] = 0  # 确保聚合时不使用 test_idx 节点
        neighbor_sum = torch.matmul(adj, feature)  # 聚合 train_val 的邻域特征
        neighbor_count = adj.sum(dim=1, keepdim=True)
        neighbor_avg = neighbor_sum / neighbor_count
        
        
        # 更新 train_val、test 和 remaining 中丢弃节点的属性
        # processed_feature[idx_drop] = neighbor_avg[idx_drop]
        nan_idx = (neighbor_count==0)[:,0]
        processed_feature[idx_drop] = neighbor_avg[idx_drop]
        processed_feature[nan_idx] = feature[idx_keep].mean(0)
        
    return processed_feature




# sp mx 2 edge_index
def sparse_2_edge_index(adj):   
    edge_index_origin = adj.nonzero()
    edge_index = torch.stack([torch.from_numpy(edge_index_origin[0]).long(), torch.from_numpy(edge_index_origin[1]).long()])
    return edge_index 

# dense mx 2 edge_index
def dense_2_edge_index(adj):
    edge_index = adj.nonzero().t().contiguous()
    return edge_index

# 防止数值过小
def non_small_zero(number):
    number = number+1e-6 if number==0 else number
    return number

# mask [True, Fasle, False]
def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx]=True
    return mask

# idx [0]
def mask2idx(mask):
    return torch.nonzero(mask, as_tuple=False).squeeze()
def normalize_features_full(features, norm_type, norm_method):
    if norm_type == 'none':
        return features  # 无归一化

    if norm_method == 'sum':
        # 基于和的归一化
        if norm_type == 'column':
            col_sum = features.sum(dim=0, keepdim=True)
            col_sum[col_sum == 0] = 1  # 避免除以 0
            return features / col_sum
        elif norm_type == 'row':
            row_sum = features.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1
            return features / row_sum

    elif norm_method == 'minmax':
        # 基于最小值和最大值的归一化
        if norm_type == 'column':
            col_min = features.min(dim=0, keepdim=True).values
            col_max = features.max(dim=0, keepdim=True).values
            scale = col_max - col_min
            scale[scale == 0] = 1  # 避免除以 0
            return (features - col_min) / scale
        elif norm_type == 'row':
            row_min = features.min(dim=1, keepdim=True).values
            row_max = features.max(dim=1, keepdim=True).values
            scale = row_max - row_min
            scale[scale == 0] = 1
            return (features - row_min) / scale

    else:
        raise ValueError(f"Unsupported feat_norm_method: {norm_method}")


def feature_normalize(feature):  # 行归一化
    '''sum_norm'''
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True) # [num_node]
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum


def feature_normalize_column(feature, feature_range=(0, 1)):  # 列归一化(求和)
    '''sum_norm'''
    feature = np.array(feature)
    data_min = feature.min(axis=0,keepdims=True)  
    data_max = feature.max(axis=0,keepdims=True)  
    rowsum = data_max - data_min  
    # rowsum = feature.sum(axis=1, keepdims=True) # [num_node]
    rowsum = np.clip(rowsum, 1, 1e10)
    scale = (feature_range[1] - feature_range[0]) / rowsum 
    min_max_data = data_min + scale * (feature - data_min) 
    min_max_data = min_max_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return min_max_data  


def feature_transform(feature, str='row'):
    if str == 'row':
        return feature_normalize_torch(feature)
    elif str == 'col':
        return feature_normalize_column_torch(feature)
    else:
        return feature


def feature_normalize_torch(feature):
    
    rowsum = feature.sum(dim=1, keepdim=True)  # [num_nodes, 1]
    rowsum = torch.clamp(rowsum, min=1.0)  # 防止出现为 0 的情况
    normalized_feature = feature / rowsum
    
    return normalized_feature


def feature_normalize_column_torch(feature, feature_range=(0, 1)):
    
    data_min = feature.min(dim=0, keepdim=True)[0]  # [1, num_features]
    data_max = feature.max(dim=0, keepdim=True)[0]  # [1, num_features]
    rowsum = data_max - data_min
    rowsum = torch.clamp(rowsum, min=1.0)  # 防止出现为 0 的情况
    
    scale = (feature_range[1] - feature_range[0]) / rowsum
    min_max_data = data_min + scale * (feature - data_min)
    min_max_data = min_max_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return min_max_data


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj_row(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def normalize_adj_sym(adj): # for gcn
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def labels_to_one_hot(labels, nclass):
    
    # 初始化一个全零张量
    # one_hot = torch.zeros(labels.size(0), nclass, device=labels.device)
    one_hot = torch.zeros(len(labels), nclass, device=labels.device)
    
    # 将对应类别的位置置为 1
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    
    return one_hot

# one-hot output , num label
def accuracy(output, labels): # logits,label()
    preds=None
    if len(output.shape)>1:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct.item() / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

# edge_index 2 sp matrix
def edge_index_2_sparse_mx(edge_index):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    num_nodes = max(np.max(row), np.max(col)) + 1
    # 使用 scipy.sparse.coo_matrix 构造函数创建 COO 稀疏矩阵
    sparse_adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return  sparse_adj

def edge_index_2_sparse_tensor(edge_index, num_nodes):
    # 构建稀疏矩阵
    values = torch.ones(edge_index.size(1))  # 假设所有边的权重都是1
    sparse_adj = torch.sparse.FloatTensor(edge_index, values, torch.Size([num_nodes, num_nodes]))
    return sparse_adj

def edge_index_2_tensor(edge_index, num_nodes):
    # 构建稀疏矩阵
    values = torch.ones(edge_index.size(1))  # 假设所有边的权重都是1
    sparse_adj = torch.sparse.FloatTensor(edge_index, values, torch.Size([num_nodes, num_nodes]))
    adj = to_dense_adj(edge_index,max_num_nodes=num_nodes)
    return adj

# sparse normalize D^{-0.5}(A+I)D^{-0.5}
def sys_normalized_adjacency(adj): # add self_loop
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


def laplacian_positional_encoding(g, pos_enc_dim):
    # 将输入的邻接矩阵转换为拉普拉斯矩阵
    laplacian = sp.csgraph.laplacian(g, normed=False)
    # laplacian = sp.csgraph.laplacian(g, normed=True)
    eignvalue, eignvector = sp.linalg.eigsh(laplacian, which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def adjacency_positional_encoding_full(g): # coo

    # eignvalue, eignvector = sp.linalg.eigsh(g, which='LM', k=pos_enc_dim)
    eignvalue, eignvector = np.linalg.eigh(g.todense()) # 前m小
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def adjacency_positional_encoding(g, pos_enc_dim): # coo
    # adj = g.adjacency_matrix_scipy(return_edge_ids=False)
    # adj = g.adj_sparse('coo',return_edge_ids=False)
    # adj = g.adjacency_matrix(scipy_fmt="coo")
    eignvalue, eignvector = sp.linalg.eigsh(g, which='LM', k=pos_enc_dim)
    # eignvalue, eignvector = sp.linalg.eigsh(g.adjacency_matrix_scipy(return_edge_ids=False).astype(float), which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix

# 保留多少比例的边
def set_seed(seed = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# 根据sp adj和节点id列表，返回子图的adj
def get_subgraph_adj_by_nodes(adj, nodes): 
    graph = dgl.from_scipy(adj) 
    subgraph = graph.subgraph(nodes) 
    return subgraph.adjacency_matrix_scipy(return_edge_ids=False).astype(float)



