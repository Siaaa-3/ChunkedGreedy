#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import *
import FS_Block_gpu

graph = np.load('/home/shijinliang/module/mnt/libra_suite/sp_matrix/vsp_c-30_data_data.npz')

num_nodes_ori =  graph['num_nodes_src']-0
num_nodes_dst =  graph['num_nodes_dst']-0
num_nodes = num_nodes_ori
if num_nodes_ori%16 !=0 :
    num_nodes = num_nodes_ori + 16 - num_nodes_ori%16
num_edges = graph['num_edges']-0
src_li = graph['src_li']
dst_li = graph['dst_li']
# edge_index = graph['edge_index_new']
edge_index = np.stack([src_li, dst_li])
avg_degree = num_edges / num_nodes
# print("Num_nodes, Num_edges: " + str(num_nodes) + ' , ' + str(num_edges))
val = [1] * num_edges
scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes_dst))
adj = scipy_coo.tocsr()
window = 8
wide = 8

column_index = torch.IntTensor(adj.indices)
row_pointers = torch.IntTensor(adj.indptr)

window_num = int(num_nodes/window)
blockPartition = torch.zeros(window_num, dtype=torch.int)
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
# column_index_ori  = column_index.cuda()
# row_pointers_ori = row_pointers.cuda()

    
# row_pointers1, \
# column_index1, \
# values, time = FS_Block_gpu.preprocess_gpu_fs(row_pointers, column_index, num_nodes, num_edges, window, wide,)
# print()

part = 32
row_pointers1, \
column_index1, \
values, \
t_window_rowTensor, \
t_atomicTensor, \
time = FS_Block_gpu.preprocess_gpu_fs_balance(row_pointers, column_index, num_nodes, num_edges, window, wide, part)
print()