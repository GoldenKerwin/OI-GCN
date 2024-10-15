import sys, os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
from sible import Interaction_GraphConvolution as i_GCNConv_sible
from neighbor import Interaction_GraphConvolution as i_GCNConv_neighbor
from torch_geometric.utils import to_undirected
import torch.nn as nn
from torch_scatter import scatter_add
from torch.multiprocessing import freeze_support
import random

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

set_seed(42)

def normalize_adjacency(x, edge_index):
    num_nodes = x.shape[0]
    device = x.device  
    edge_index = edge_index.to(device)
    edge_weight = th.ones((edge_index.size(1), ), dtype=th.float32).to(device)
    row, col = edge_index
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight_normalized = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    # Use sparse matrices to reduce video memory
    normalized_adjacency_sparse = th.sparse.FloatTensor(edge_index, edge_weight_normalized, th.Size([num_nodes, num_nodes])).to(device)
    return  normalized_adjacency_sparse



class rumorGCN1(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(rumorGCN1, self).__init__()
        self.conv1 = i_GCNConv_neighbor(in_feats, hid_feats)
        self.conv2 = GCNConv(in_feats, hid_feats)
        self.conv3 = GCNConv(2*(hid_feats)+ in_feats, out_feats)
        self.batchnorm1 = th.nn.BatchNorm1d(2*hid_feats)
        self.batchnorm2 = th.nn.BatchNorm1d(hid_feats)

    def forward(self, data, x, edge_index, adj):
        x = x.float().to(device)
        adj = adj.to(device)
        x1 = x.clone()
        x_neighbor = self.conv1(x, adj)
        x_neighbor = F.relu(x_neighbor)
        epsilon = np.finfo(np.float32).tiny
        x_neighbor = th.sqrt(x_neighbor + epsilon)
        x_gcn = self.conv2(x, edge_index)
        x = th.cat((x_neighbor, x_gcn), 1)
        x2 = x.clone()
        
        root_indices = data.rootindex[data.batch]
        root_extend = x1[root_indices]
        x = th.cat((x, root_extend), dim=1)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        root_extend = x2[root_indices]
        x = th.cat((x, root_extend), dim=1)

        x = scatter_mean(x, data.batch, dim=0)

        return x


class rumorGCN2(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(rumorGCN2, self).__init__()
        self.conv1 = i_GCNConv_sible(in_feats, hid_feats)
        self.conv2 = GCNConv(in_feats, hid_feats)
        self.conv3 = GCNConv(2*(hid_feats) + in_feats, out_feats)

    def forward(self, data, x, edge_index, adj):
        x = x.float().to(device)
        adj = adj.to(device)
        x1 = x.clone()

        x_sible = self.conv1(x, adj)
        x_sible = F.relu(x_sible)
        epsilon = np.finfo(np.float32).tiny
        # Define a small constant (epsilon) to avoid taking the square root of zero  
        x_sible = th.sqrt(x_sible + epsilon)
        x_gcn = self.conv2(x, edge_index)
        x = th.cat((x_sible, x_gcn), 1)
        x2 = x.clone()
        
        root_indices = data.rootindex[data.batch]
        root_extend = x1[root_indices]
        x = th.cat((x, root_extend), dim=1)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        root_extend = x2[root_indices]
        x = th.cat((x, root_extend), dim=1)
        
        x = scatter_mean(x, data.batch, dim=0)

        return x

class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.rumorGCN1 = rumorGCN1(in_feats, hid_feats, out_feats)
        self.rumorGCN2 = rumorGCN2(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear(2 * (out_feats + hid_feats) + 2*hid_feats, 2)

    def forward(self, data, x, edge_index, adj):
        GCN1_x = self.rumorGCN1(data, x, edge_index, adj)
        GCN2_x = self.rumorGCN2(data, x, edge_index, adj)
        x = th.cat((GCN1_x, GCN2_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter):
    model = Net(5000, 512, 256).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sible_params = list(map(id, model.rumorGCN2.conv1.parameters()))
    sible_params += list(map(id, model.rumorGCN2.conv2.parameters()))
    sible_params += list(map(id, model.rumorGCN2.conv3.parameters()))
    base_params = filter(lambda p: id(p) not in sible_params, model.parameters())
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.rumorGCN2.conv1.parameters(), 'lr': lr/5 },
        {'params': model.rumorGCN2.conv2.parameters(), 'lr': lr/5 },
        {'params': model.rumorGCN2.conv3.parameters(), 'lr': lr/5 }
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    th.cuda.empty_cache()

    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=8)  # 减少num_workers
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=8)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        optimizer.zero_grad()
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            edge_index = to_undirected(Batch_data.edge_index).to(device)
            x = Batch_data.x.to(device)
            adj = normalize_adjacency(x, edge_index)
            out_labels = model(Batch_data, x, edge_index, adj)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss 
            loss.backward()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_loss.append(loss.item())
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter, epoch, batch_idx, loss.item(), train_acc))
            batch_idx += 1

            del Batch_data, edge_index, x, adj, out_labels, finalloss, loss
            th.cuda.empty_cache()

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        with th.no_grad():
            for Batch_data in tqdm_test_loader:
                Batch_data.to(device)
                edge_index = to_undirected(Batch_data.edge_index).to(device)
                x = Batch_data.x.to(device)
                adj = normalize_adjacency(x, edge_index)
                
                val_out = model(Batch_data, x, edge_index, adj)
                val_loss = F.nll_loss(val_out, Batch_data.y)
                temp_val_losses.append(val_loss.item())
                _, val_pred = val_out.max(dim=1)
                correct = val_pred.eq(Batch_data.y).sum().item()
                val_acc = correct / len(Batch_data.y)
                Acc_all, Acc1, Prec1, Recll1, F1_score, Acc2, Prec2, Recll2, F2_score = evaluationclass(
                    val_pred, Batch_data.y)
                temp_val_Acc_all.append(Acc_all)
                temp_val_Acc1.append(Acc1)
                temp_val_Prec1.append(Prec1)
                temp_val_Recll1.append(Recll1)
                temp_val_F1.append(F1_score)
                temp_val_Acc2.append(Acc2)
                temp_val_Prec2.append(Prec2)
                temp_val_Recll2.append(Recll2)
                temp_val_F2.append(F2_score)
                temp_val_accs.append(val_acc)

                del Batch_data, edge_index, x, adj, val_out, val_loss, val_pred
                th.cuda.empty_cache()

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model, 'BiGCN', "weibo")
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1_score = np.mean(temp_val_F1)
        F2_score = np.mean(temp_val_F2)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1_score = early_stopping.F1
            F2_score = early_stopping.F2
            break
        model.train()
        th.cuda.empty_cache()
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1_score, acc2, pre2, rec2, F2_score

def main():
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 16
    TDdroprate = 0.2
    BUdroprate = 0.2
    datasetname = "Weibo"
    iterations = 1
    test_accs, ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1_list, F2_list = [], [], [], [], [], [], [], [], []
    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train,  \
        fold3_x_test, fold3_x_train,  \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)
        treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               TDdroprate,BUdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs,accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
    train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
    ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    ACC2.append((acc2_0 + acc2_1 +acc2_2 + acc2_3 +acc2_4) / 5)
    PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    F1_list.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    F2_list.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    print("weibo:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1_list) / iterations, sum(F2_list) / iterations))

if __name__ == '__main__':
    freeze_support()
    main()
