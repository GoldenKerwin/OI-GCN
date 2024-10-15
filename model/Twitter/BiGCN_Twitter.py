import sys, os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
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
    mask = row != col
    edge_index_no_self_loops = edge_index[:, mask]
    edge_weight_no_self_loops = edge_weight[mask]
    row_no_self_loops = row[mask]
    col_no_self_loops = col[mask]
    adjacency_sparse = th.sparse_coo_tensor(edge_index_no_self_loops, edge_weight_no_self_loops, (num_nodes, num_nodes)).to(device)
    adjacency_dense = adjacency_sparse.to_dense()
    
    deg = scatter_add(edge_weight_no_self_loops, col_no_self_loops, dim=0, dim_size=num_nodes)  
    deg_inv_sqrt = deg.pow(-0.5)  
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  
    edge_weight_normalized = deg_inv_sqrt[row_no_self_loops] * edge_weight_no_self_loops * deg_inv_sqrt[col_no_self_loops]  
    normalized_adjacency_sparse = th.sparse_coo_tensor(edge_index_no_self_loops, edge_weight_normalized, (num_nodes, num_nodes)).to(device)  
    normalized_adjacency_dense = normalized_adjacency_sparse.to_dense()  

    return normalized_adjacency_dense, adjacency_dense



def normalize_sibling_matrix(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    adj_matrix = adj_matrix.float() 
    sibling_matrix = th.mm(adj_matrix, adj_matrix)
    sibling_matrix[range(num_nodes), range(num_nodes)] = 0
    sibling_matrix = sibling_matrix.to(device)
    deg = sibling_matrix.sum(dim=1)  
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  
    deg_inv_sqrt_matrix = th.diag(deg_inv_sqrt) 
    sibling_matrix = deg_inv_sqrt_matrix @ sibling_matrix @ deg_inv_sqrt_matrix 
    return sibling_matrix



class rumorGCN1(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(rumorGCN1, self).__init__()
        self.conv1 = i_GCNConv_neighbor(in_feats, hid_feats)
        self.conv2 = GCNConv(in_feats, hid_feats)
        self.conv3 = GCNConv(2*(hid_feats)+in_feats, out_feats)

    def forward(self, data, x, edge_index, adj):
        x = x.float().to(device)
        adj = adj.to(device)
        x1 = x.clone()
        
        x_neighbor = self.conv1(x, adj)
        x_neighbor = F.relu(x_father)
        # Define a small constant (epsilon) to avoid taking the square root of zero  
        epsilon = np.finfo(np.float32).tiny
        x_neighbor = th.sqrt(x_neighbor + epsilon)
        
        x_gcn = self.conv2(x, edge_index)
        x = th.cat((x_neighbor, x_gcn), dim=1)
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
        self.conv3 = GCNConv(2*(hid_feats)+in_feats, out_feats)

    def forward(self, data, x, edge_index, adj, sibling_adj):
        x = x.float().to(device)
        adj = adj.to(device)
        x1 = x.clone()

        x_sible = self.conv1(x, adj, sibling_adj)
        x_sible = F.relu(x_sible)
        epsilon = np.finfo(np.float32).tiny
        x_sible = th.sqrt(x_sible + epsilon)
        x_gcn = self.conv2(x, edge_index)
        x = th.cat((x_sible, x_gcn), dim=1)
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
        self.fc = th.nn.Linear(4*hid_feats+2*out_feats, 4)

    def forward(self, data, x, edge_index, adj, sibling_adj):
        GCN1_x = self.rumorGCN1(data, x, edge_index, adj)
        GCN2_x = self.rumorGCN2(data, x, edge_index, adj, sibling_adj)
        x = th.cat((GCN1_x, GCN2_x), dim=1)
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
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            edge_index = to_undirected(Batch_data.edge_index.to(device))
            x = Batch_data.x.to(device)
            adj, unnorm_adj = normalize_adjacency(x, edge_index)
            sibling_adj = normalize_sibling_matrix(unnorm_adj)
            out_labels = model(Batch_data, x, edge_index, adj, sibling_adj)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter, epoch, batch_idx, loss.item(), train_acc))
            batch_idx = batch_idx + 1
            del Batch_data, edge_index, x, adj, out_labels, finalloss, loss
            th.cuda.empty_cache()
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            edge_index = to_undirected(Batch_data.edge_index.to(device))
            x = Batch_data.x.to(device)
            adj, unnorm_adj = normalize_adjacency(x, edge_index)
            sibling_adj = normalize_sibling_matrix(unnorm_adj)
            val_out = model(Batch_data, x, edge_index, adj, sibling_adj)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
            del Batch_data, edge_index, x, adj, val_out, val_loss, val_pred
            th.cuda.empty_cache()
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses), np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1), np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2), np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3), np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4), np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2), np.mean(temp_val_F3), np.mean(temp_val_F4), np.mean(temp_val_Acc_all), model, 'BiGCN', dataname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4

lr = 0.0005
weight_decay = 1e-4
patience = 10
n_epochs = 200
batchsize = 128
TDdroprate = 0.2
BUdroprate = 0.2
datasetname = sys.argv[1]  # "Twitter15"、"Twitter16"
iterations = int(sys.argv[2])
model = "GCN"

test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic, fold0_x_test, fold0_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic, fold1_x_test, fold1_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic, fold2_x_test, fold2_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic, fold3_x_test, fold3_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic, fold4_x_test, fold4_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
    test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))