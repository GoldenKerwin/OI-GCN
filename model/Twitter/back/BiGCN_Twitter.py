import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th  #用于构建和训练神经网络
from torch_scatter import scatter_mean  #用于在 PyTorch 中执行散射操作
import torch.nn.functional as F #PyTorch 的函数库
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader #导入了 PyTorch Geometric 库中的 DataLoader 类，该类用于加载图形数据
from tqdm import tqdm   #一个快速、可扩展的 Python 进度条库
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv  #该类用于实现图卷积网络（GCN）的卷积层
import copy #该模块提供了复制 Python 对象的功能

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,device):
        #调用了父类 Module 的初始化函数
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        #定义了两个图卷积层，第一个图卷积层将输入特征转换为隐藏特征，第二个图卷积层将隐藏特征和输入特征的连接转换为输出特征。
        self.device = device

    #在前向传播过程中，对节点特征进行两次图卷积操作，并在每次操作后都将根节点的特征添加到每个节点的特征中
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #提取出节点特征 x 和边索引 edge_index
        
        x1=copy.copy(x.float())
        #创建了节点特征 x 的一个浮点数副本 x1
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        #创建了一个全零张量 root_extend，其形状为 (len(data.batch), x1.size(1))，并将其移动到指定的设备上
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        #这段代码遍历每个批次，对于每个批次，它都会找出属于该批次的节点，并将这些节点在 root_extend 中对应的行设置为根节点的特征
        x = th.cat((x,root_extend), 1)
        #这行代码将转换后的节点特征 x 和 root_extend 沿着列方向进行拼接

        x = F.relu(x)
        #ReLU 被应用在特征融合之后，以增加模型的非线性
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        #可能是因为在这一阶段，模型已经学习到了足够的特征表示，无需再进行特征融合。
        #先应用 ReLU 可以增加模型的非线性，然后再进行特征融合，可以进一步提取特征
        x= scatter_mean(x, data.batch, dim=0)
        #对每个图的节点特征进行平均池化

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,device):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,device):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats,device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats,device)
        #创建了两个图卷积网络，一个是 TDrumorGCN，另一个是 BUrumorGCN
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)
        #定义了一个全连接层，该层将两个图卷积网络的输出特征连接起来，并将其转换为四个输出特征
        self.device = device


    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        #执行前向传播
        x = th.cat((BU_x,TD_x), 1)
        #将两个图卷积网络的输出特征连接起来
        x=self.fc(x)
        #通过全连接层进行转换
        x = F.log_softmax(x, dim=1)
        #应用 log softmax 函数得到最终的输出
        return x





def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,device):
    #batchsize：每个批次的大小，iter：迭代次数，weight_decay：权重衰减
    model = Net(5000,64,64, device).to(device)
    #初始化了一个网络模型，模型的输入维度为5000，隐藏层的维度为64，输出层的维度也为64，并将模型放到指定的设备上。
    
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    #获取了模型中BUrumorGCN部分的第一层和第二层的参数，并将它们的id存储到BU_params列表中
    # base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    def filter_params(p):
        return id(p) not in BU_params
    base_params = filter(filter_params, model.parameters())
    #过滤出模型中不在BU_params列表中的参数，也就是除了BUrumorGCN部分的第一层和第二层之外的其他参数
    
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    #定义了优化器，其中base_params参数的学习率为lr，BUrumorGCN部分的第一层和第二层的学习率为lr/5。
    
    model.train()   #将模型设置为训练模式
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    #用于存储训练损失、验证损失、训练精度和验证精度
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    #初始化了一个早停策略，当验证损失在连续patience个epoch中没有改善时，训练将停止
    
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        #加载训练集和测试集
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        #创建了训练数据和测试数据的数据加载器，并指定批次大小、是否打乱顺序、使用的线程数
        avg_loss = []
        avg_acc = []
        #用于存储每个批次的损失和精度
        batch_idx = 0
        #初始化了批次索引
        tqdm_train_loader = tqdm(train_loader)
        #这行代码创建了一个进度条，用于显示训练的进度
        for Batch_data in tqdm_train_loader:
            #循环的次数为训练数据的批次数
            Batch_data.to(device)
            #将训练数据放到指定的设备上
            out_labels= model(Batch_data)
            #通过模型对批次数据进行预测，得到预测的标签
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            #计算损失函数
            loss=finalloss
            optimizer.zero_grad()
            #将优化器中的梯度清零
            loss.backward()
            #对损失进行反向传播，计算梯度
            avg_loss.append(loss.item())
            #将当前的损失添加到平均损失的列表中
            optimizer.step()
            #更新模型的参数
            _, pred = out_labels.max(dim=-1)
            #得到预测标签中概率最大的类别作为预测的类别
            correct = pred.eq(Batch_data.y).sum().item()
            #计算预测正确的样本数
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            #计算训练精度
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            #打印当前的迭代次数、epoch、批次索引、训练损失和训练精度
            batch_idx = batch_idx + 1
            #将批次索引加1，表示进入下一批次
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        #用于存储四个类别的精度、查准率、查全率和F1值

        model.eval()
        #将模型设置为评估模式

        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            #循环的次数为验证数据的批次数
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            #调用evaluation4class函数计算四个类别的精度、查准率、查全率和F1值
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            #将计算得到的各种性能指标添加到对应的列表中
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        #打印当前的epoch、验证损失和验证精度

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        #计算了四个类别的平均精度、查准率、查全率和F1值，并将它们存储到res列表中
        print('results:', res)

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        #调用早停策略，并将当前的模型、模型名称、数据集名称、四个类别的平均精度、平均F1值等信息存储到early_stopping对象中

        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        #五行代码计算了验证精度和四个类别的F1值的平均值
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

# lr=0.0005
# weight_decay=1e-4
# patience=10
# n_epochs=200
# batchsize=128
# TDdroprate=0.2
# BUdroprate=0.2
# datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
# iterations=int(sys.argv[2])
# model="GCN"
# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# test_accs = []
# NR_F1 = []
# FR_F1 = []
# TR_F1 = []
# UR_F1 = []
# for iter in range(iterations):
#     fold0_x_test, fold0_x_train, \
#     fold1_x_test,  fold1_x_train,  \
#     fold2_x_test, fold2_x_train, \
#     fold3_x_test, fold3_x_train, \
#     fold4_x_test,fold4_x_train = load5foldData(datasetname)
#     treeDic=loadTree(datasetname)
#     train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
#                                                                                                fold0_x_test,
#                                                                                                fold0_x_train,
#                                                                                                TDdroprate,BUdroprate,
#                                                                                                lr, weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter)
#     train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
#                                                                                                fold1_x_test,
#                                                                                                fold1_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter)
#     train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
#                                                                                                fold2_x_test,
#                                                                                                fold2_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter)
#     train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
#                                                                                                fold3_x_test,
#                                                                                                fold3_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter)
#     train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
#                                                                                                fold4_x_test,
#                                                                                                fold4_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter)
#     test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
#     NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
#     FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
#     TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
#     UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
# print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
#     sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))

def main():
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
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    for iter in range(iterations):
        #循环的次数为迭代次数
        fold0_x_test, fold0_x_train, \
            fold1_x_test, fold1_x_train, \
            fold2_x_test, fold2_x_train, \
            fold3_x_test, fold3_x_train, \
            fold4_x_test, fold4_x_train = load5foldData(datasetname)
        #加载了五折交叉验证的数据
        treeDic = loadTree(datasetname)
        #加载了树形结构的数据
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,
                                                                                                   TDdroprate,
                                                                                                   BUdroprate,
                                                                                                   lr, weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter, device)
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   TDdroprate,
                                                                                                   BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter, device)
        train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   TDdroprate,
                                                                                                   BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter, device)
        train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   TDdroprate,
                                                                                                   BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter, device)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                   TDdroprate,
                                                                                                   BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter, device)
        #会对每一折的数据进行训练和验证，然后将结果存储到对应的列表中
        test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
        NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
        #计算了五折交叉验证的平均测试精度和四个类别的F1值，并将它们添加到对应的列表中
    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
        sum(UR_F1) / iterations))
    #打印了平均测试精度和四个类别的F1值

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    #freeze_support()函数是为了让这个脚本在Windows平台上可以被多进程正确地使用
    main()