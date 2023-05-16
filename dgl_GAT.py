import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from dgl import LaplacianPE
from dgl.data import CoraGraphDataset, DGLDataset
from dgl.nn.pytorch import GraphConv, SAGEConv, GCN2Conv, GATConv, GATv2Conv, EdgeConv, SGConv, APPNPConv, GINEConv, \
    GatedGraphConv, GMMConv, ChebConv, DotGatConv, TWIRLSConv, PNAConv, DGNConv, TAGConv
import math
import dgl
import numpy as np
import torch
import pandas as pd
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# 用GPU
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# 设置tensor的精度为小数点后precision位
torch.set_printoptions(precision=8)

torch.set_printoptions(profile="full")
# torch.set_printoptions(profile="default") # reset
# np.set_printoptions(suppress=True) //数组显示不全
torch.set_printoptions(sci_mode=False)

## 图构建

# 路网关系图(图的edge_index)构建
osm = pd.read_csv('Adjacency relation.csv')  # 道路邻接关系读取
edge_index = []
O = osm['RoadID'].values.tolist()
D = osm['RoadID_2'].values.tolist()
for index in range(len(O)):
    edge_index.append([O[index] - 1, D[index] - 1])
edge_index = torch.tensor(edge_index, dtype=torch.long)  # 用作索引的张量必须是长张量、字节张量或布尔张量

# 图的x构建
embd = pd.read_csv('The semantic segmentation features of each road after processing.csv')  # 道路语义分割特征读取，一共有5075个道路
deep = pd.read_excel('计算好的道路深度_均值替换.xls')  # 道路深度特征读取，一共有5075个道路
data_x = []
for i in range(5075):

    try:
        e = embd[(embd['RoadID']) == i + 1].values[0][2:21].tolist()
        # d = np.array(MinMaxScaler(feature_range=(0, 1)).fit_transform(
        #     deep[(deep['RoadID']) == i + 1].values[0][1:20].reshape(-1, 1))).reshape(19).tolist()
        # d = deep[(deep['RoadID']) == i + 1].values[0][1:20].tolist()
        data_x.append(e)

    except:
        data_x.append([0] * 19)  # 其中一部分road没有特征，用0填充
        # data_x.append([0] * 38)  # 其中一部分road没有特征，用0填充

# 图的y构建
data_y_org = pd.read_csv('CGrade.csv')  # 碳排放量读取，5075个道路
# data_y = data_y_org['NatureBreak'].values.tolist()
data_y = data_y_org['TotalBreak'].values.tolist()
for t in range(5075):
    if math.isnan(data_y[t]) or data_y[t]==0:
        data_y[t] = 1  # 如果为空nan，则用0填充


# 整理数据格式
x = torch.tensor(data_x, dtype=torch.float)  # 把x转换为tensor，并设置为float浮点数，不然会报错
y = torch.tensor(data_y, dtype=torch.float)  # 把y转换为tensor，并设置为float浮点数，不然会报错

# 构建输入数据集
data = Data(x=x,edge_index=edge_index.t().contiguous(),y=y).to(device)  # 使用GPU计算




class RoadGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='road_network')

    def process(self):
        self.graph = dgl.graph((edge_index.t()[0], edge_index.t()[1]), num_nodes=data.num_nodes).to(device)
        # self.graph = dgl.add_self_loop(self.graph,fill_data='sum')

        # If your dataset is a node classification dataset, you will  need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = data.num_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        # for i in range(5075):
        #     if data_x[i][5] == -1:  # y中存在空值【0】则用mask遮蔽，不参加训练测试
        #         train_mask[i] = False
        #         test_mask[i] = False
        #         val_mask[i] = False

        self.graph.ndata['train_mask'] = train_mask.to(device)
        self.graph.ndata['val_mask'] = val_mask.to(device)
        self.graph.ndata['test_mask'] = test_mask.to(device)

        self.graph.ndata['feat'] = data.x
        self.graph.ndata['label'] = data.y

    def __getitem__(self, i):
        if i != 0:
            raise IndexError('This dataset has only one graph')
        return self.graph

    def __len__(self):
        return 1


# 定义GAT神经层
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # 数据
        self.g = g
        # 对应公式中1的 W，用于特征的线性变换
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 对应公式2中的 a, 输入拼接的zi和zj（2 * out_dim），输出eij（一个数值）
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 随机初始化需要学习的参数
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # 对应公式2中的拼接操作，即zi || zj
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # 拼接之后对应公式2中激活函数里的计算操作，即a(zi || zj)
        a = self.attn_fc(z2)
        # 算出来的值经过leakyReLU激活得到eij,保存在e变量中
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 汇聚信息，传递之前计算好的z（对应节点的特征） 和 e
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 对应公式3，eij们经过softmax即可得到特征的权重αij
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 计算出权重之后即可通过 权重αij * 变换后的特征zj 求和计算出节点更新后的特征
        # 不过激活函数并不在这里，代码后面有用到ELU激活函数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    # 正向传播方式
    def forward(self, h):
        # 对应公式1，先转换特征
        z = self.fc(h)
        # 将转换好的特征保存在z
        self.g.ndata['z'] = z
        # 对应公式2，得出e
        self.g.apply_edges(self.edge_attention)
        # 对应公式3、4计算出注意力权重α并且得出最后的hi
        self.g.update_all(self.message_func, self.reduce_func)
        # 返回并清除hi
        return self.g.ndata.pop('h')

# 定义多头注意力机制的GAT层
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        # 多头注意力机制的头数（注意力机制的数量）
        self.heads = nn.ModuleList()
        # 添加对应的注意力机制层，即GAT神经层
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge  # 使用拼接的方法，否则取平均

    def forward(self, h):
        # 获取每套注意力机制得到的hi
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 每套的hi拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 所有的hi对应元素求平均
            return torch.mean(torch.stack(head_outs))


class GCN(torch.nn.Module):
    def __init__(self,g, in_feats, h_feats, num_class):
        super(GCN, self).__init__()
        torch.manual_seed(1)

        self.conv1 = MultiHeadGATLayer(g,in_feats, h_feats,4)
        self.conv2 = MultiHeadGATLayer(g,h_feats*4, num_class,1)


    def forward(self, feat):

        h = self.conv1( feat)
        h=F.relu(h)
        h = self.conv2( h)
        # h=F.log_softmax(h, dim=1)

        return h


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004,weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=5e-4)
    # criterion = torch.nn.MSELoss().to(device)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat'].float()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # print(len(g))

    # print(labels[train_mask])
    # 所有标签减1，因为下标从0开始
    print(len(labels[train_mask]) + len(labels[val_mask]) + len(labels[test_mask]))
    labels[train_mask] = torch.tensor([j - 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j - 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j - 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    losses=[]

    for e in range(1001):
        logits = model(features)

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask].long()).to(device)
        # loss=F.nll_loss(logits[train_mask],labels[train_mask].long())

        # pred[train_mask] = torch.tensor([j +1 for j in np.array(pred[train_mask])], dtype=torch.long)
        # pred[val_mask] = torch.tensor([j + 1 for j in np.array(pred[val_mask])], dtype=torch.long)
        # pred[test_mask] = torch.tensor([j + 1 for j in np.array(pred[test_mask])], dtype=torch.long)

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_test_acc < test_acc:
            best_test_acc = test_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            losses.append(loss.cpu().detach().numpy())
            print('In epoch: {}, loss: {:.3f}, train_acc: {:.3f}, val_acc: {:.3f}(best {:.3f}), test_acc: {:.3f}(best {:.3f})'.format(
                    e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))

    loss_pd=pd.read_excel("loss记录.xlsx",index_col="epoch")
    loss_pd['GAT']=np.array(losses)
    # print(loss_pd.head(20))
    # print(loss_pd.head(-20))
    print(loss_pd.head(3))
    loss_pd.to_excel("loss记录.xlsx")

    model.eval()
    logits = model(features)
    pred = logits.argmax(1)

    # # 由于0值对MAPE影响非常大，所以去掉预测值和原始值中的0值
    # l_mask=torch.zeros(data.num_nodes, dtype=torch.bool)
    # for i in range(len(labels.cpu())):
    #     if labels[i]!=0 and pred[i]!=0:
    #         l_mask[i]=True
    #
    # # print(l_mask)
    # print(len(labels[l_mask]))

    pred[train_mask] = torch.tensor([j + 1 for j in np.array(pred[train_mask].cpu())], dtype=torch.long).to(device)
    pred[val_mask] = torch.tensor([j + 1 for j in np.array(pred[val_mask].cpu())], dtype=torch.long).to(device)
    pred[test_mask] = torch.tensor([j + 1 for j in np.array(pred[test_mask].cpu())], dtype=torch.long).to(device)

    labels[train_mask] = torch.tensor([j + 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j + 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j + 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    MSE = metrics.mean_squared_error(pred.cpu(), labels.cpu())
    RMSE = metrics.mean_squared_error(pred.cpu(), labels.cpu()) ** 0.5
    MAE = metrics.mean_absolute_error(pred.cpu(), labels.cpu())
    MAPE = metrics.mean_absolute_percentage_error(pred.cpu(), labels.cpu())
    ME = metrics.max_error(pred.cpu(), labels.cpu())
    MSL = metrics.mean_squared_log_error(pred.cpu(), labels.cpu())
    print("MSE:{", MSE, "}RMSE:{", RMSE, "}MAE:{", MAE, "}MAPE:{", MAPE, "}ME:{", ME, "}MSL{", MSL, "}")


dataset = RoadGraphDataset()
g = dataset[0].to(device)

print(g.ndata['feat'].shape[1])
model = GCN(g,g.ndata['feat'].shape[1], 16, 7).to(device)
train(g, model)
