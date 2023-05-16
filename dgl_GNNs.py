import dgl.function as fn
import torch.optim
from dgl import LaplacianPE
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv, SAGEConv, GCN2Conv, GATConv, GATv2Conv, EdgeConv, SGConv, APPNPConv, GINEConv, \
    GatedGraphConv, GMMConv, ChebConv, DotGatConv, TWIRLSConv, PNAConv, DGNConv, TAGConv, GNNExplainer
import math
import dgl
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn import metrics

import matplotlib.pyplot as plt
from torch_geometric.data import Data

# use GPU
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the precision of tensor to the precision decimal place
torch.set_printoptions(precision=8)

torch.set_printoptions(profile="full")
# torch.set_printoptions(profile="default") # reset
# np.set_printoptions(suppress=True) //The array display is incomplete
torch.set_printoptions(sci_mode=False)

## Graph construction

# Construction of road network diagram (edge index of diagram)
osm = pd.read_csv('Adjacency relation.csv')  # Road adjacency read
edge_index = []
O = osm['RoadID'].values.tolist()
D = osm['RoadID_2'].values.tolist()
for index in range(len(O)):
    edge_index.append([O[index] - 1, D[index] - 1])
edge_index = torch.tensor(edge_index, dtype=torch.long)  # The tensor used as an index must be a long tensor, a byte tensor, or a Boolean tensor

# The x construction of the graph
embd = pd.read_csv('The semantic segmentation features of each road after processing.csv')  # Road semantic segmentation feature reading, a total of 5075 roads
data_x = []
for i in range(5075):
    try:
        e = embd[(embd['RoadID']) == i + 1].values[0][2:21].tolist()
        data_x.append(e)
    except:
        data_x.append([0] * 19)  # Part of the road has no features and is filled with zeros

# The y construction of the graph
data_y_org = pd.read_csv('CGrade.csv')  # CO emissions read, 5,075 roads
data_y = data_y_org['TotalBreak'].values.tolist()
for t in range(5075):
    if math.isnan(data_y[t]) or data_y[t]==0:
        data_y[t] = 1  # If it is an empty nan, it is filled with 0


# Collate data format
x = torch.tensor(data_x, dtype=torch.float)  # Convert x to tensor and set it to be a float, or you'll get an error
y = torch.tensor(data_y, dtype=torch.float)  # Convert y to tensor and set it to be a float, or you'll get an error

# Build the input data set
data = Data(x=x,edge_index=edge_index.t().contiguous(),y=y).to(device)  # GPU computing




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


class GNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_class):
        super(GNN, self).__init__()
        torch.manual_seed(1234567)

        self.conv1 = SAGEConv(in_feats, h_feats,'lstm')
        self.conv2 = SAGEConv(h_feats, num_class,'lstm')

        # self.conv1=GraphConv(in_feats,h_feats)
        # self.conv2=GraphConv(h_feats,num_class)

        # self.conv1 = GCN2Conv(in_feats, layer=1, alpha=0.5,  project_initial_features=True, allow_zero_in_degree=True)
        # self.conv2 = GCN2Conv(in_feats, layer=2, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)

        # self.conv1 = GATConv(in_feats, h_feats,4)
        # self.conv2 = GATConv(h_feats*4, num_class,1)

        # self.conv1 = GATv2Conv(in_feats, h_feats,1)
        # self.conv2 = GATv2Conv(h_feats*1, num_class,7)

        # self.conv1 = EdgeConv(in_feats, h_feats)
        # self.conv2 = EdgeConv(h_feats, num_class)

        # self.conv1 = SGConv(in_feats, h_feats,2)
        # self.conv2 = SGConv(h_feats, num_class,1)

        # self.conv1 = ChebConv(in_feats, h_feats,  2)
        # self.conv2 = ChebConv(h_feats, num_class, 2)

        # self.conv1 = TWIRLSConv(in_feats,h_feats, h_feats,prop_step = 64)
        # self.conv2 = TWIRLSConv(h_feats,num_class, num_class,prop_step = 64)

        # self.conv1 = PNAConv(in_feats, h_feats, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
        # self.conv2 = PNAConv(h_feats, num_class,['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)

        # self.conv1 = DGNConv(in_feats, h_feats, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
        # self.conv2 = DGNConv(h_feats, num_class, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)

        # self.conv1 = TAGConv(in_feats, h_feats, k=2)
        # self.conv2 = TAGConv(h_feats, num_class,2)

    def forward(self, graph, feat, eweight=None):
        ## GCNv2Conv Need
        # res=feat
        # DGN
        # eig = graph.ndata['eig']
        # graph=dgl.sampling.sample_neighbors(g,10,-1,replace=False)

        h = self.conv1(graph, feat)
        h = F.relu(h)
        h = self.conv2(graph, h)
        # h=F.log_softmax(h, dim=1)

        graph.ndata['h'] = h
        if eweight is None:
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        else:
            graph.edata['w'] = eweight
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        return graph.ndata['h']


def train_and_pred(g, model):
    # DGNConv
    # transform = LaplacianPE(k=3, feat_name='eig')
    # g = transform(g)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004,weight_decay=5e-5) #[wd=5e-4] 0.005->0.617 || 0.004->0.627 || 0.003->0.618

    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat'].float()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # Subtract 1 from all the labels, because the subscript starts at 0
    print(len(labels[train_mask]) + len(labels[val_mask]) + len(labels[test_mask]))
    print(len(labels[train_mask]))
    print(len(labels[val_mask]))
    print(len(labels[test_mask]))

    labels[train_mask] = torch.tensor([j - 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j - 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j - 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    # Train 1000 times
    for e in range(1001):
        logits = model(g, features)

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask].long()).to(device)

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
            print('In epoch: {}, loss: {:.3f}, train_acc: {:.3f}, val_acc: {:.3f}(best {:.3f}), test_acc: {:.3f}(best {:.3f})'.format(
                    e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))


    model.eval()
    logits = model(g, features)
    pred = logits.argmax(1)

    pred[train_mask] = torch.tensor([j + 1 for j in np.array(pred[train_mask].cpu())], dtype=torch.long).to(device)
    pred[val_mask] = torch.tensor([j + 1 for j in np.array(pred[val_mask].cpu())], dtype=torch.long).to(device)
    pred[test_mask] = torch.tensor([j + 1 for j in np.array(pred[test_mask].cpu())], dtype=torch.long).to(device)

    labels[train_mask] = torch.tensor([j + 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j + 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j + 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    # Calculate each index
    MSE = metrics.mean_squared_error(pred.cpu(), labels.cpu())
    RMSE = metrics.mean_squared_error(pred.cpu(), labels.cpu()) ** 0.5
    MAE = metrics.mean_absolute_error(pred.cpu(), labels.cpu())
    MAPE = metrics.mean_absolute_percentage_error(pred.cpu(), labels.cpu())
    ME = metrics.max_error(pred.cpu(), labels.cpu())
    MSL = metrics.mean_squared_log_error(pred.cpu(), labels.cpu())
    print("MSE:{", MSE, "}RMSE:{", RMSE, "}MAE:{", MAE, "}MAPE:{", MAPE, "}ME:{", ME, "}MSL{", MSL, "}")

    # # draw the truth value of y
    plt.figure()
    plt.xlabel('RoadID')
    plt.ylabel('CO_Grade')
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.title('武汉市街道出租车CO排放强度预测', fontproperties='Microsoft YaHei')
    original, = plt.plot(labels.cpu(), c='b')

    pred, = plt.plot(pred.cpu(), color='g')
    plt.legend(handles=[original, pred], labels=['原始值', '预测值'], loc='best')
    # plt.show()


# Build data set
dataset = RoadGraphDataset()
g = dataset[0].to(device)

# Start training and predicting and mapping
model = GNN(g.ndata['feat'].shape[1], 16, 7).to(device)
train_and_pred(g, model)



