#224 0.9299 200epoch
#su 91.88 200epoch
#all 99.08 200epoch
#label 71.17/94.21/81.08 200epoch

import os.path as osp
import argparse
import data_deal as dd
import create_dataset as cd#该py文件的MyOwnDataset类
import torch_geometric.utils as tu

import torch
import torch.nn.functional as F#函数模块
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T# PyG同样有其自己的transform操作，使用Data对象作为输入，返回一个新的Data对象。
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import data_deal as dd
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
epoch_list=[]
value_list=[]
type_list=[]
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
dataset=cd.MyOwnDataset(transform=T.NormalizeFeatures())#归一化特征
dataset.shuffle()#dataset[0]因为有get方法，可以使用[0]作为idx传入get方法，获得data_0.pt
data=dataset[0]#data对象，包含处理后的特征，包含节点特征和邻接特征
print(data)
data.edge_attr=data.edge_attr.flatten()#矩阵压缩存储，要按行展开
import pdb
# pdb.set_trace()
# PyG提供了一个名为GDC(Graph diffusion convolution , 图扩散卷积)的图数据预处理方法，
# 其结合了message passing和spectral methods优点，可以减少图中噪音的影响，
# 可以在有监督和无监督任务的各种模型以及各种数据集上显着提高性能，并且GDC不仅限于GNN，
# 还可以与任何基于图的模型或算法（例如频谱聚类）轻松组合，而无需对其进行任何更改或影响其计算复杂性。
if args.use_gdc:#None #不使用默认的gdc
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

# in_channels：输入通道，比如节点分类中表示每个节点的特征数。
# out_channels：输出通道，最后一层GCNConv的输出通道为节点类别数（节点分类）。
# improved：如果为True表示自环增加，也就是原始邻接矩阵加上2I而不是I，默认为False。
# cached：如果为True，GCNConv在第一次对邻接矩阵进行归一化时会进行缓存，以后将不再重复计算。
# add_self_loops：如果为False不再强制添加自环，默认为True。
# normalize：默认为True，表示对邻接矩阵进行归一化。
# bias：默认添加偏置。
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)#第一层卷积
        self.conv2 = GCNConv(16, 2, cached=True,
                             normalize=not args.use_gdc)#第二层卷积
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self,data):#前向传播
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr#
        x = F.relu(self.conv1(x, edge_index, edge_weight))#第一层激活函数，将特征和邻接矩阵编号和权重一起训练
        x = F.dropout(x, training=self.training)#
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)#分类


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)#设置训练环境
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)#优化器
], lr=0.01)  # Only perform weight-decay on first convolution.

# params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
# lr (float, 可选) – 学习率（默认：1e-3）
# betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
# eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
# weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）

def train():
    model.train()#训练模式
    optimizer.zero_grad()#设置优化器的梯度
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()#反向传播
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()#验证模式
    logits, accs = model(data), []#在深度学习中，logits就是最终的全连接层的输出，而非其本意。通常神经网络中都是先有logits，而后通过sigmoid函数或者softmax函数得到概率p的，所以大部分情况下都无需用到logit函数的表达式。
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        print(tu.precision(pred, data.y[mask], 2))
        print(tu.recall(pred, data.y[mask], 2))
        print(tu.f1_score(pred, data.y[mask], 2))
        print('\n')
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    value_list.append(tu.precision(pred, data.y[mask], 2)[1].item())
    value_list.append(tu.recall(pred, data.y[mask], 2)[1].item())
    value_list.append(tu.f1_score(pred, data.y[mask], 2)[1].item())
    type_list.append('precision')
    type_list.append('recall')
    type_list.append('f1')
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 501):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    epoch_list.append(epoch)
    epoch_list.append(epoch)
    epoch_list.append(epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
dd.plot_solute(epoch_list,value_list,type_list)
#epoch 500
_, pred = model(data).max(1)
print(pred)
import csv
with open('model_gcn_precision.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','precision','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i],type_list[3*i],epoch_list[3*i]])
with open('model_gcn_recall.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','recall','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i+1],type_list[3*i+1],epoch_list[3*i+1]])
with open('model_gcn_f1.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','recall','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i+2],type_list[3*i+2],epoch_list[3*i+2]])
print('gcn')
# dd.get_pred(pred,data.y)
#1pre0.9679 f1 0.6294 recall 0.4664
#2pre 0.9237 f10.6747 recall 0.5317
#3pre0.9675 f10.6190 recall 0.4551

