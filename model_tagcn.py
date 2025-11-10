#224 97.69 200epoch
#su 99.51 200epoch
#all 99.08 200epoch
#label 72.74/95.80/82.69 200epoch

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import TAGConv
import create_dataset as cd
import torch_geometric.utils as tu
import data_deal as dd
import numpy as np
epoch_list=[]
value_list=[]
type_list=[]
import pandas as pd
# dataset = 'Cora'
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = cd.MyOwnDataset(transform=T.NormalizeFeatures())
data = dataset[0]
# print(type(data))
# print(data)
# print(len(data))

for att in dir(data):
    print (att, getattr(data,att),len(att),type(att),str(att))
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# def visualize(h, color):
#     z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
#     plt.figure(figsize=(10,10))
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = TAGConv(dataset.num_features, 16)
        self.conv2 = TAGConv(16, 2)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)#正确的
        # return F.log_softmax(x, dim=1),x#错误的

#Data(x=[332191, 3], edge_index=[2, 197687], edge_attr=[197687, 1], y=[332191], train_mask=[332191], val_mask=[332191], test_mask=[332191])
#使用的是Data类，x是节点个数和种类，edge_index是2和边数，edge_attr是边数和1，y是节点个数，train_mask是节点个数




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#可视化分类结果：
# _,out = model(data)
# visualize(out, color=data.y)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(data), []
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
#epoch 1000
torch.save(model.state_dict(),'test.pt')


# model = Net().to(device)
# model.load_state_dict(torch.load('test.pt'))#加载模型
model.eval()#进行预测
_, pred = model(data).max(dim=1)#取出预测结果，预测的分类就是二分类中概率较高的那一类
pred = list(pred)#预测的结果
y = list(data.y)#原来的节点分类
predict = pd.DataFrame({'pred':pred,'y':y})
pd.set_option('display.max_rows', None)
# print(predict)
predict.to_csv('predict-gdc2.csv')#存储预测分类结果和原来的分类

import csv
with open('model_tagcn_precision.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','precision','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i],type_list[3*i],epoch_list[3*i]])
with open('model_tagcn_recall.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','recall','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i+1],type_list[3*i+1],epoch_list[3*i+1]])
with open('model_tagcn_f1.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['value','recall','epoch'])
    for i in range(0,500):
        writer.writerow([value_list[3*i+2],type_list[3*i+2],epoch_list[3*i+2]])

#0是normal 1是scam 2是unknown
# dd.get_pred(pred,data.y)
print('tagcn')
dd.plot_solute(epoch_list,value_list,type_list)

# dd.print_solute(epoch_list,value_list,type_list,'tagcn list.txt')
# node_df=dd.read_csv(r'small/su_node.csv')
# edge_df=dd.read_csv(r'small/su_edge.csv')
# node_fea,type_list=dd.get_node_fea(node_df)
# train_mask,val_mask,test_mask=dd.split_data(type_list,[6,2,2])
# edge_list=dd.id_to_num(node_df,edge_df)
# edge_fea=dd.get_edge_fea(edge_df)
# data1=dd.make_torch_data1(node_fea,edge_list,type_list,train_mask,val_mask,test_mask,edge_fea)
# data1 = data1.to(device)
# model.eval()
# logits, accs = model(), []


# epoch 500
#1 pre0.9285 f10.8819 recall 0.8403
#2 pre0.9250 f10.8976 recall 0.8718
#3 pre0.9318 f10.8896 recall 0.8570




