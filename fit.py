#%%
import torch
from Net import Net
from Data import Data
from Train import fit
import matplotlib.pyplot as plt

data_set = Data(128)
x, y = data_set.get()
matrix = data_set.matrix()
Layers = [1, 16, 16, 16, 16, 1]
net = Net(Layers)
learning_rate = 0.01
epochs = 500
init_lr = 0.1
decay_rate = 1
# data_set.plot_data()
fit(x, y, net, epochs, init_lr, decay_rate, diagram=True)
# todo: tell Yousef that cross entropy doesn't work

# print (net(x))
# plt.plot (x, net(x))