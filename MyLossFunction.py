# %%
import torch
import numpy as np

'''
Define my loss functions
cross_entropy_loss function 
mse_loss function
and the loss function form of free energy
yhat: the real output of NN
y: traget output of NN
x: input of NN
@Author: Naicheng Deng
'''


def cross_entropy_loss(yhat, y):
    L = len(yhat)
    loss = - (1 / L) * (torch.mm(y.T, torch.log(yhat))
                        + torch.mm((1 - y).T, torch.log(1 - yhat)))
    return loss


def mse_loss(yhat, y):
    loss = torch.mean((y - yhat) ** 2)

    return loss


def free_energy (yhat, rho, x, simpson_matrix, matrix, lambd):
    f1 = torch.mm (yhat, yhat.T) * matrix
    dx = x[1] - x[0]
    first_term = (dx/3) * torch.mm(simpson_matrix,yhat * torch.log(rho * yhat))
    second_term = (rho/2) * (dx/3)**2 * torch.mm(torch.mm(simpson_matrix, f1), simpson_matrix.T)
    third_term = lambd * (((dx/3) *(torch.mm(simpson_matrix,yhat))) - 1 ) **2
    loss = first_term + second_term + third_term
    return loss
