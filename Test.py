import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
from Mesh import param
import Model

def bte_test(x,mu,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,a,index,path,device):
    net0=Model.NNet(8,10,60).to(device)
    net0.load_state_dict(torch.load(path+'model.10.pt',map_location=device)['state_dict'])
    net0.eval()

    net1=Model.NNet(4,10,60).to(device)
    net1.load_state_dict(torch.load(path+"model1.pt",map_location=device)['state_dict'])
    net1.eval()

    p = np.concatenate((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))),0)
    v,tau,C=param(np.tile(k,(Np,1)),p,Tr)
    v=torch.FloatTensor(v).to(device)
    tau=torch.FloatTensor(tau).to(device)
    C=torch.FloatTensor(C).to(device)

    k=k/(np.pi*2/a)
    mu=torch.FloatTensor(mu).repeat(Nk,1).to(device)
    vt0 = torch.FloatTensor(vt0).repeat(1, Ns).reshape(-1, 1).to(device)
    vt1 = torch.FloatTensor(vt1).repeat(1, Ns).reshape(-1, 1).to(device)
    k = torch.FloatTensor(k).repeat(1, Ns).reshape(-1, 1).to(device)
    w = torch.FloatTensor(w).to(device)
    wk = np.pi * 2 / a / Nk
    TC = (1 / 3) * torch.sum(C * v ** 3 * tau * wk) / (dT * 2) * 1e11
    deltaT = np.zeros((Nx, len(L)))
    q = np.zeros((Nx, len(L)))
    tic = time.time()
    for j in range(len(L)):
        for i in range(Nx):
            x1 = torch.FloatTensor(x[i]).repeat(Ns * Nk, 1).to(device)
            L1 = torch.FloatTensor(L[j]).repeat(Ns * Nk, 1).to(device)

            eEq = net1(torch.cat((x1, L1), 1)) * dT
            e0_in = torch.cat((x1, mu, k, vt0, L1, torch.zeros_like(x1)), 1)
            e1_in = torch.cat((x1, mu, k, vt1, L1, torch.ones_like(x1)), 1)
            e0 = net0(e0_in) * (10 ** vt0) / (10 ** L1) * dT
            e1 = net0(e1_in) * (10 ** vt1) / (10 ** L1) * dT
            e = torch.cat((e0 + eEq, e0 + eEq, e1 + eEq), 0)

            sum_e = torch.matmul(e.reshape(-1, Ns), w).reshape(-1, 1)
            T = torch.sum(sum_e * C * wk / tau * v / (4 * np.pi)) / torch.sum(C / tau * wk * v)
            sum_ve = torch.matmul(e.reshape(-1, Ns), w * mu[0:Ns].reshape(-1, 1)).reshape(-1, 1)

            q[i, j] = torch.sum(sum_ve * C * wk * v ** 2 / (4 * np.pi)).cpu().data.numpy() / TC.cpu().data.numpy() * (
                        10 ** L[j])
            deltaT[i, j] = T.cpu().data.numpy()

    toc = time.time()
    elapseTime = toc - tic
    print("elapse time = ", elapseTime)
    np.savez(str(int(index))+'Kn_1d_ng',x = x,T = (deltaT+dT)/(2*dT),q = np.mean(q,axis=0).T,L = L)


