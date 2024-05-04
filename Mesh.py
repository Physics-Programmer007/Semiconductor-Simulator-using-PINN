import numpy as np

def oned_mesh(Nx,Ns,Nk,a):
    x = np.linspace(0, 1, Nx + 2)[1:Nx + 1].reshape(-1, 1)

    mu, w = np.polynomial.legendre.leggauss(Ns)
    mu = mu.reshape(-1, 1)
    w = w.reshape(-1, 1) * 2 * np.pi

    k = np.linspace(0, 1, Nk * 2 + 1)[1:Nk * 2 + 1].reshape(-1, 1)
    k = k[np.arange(0, Nk * 2 - 1, 2)] * np.pi * 2 / a

    return x, mu, w, k

def param(k,p,T):
    hbar = 1
    hkB = (1.054572 / 1.380649) * 1e2
    c10 = 5.23
    c20 = -2.26
    c11 = 9.01
    c21 = -2.0
    Ai = 1.498e7
    BL = 1.18e2
    BT = 8.708
    BU = 2.89e8
    a = 5.431

    c1 = (c11 - c10) * p + c10
    c2 = (c21 - c20) * p + c20
    om = c1 * k.T + c2 * (k ** 2).T
    v = c1 + 2 * c2 * k.T

    step = np.heaviside(k - np.pi / a, 1)
    ti = Ai * om ** 4
    t1 = BL * om ** 2 * T ** 3
    t01 = BT * om * T ** 4
    t02 = BU * om ** 2 / np.sinh(hkB * om / T)
    t0 = t01.T * (1 - step) + t02.T * step
    tNU = (t1 - t0.T) * p + t0.T
    tau = 1 / (ti + tNU)  # s

    dfeq = hkB * om / (T ** 2) * np.exp(hkB * om / T) / (np.exp(hkB * om / T) - 1) ** 2
    D = k ** 2 / (2 * np.pi ** 2 * v.T)
    C = hbar * om * D.T* dfeq

    return v, tau, C

def oned_vt(k,Tr):
    v0,tau0,_ = param(k,np.zeros_like(k),Tr)
    v1,tau1,_ = param(k,np.ones_like(k),Tr)
    vt0 = np.log10(v0*tau0*1e11)
    vt1 = np.log10(v1*tau1*1e11) # v*tau

    return vt0, vt1