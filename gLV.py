#!/usr/bin/env python3

import numpy as np

# x = steady state (Nx1 np.array)
# mu = growth rates (Nx1 np.array)
# M  = interaction values (NxN np.array)
def integrand(x, t, mu, M):
    """ Return N-dimensional gLV equations """
    dxdt = ( np.dot(np.diag(mu), x)
             + np.dot(np.diag(x), np.dot(M, x)) )
    return dxdt

def jacobian(x, t, mu, M):
    """ Return jacobian of N-dimensional gLV equation at steady state x """
    N = len(x)
    jac = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i is j:
                val = mu[i] + np.dot(M, x)[i] + M[i,i]*x[i]
                jac[i, j] = val
            else:
                val = x[i]*M[i,j]
                jac[i, j] = val
    return jac

def get_stability(x, mu, M):
    """ Evaluate stability of steady state x """
    jac = jacobian(x, 0, mu, M)
    eig_vals, eig_vecs = np.linalg.eig(jac)

    if all(eig_vals < 0):
        print('stable!')
    else:
        print('unstable!')


x = np.array([.01, .01])
mu = np.array([3, 5])
M = np.array([[-1, 3], [2, -1]])
output = integrand(x, 0, mu, M)

x = np.array([0, 0])
output = jacobian(x, 0, mu, M)
get_stability(x, mu, M)

