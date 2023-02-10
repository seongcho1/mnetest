import numpy as np
from scipy import linalg as LA

def csp(X1=None,X2=None,*args,**kwargs):

    # Error detection
    # if nargin < 2:
    #     raise ValueError('Not enough parameters.')

    # if length(size(X1)) != 2 or length(size(X2)) != 2:
    #     raise ValueError('The size of trial signals must be [C x T]')

    # Compute the covariance matrix of each class
    S1=np.cov(X1)
    S2=np.cov(X2)

    # Solve the eigenvalue problem S1·W = l·S2·W
    #W,L=LA.eig(S1, S1+S2)
    l,W=LA.eigh(S1, S1+S2)
    l = np.round(l, 5)
    A=(np.linalg.inv(W)).T

    # Further notes:
    #   - CSP filtered signal is computed as: X_csp = W'*X;
    return W, l, A

