import numpy as np
from scipy import linalg as LA

def csp(X1=None,X2=None,*args,**kwargs):

    # Error detection
    # if nargin < 2:
    #     raise ValueError('Not enough parameters.')

    # if length(size(X1)) != 2 or length(size(X2)) != 2:
    #     raise ValueError('The size of trial signals must be [C x T]')

    # 1. Compute the covariance matrix of each class
    S1=np.cov(X1)
    S2=np.cov(X2)

    print(f"X1.shape={X1.shape}, S1.shape={S1.shape}")
    print(f"X2.shape={X2.shape}, S2.shape={S2.shape}")

    # 2. Solve the eigenvalue problem S1·W = l·S2·W
    #W,L=LA.eig(S1, S1+S2)
    l,W=LA.eigh(S1, S1+S2)
    l = np.round(l, 5)
    A=(np.linalg.inv(W)).T

    print(f"eigenvalue={l}, eigenvector.shape={W.shape}")

    # Further notes:
    #   - CSP filtered signal is computed as: X_csp = W'*X;
    return W, l, A

