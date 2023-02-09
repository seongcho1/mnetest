import numpy as np
from scipy import linalg as LA

def csp(X1=None,X2=None,*args,**kwargs):

    # Error detection
    # if nargin < 2:
    #     raise ValueError('Not enough parameters.')

    # if length(size(X1)) != 2 or length(size(X2)) != 2:
    #     raise ValueError('The size of trial signals must be [C x T]')

    # Compute the covariance matrix of each class
    S1=np.cov(np.transpose(X1))
# csp.m:9

    S2=np.cov(np.transpose(X2))
# csp.m:10

    # Solve the eigenvalue problem S1·W = l·S2·W
    W,L=LA.eig(S1, S1 + S2)
# csp.m:12

    lambda_=LA.eigvals(L)
# csp.m:13

    A=np.transpose(np.linalg.inv(W))
# csp.m:14

    # Further notes:
    #   - CSP filtered signal is computed as: X_csp = W'*X;
    return W, lambda_, A

if __name__ == '__main__':
    pass
