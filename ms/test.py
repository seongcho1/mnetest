from math import pi, cos, sin
import numpy as np
from csp import csp


if __name__ == '__main__':

    # Generate data
    N=3#00
    mu=np.array([0,0])
    sigma=np.array([6,1])
    rot1=np.identity(2)
    theta=np.dot(15, pi) / 180
    rot2=np.array([[cos(theta),- sin(theta)],[sin(theta),cos(theta)]])

    # r = np.random.rand(N,2)
    # r = np.tile(np.array([0.1,0.1]), (N,1))
    # r = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8],[0.9,1.0]])
    r = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
    r2 = r * np.tile(sigma,(N,1))

    data0 = np.matmul(np.tile(mu,(N,1)) + r2, rot1)

    print(data0)
    exit()

    a = np.tile(mu,(N,1)) + r * np.tile(sigma,(N,1))
    print(np.dot(rot1, a))
    data1=(np.dot(rot1, a).T).T
    print(data1)

    #data1=(dot(rot1,(repmat(mu,N,1) + multiply(randn(N,2),repmat(sigma,N,1))).T)).T
# # test.m:10
#     data2=(dot(rot2,(repmat(mu,N,1) + multiply(randn(N,2),repmat(sigma,N,1))).T)).T
# # test.m:11
#     d1=dot(rot1,concat([[1],[0]]))
# # test.m:12
#     d2=dot(rot2,concat([[1],[0]]))

# # test.m:13
#     # Plot the generated data and their directions
#     subplot(1,2,1)
#     scatter(data1(arange(),1),data1(arange(),2))
#     hold('on')
#     scatter(data2(arange(),1),data2(arange(),2))
#     hold('on')
#     plot(multiply(concat([0,d1(1)]),max(ravel(data1))),multiply(concat([0,d1(2)]),max(ravel(data1))),'linewidth',2)
#     hold('on')
#     plot(multiply(concat([0,d2(1)]),max(ravel(data2))),multiply(concat([0,d2(2)]),max(ravel(data2))),'linewidth',2)
#     hold('on')
#     legend('class 1','class 2','d_1','d_2')
#     hold('off')
#     grid('on')
#     axis('equal')
#     title('Before CSP filtering')
#     xlabel('Channel 1')
#     ylabel('Channel 2')
#     # CSP
#     X1=data1.T
# # test.m:27

#     X2=data2.T
# # test.m:28

#     W,l,A=csp(X1,X2,nargout=3)
# # test.m:29
#     X1_CSP=dot(W.T,X1)
# # test.m:30
#     X2_CSP=dot(W.T,X2)

# # test.m:31
#     # Plot the results
#     subplot(1,2,2)
#     scatter(X1_CSP(1,arange()),X1_CSP(2,arange()))
#     hold('on')
#     scatter(X2_CSP(1,arange()),X2_CSP(2,arange()))
#     hold('on')
#     legend('class 1','class 2')
#     hold('off')
#     axis('equal')
#     grid('on')
#     title('After CSP filtering')
#     xlabel('Channel 1')
#     ylabel('Channel 2')
