# https://matlab.mathworks.com/
# seongcho@student.42seoul.kr
# first captial usual$

from math import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
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
    r3 = np.tile(mu,(N,1)) + r2
    r3t = np.transpose(r3)
    data9 = np.matmul(r3, rot2)
    data0 = np.transpose(np.matmul(rot2, r3t))

    #data1 = (rot1*(repmat(mu,N,1)+ randn(N,2).*repmat(sigma,N,1))')';
    data1 = np.transpose(np.matmul(rot1, r3t))
    #data2 = (rot2*(repmat(mu,N,1)+ randn(N,2).*repmat(sigma,N,1))')';
    data2 = np.transpose(np.matmul(rot2, r3t))
    d1 = np.matmul(rot1, [[1],[0]])
    d2 = np.matmul(rot2, [[1],[0]])

    print(data2)
    print(d1)
    print(d2)
    print('-'*42)


    # plt.scatter(*zip(*data1), cmap='blue')
    # plt.scatter(*zip(*data2), cmap='red')
    # plt.show()


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


    # CSP
    X1=np.transpose(data1)
    X2=np.transpose(data2)

    W,l,A=csp(X1,X2)

    X1_CSP=np.dot(np.transpose(W),X1)
    X2_CSP=np.dot(np.transpose(W),X2)

    print(X2_CSP)


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
