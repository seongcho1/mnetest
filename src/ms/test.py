# https://matlab.mathworks.com/
# seongcho@student.42seoul.kr
# first captial usual$

from math import pi, cos, sin
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from csp import csp


if __name__ == '__main__':

    # Generate data
    N=500
    mu=np.array([0,0])
    sigma=np.array([6,1])
    rot1=np.identity(2)
    theta=np.dot(15, pi) / 180
    rot2=np.array([[cos(theta),- sin(theta)],[sin(theta),cos(theta)]])

    #r1 = np.random.randn(N,2)
    #r2 = np.random.randn(N,2)
    # r = np.tile(np.array([0.1,0.1]), (N,1))
    # r = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8],[0.9,1.0]])
    # r = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
    #r2 = r1 * np.tile(sigma,(N,1))
    #r3 = np.tile(mu,(N,1)) + r2

    # rd_1 = np.tile(mu,(N,1)) + np.random.randn(N,2) * np.tile(sigma,(N,1))
    # rd_2 = np.tile(mu,(N,1)) + np.random.randn(N,2) * np.tile(sigma,(N,1))
    rd_1 = mu + np.random.randn(N,2) * sigma
    rd_2 = mu + np.random.randn(N,2) * sigma
    #rd = np.tile(mu,(N,1)) + np.random.rand(N,2) * np.tile(sigma,(N,1))

    #data9 = np.dot(r3, rot2)
    #data0 = (np.dot(rot2, r3.T)).T

    #data1 = (rot1*(repmat(mu,N,1)+ randn(N,2).*repmat(sigma,N,1))')';
    data1 = (np.dot(rot1, rd_1.T)).T
    #data1 = (np.dot(rot1, (np.tile(mu,(N,1)) + np.random.randn(N,2) * np.tile(sigma,(N,1))).T)).T
    #data2 = (rot2*(repmat(mu,N,1)+ randn(N,2).*repmat(sigma,N,1))')';
    data2 = (np.dot(rot2, rd_2.T)).T
    #data2 = (np.dot(rot2, (np.tile(mu,(N,1)) + np.random.randn(N,2) * np.tile(sigma,(N,1))).T)).T

    d1 = np.dot(rot1, [[1],[0]])
    d2 = np.dot(rot2, [[1],[0]])

    # print(data2)
    print(d1)
    print(d2)

    #print(np.array([0, *d1[0]]) * 3)
    #print(np.array([0, *d1[0]]) * np.max(data1))
    print('-'*42)
    # print(*zip(*data2))
    # print('-'*42)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    #plt.scatter(*zip(*data1), cmap='blue', label='class 1')
    x1, y1 = data1.T

    plt.scatter(x1, y1, c='blue', alpha=0.3, label='class 1')
    #plt.scatter(*zip(*data2), cmap='red', label='class 2')
    x2, y2 = data2.T
    plt.scatter(x2, y2, c='orange', alpha=0.3, label='class 2')

    plt.plot(np.array([0, *d1[0]])* np.max(data1), np.array([0, *d1[1]]) *np.max(data1), linewidth=3, label='d1')
    plt.plot(np.array([0, *d2[0]]) * np.max(data2), np.array([0, *d2[1]]) *np.max(data2), linewidth=3, label='d2')

    #plt.plot(multiply(concat([0,d1(1)]),max(ravel(data1))),multiply(concat([0,d1(2)]),max(ravel(data1))),'linewidth',2)
    #plt.plot(multiply(concat([0,d2(1)]),max(ravel(data2))),multiply(concat([0,d2(2)]),max(ravel(data2))),'linewidth',2)
    plt.legend()  #(loc='upper right')#,'d_1','d_2')
    plt.grid('on')
    plt.axis('equal')
    plt.title('Before CSP filtering')
    plt.xlabel('Channel 1')
    plt.ylabel('Channel 2')
    #plt.show()

    # CSP
    X1 = data1.T
    X2 = data2.T

    # 1. Compute the covariance matrix
    # 2. Solve the eigenvalue problem
    W,l,A=csp(X1,X2)

    # 3. get CSP
    X1_CSP=np.dot(W.T,X1)
    X2_CSP=np.dot(W.T,X2)

    # print(X2_CSP)
    # print('-'*42)
    # print(*zip(*X2_CSP.T))

    # Plot the results
    plt.subplot(1, 2, 2)
    #plt.scatter(*zip(*X1_CSP.T), cmap='blue', label='class 1')
    #plt.scatter(X1_CSP[0], X1_CSP[1], cmap='blue', label='class 1')
    x1_c, y1_c = X1_CSP
    plt.scatter(x1_c, y1_c, c='blue', alpha=0.3, label='class 1')
    #plt.scatter(*zip(*X2_CSP.T), cmap='red', label='class 2')
    #plt.scatter(X2_CSP[0], X2_CSP[1], cmap='red', label='class 2')
    x2_c, y2_c = X2_CSP
    plt.scatter(x2_c, y2_c, c='orange', alpha=0.3, label='class 2')
    plt.legend()  #(loc='upper right')
    plt.grid('on')
    plt.axis('equal')
    plt.title('After CSP filtering')
    plt.xlabel('Channel 1')
    plt.ylabel('Channel 2')
    plt.show()

