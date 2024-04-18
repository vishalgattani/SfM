import numpy as np


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    sz = x1.shape[0]
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    #     print(P2.shape)
    X1 = np.hstack((x1, np.ones((sz, 1))))
    X2 = np.hstack((x2, np.ones((sz, 1))))

    X = np.zeros((sz, 3))

    for i in range(sz):
        skew1 = skew(X1[i, :])
        skew2 = skew(X2[i, :])
        A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
        _, _, v = np.linalg.svd(A)
        x = v[-1] / v[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0:3].T

    return X

import scipy.optimize as opt


def NonLinearTriangulation(K, x1, x2, X_init, R1, C1, R2, C2):
    """Summary

    Args:
        K (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description
        X_init (TYPE): Description
        R1 (TYPE): Description
        C1 (TYPE): Description
        R2 (TYPE): Description
        C2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    sz = x1.shape[0]
    # print(R2)
    # print(C2)
    assert x1.shape[0] == x2.shape[0] == X_init.shape[
        0], "2D-3D corresspondences have different shape "
    X = np.zeros((sz, 3))

    init = X_init.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(
        fun=minimizeFunction,
        x0=init,
        method="dogbox",
        args=[K, x1, x2, R1, C1, R2, C2])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def minimizeFunction(init, K, x1, x2, R1, C1, R2, C2):
    """Summary

    Args:
        init (TYPE): Description
        K (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description
        R1 (TYPE): Description
        C1 (TYPE): Description
        R2 (TYPE): Description
        C2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    sz = x1.shape[0]
    X = np.reshape(init, (sz, 3))

    I = np.identity(3)
    C2 = np.reshape(C2, (3, -1))

    X = np.hstack((X, np.ones((sz, 1))))

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    error1 = 0
    error2 = 0
    error = []

    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    #     print(u1.shape,x1.shape)
    assert u1.shape[0] == x1.shape[0], "shape not matched"

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)