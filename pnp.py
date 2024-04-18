""" File to implement Non Linear PnP method
"""
import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as Rscipy
import random
from tqdm import tqdm



def convertHomogeneouos(x):
    """Summary

    Args:
        x (array): 2D or 3D point

    Returns:
        TYPE: point appended with 1
    """
    m, n = x.shape
    if (n == 3 or n == 2):
        x_new = np.hstack((x, np.ones((m, 1))))
    else:
        x_new = x
    return x_new


def LinearPnP(X, x, K):
    """Summary

    Args:
        X (TYPE): 3D points
        x (TYPE): 2D points
        K (TYPE): intrinsic Matrix

    Returns:
        TYPE: C_set, R_set
    """
    N = X.shape[0]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]  #.reshape((1, 3))

        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))

        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))

    _, _, v = np.linalg.svd(A)
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    u, _, v = np.linalg.svd(R)

    R = np.matmul(u, v)
    d = np.identity(3)
    d[2][2] = np.linalg.det(np.matmul(u, v))
    R = np.dot(np.dot(u, d), v)
    C = -np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R


def proj3Dto2D(x3D, K, C, R):
    """Summary

    Args:
        x3D (TYPE): Description
        K (TYPE): Description
        C (TYPE): Description
        R (TYPE): Description

    Returns:
        TYPE: Description
    """
    C = C.reshape(-1, 1)
    x3D = x3D.reshape(-1, 1)
    # print("K", K.shape, R.shape, C.shape, x3D.shape)
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    X3D = np.vstack((x3D, 1))

    # print("P",P.shape, X3D.shape)
    u_rprj = (np.dot(P[0, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    v_rprj = (np.dot(P[1, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    X2D = np.hstack((u_rprj, v_rprj))
    return X2D


def PnPRANSAC(X, x, K):
    """Summary

    Args:
        X (TYPE): Description
        x (TYPE): Description
        K (TYPE): Description

    Returns:
        TYPE: Description
    """
    cnt = 0
    M = x.shape[0]
    threshold = 5  #6
    x_ = convertHomogeneouos(x)

    Cnew = np.zeros((3, 1))
    Rnew = np.identity(3)

    for trails in tqdm(range(500)):
        # random.randrange(0, len(corr_list))
        random_idx = random.sample(range(M), 6)
        C, R = LinearPnP(X[random_idx][:], x[random_idx][:], K)
        S = []
        for j in range(M):
            reprojection = proj3Dto2D(x_[j][:], K, C, R)
            e = np.sqrt(
                np.square((x_[j, 0]) - reprojection[0]) +
                np.square((x_[j, 1] - reprojection[1])))
            if e < threshold:
                S.append(j)
        countS = len(S)
        if (cnt < countS):
            cnt = countS
            Rnew = R
            Cnew = C

        if (countS == M):
            break
    # print("Inliers = " + str(cnt) + "/" + str(M))
    return Cnew, Rnew


def reprojError(CQ, K, X, x):
    """Function to calculate reprojection error

    Args:
        K (TYPE): intrinsic matrix
        X (TYPE): 3D points
        x (TYPE): 2D points

    Returns:
        TYPE: Reprojection error
    """
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    C = CQ[0:3]
    R = CQ[3:7]
    C = C.reshape(-1, 1)
    r_temp = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    R = r_temp.as_dcm()

    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))

    # print("P",P.shape, X3D.shape)
    u_rprj = (np.dot(P[0, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    v_rprj = (np.dot(P[1, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    e1 = x[:, 0] - u_rprj
    e2 = x[:, 1] - v_rprj
    e = e1 + e2

    return sum(e)


def NonLinearPnP(X, x, K, C0, R0):

    q_temp = Rscipy.from_dcm(R0)
    Q0 = q_temp.as_quat()
    # reprojE = reprojError(C0, K, X, x)

    CQ = [C0[0], C0[1], C0[2], Q0[0], Q0[1], Q0[2], Q0[3]]
    assert len(CQ) == 7, "length of init in nonlinearpnp not matched"
    optimized_param = opt.least_squares(
        fun=reprojError, method="dogbox", x0=CQ, args=[K, X, x])
    Cnew = optimized_param.x[0:3]
    assert len(Cnew) == 3, "Translation Nonlinearpnp error"
    R = optimized_param.x[3:7]
    r_temp = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    Rnew = r_temp.as_dcm()

    return Cnew, Rnew