import cv2
import numpy as np
import random
from image_utils import *
import matplotlib
import matplotlib.pyplot as plt

def DisambiguateCameraPose(Cset, Rset, Xset):
    """ Function to implement camera pose correction

    Args:
        Cset (TYPE): Set of calculated camera poses
        Rset (TYPE): Set of calculated Rotation matrices
        Xset (TYPE): 3D points

    Returns:
        TYPE: Corrected X, R_set, C_set
    """
    best = 0
    for i in range(4):

        #         Cset[i] = np.reshape(Cset[i],(-1,-1))
        N = Xset[i].shape[0]
        n = 0
        for j in range(N):
            if ((np.dot(Rset[i][2, :], (Xset[i][j, :] - Cset[i])) > 0)
                    and Xset[i][j, 2] >= 0):
                n = n + 1
        if n > best:
            C = Cset[i]
            R = Rset[i]
            X = Xset[i]
            best = n

    return X, R, C


def draw_keypoints_and_match(img1, img2):
    """This function is used for finding keypoints and dercriptors in the image and
        find best matches using brute force/FLANN based matcher."""

    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create(100)

    pts = cv2.goodFeaturesToTrack(np.mean(img1,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
    kp1,des1 = orb.compute(img1,kps)

    pts = cv2.goodFeaturesToTrack(np.mean(img2,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
    kp2,des2 = orb.compute(img2,kps)

    # kp1, des1 = orb.detectAndCompute(gray1,None)
    # kp2, des2 = orb.detectAndCompute(gray2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    g = []
    bpts1 = []
    bpts2 = []
    b = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append([m])
            g.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        else:
            b.append(m)
            bpts1.append((kp1[m.queryIdx].pt))
            bpts2.append(kp2[m.trainIdx].pt)

    gray1 = img2gray(img1)

    if len(g)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in g ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in g ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = gray1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(g), 10) )
        matchesMask = None

    # Draw keypoints
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,g,None,**draw_params)
    plt.imshow(img3, 'gray')
    plt.show()

    # Draw keypoints
    draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
    img4 = cv2.drawMatches(img1,kp1,img2,kp2,b,None,**draw_params)
    plt.imshow(img4, 'gray')
    plt.show()


    # img_with_keypoints = cv2.drawMatches(img1,kp1,img2,kp2,final_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite("images_with_matching_keypoints.png", img_with_keypoints)

    # Getting x,y coordinates of the matches
    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in g]
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in g]

    return list_kp1, list_kp2




def calculate_E_matrix(F, K1, K2):
    """Calculation of Essential matrix"""

    E = np.dot(K2.T, np.dot(F,K1))
    return E


def get_RTset(E):

    U, S, V = np.linalg.svd(E,full_matrices=True)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W,V))
    R3 = np.dot(U,np.dot(W.T,V))
    R4 = np.dot(U,np.dot(W.T,V))

    T1 = U[:,2]
    T2 = -U[:,2]
    T3 = U[:,2]
    T4 = -U[:,2]

    R = [R1,R2,R3,R4]
    T = [T1,T2,T3,T4]

    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            T[i] = -T[i]

    return R, T

def extract_camerapose(E):
    """This function extracts all the camera pose solutions from the E matrix"""

    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0,-1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1, C2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W,Vt)), np.dot(U, np.dot(W.T, Vt))
    # print("C1", C1, "\n", "C2", C2, "\n", "R1", R1, "\n", "R2", R2, "\n")

    camera_poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]



    return camera_poses


def disambiguate_camerapose(camera_poses, list_kp1):
    """This fucntion is used to find the correct camera pose based on the chirelity condition from all 4 solutions."""

    max_len = 0
    # Calculating 3D points
    for pose in camera_poses:

        front_points = []
        for point in list_kp1:
            # Chirelity check
            X = np.array([point[0], point[1], 1])
            V = X - pose[1]

            condition = np.dot(pose[0][2], V)
            # print(condition)
            if condition.any() > 0:
                front_points.append(point)

        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose =  pose

    return best_camera_pose

def getRt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3,:3]= R
    ret[:3,3:]= t
    # print(R,t)
    # print("Using my method:",ret)
    return R,t,ret


def point_triangulation(k1,k2,pt1,pt2,R1,C1,R2,C2):
    points_3d = []

    I = np.identity(3)
    C1 = C1.reshape(3,1)
    C2 = C2.reshape(3,1)

    #calculating projection matrix P = K[R|T]
    P1 = np.dot(k1,np.dot(R1,np.hstack((I,-C1))))
    P2 = np.dot(k2,np.dot(R2,np.hstack((I,-C2))))

    #homogeneous coordinates for images
    xy = np.hstack((pt1,np.ones((len(pt1),1))))
    xy_cap = np.hstack((pt2,np.ones((len(pt1),1))))


    p1,p2,p3 = P1
    p1_cap, p2_cap,p3_cap = P2

    #constructing contraints matrix
    for i in range(len(xy)):
        A = []
        x = xy[i][0]
        y = xy[i][1]
        x_cap = xy_cap[i][0]
        y_cap = xy_cap[i][1]

        A.append((y*p3) - p2)
        A.append((x*p3) - p1)

        A.append((y_cap*p3_cap)- p2_cap)
        A.append((x_cap*p3_cap) - p1_cap)

        A = np.array(A).reshape(4,4)

        _, _, v = np.linalg.svd(A)
        x_ = v[-1,:]
        x_ = x_/x_[-1]
        x_ = x_[:3]
        points_3d.append(x_)

    return points_3d

def linear_triangulation(R_Set,T_Set,pt1,pt2,k1,k2):
    R1_ = np.identity(3)
    T1_ = np.zeros((3,1))
    points_3d_set = []
    for i in range(len(R_Set)):
        points3d = point_triangulation(k1,k2,pt1,pt2,R1_,T1_,R_Set[i],T_Set[i])
        points_3d_set.append(points3d)

    return points_3d_set

def compute_cheriality(pt,r3,t):
    count_depth = 0
    for xy in pt:
        if np.dot(r3,(xy-t)) > 0 and t[2] > 0:
            count_depth +=1
    return count_depth

def extract_pose(E,pt1,pt2,k1,k2):
    #get four rotation and translation matrices
    R_set, T_set = get_RTset(E)



    #get 3D points using triangulation
    pts_3d = linear_triangulation(R_set,T_set,pt1,pt2,k1,k2)
    threshold = 0
    #Four sets are available for each possibility
    for i in range(len(R_set)):
        R = R_set[i]
        T = T_set[i]
        r3 = R[2]
        pt3d = pts_3d[i]
        #calculating which R satisfies the condition
        num_depth_positive = compute_cheriality(pt3d,r3,T)
        if num_depth_positive > threshold:
            index = i
            threshold = num_depth_positive

    R_best = R_set[index]
    T_best = T_set[index]

    return R_best,T_best,index,pt3d
