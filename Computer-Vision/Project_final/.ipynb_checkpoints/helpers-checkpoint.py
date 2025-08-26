import cv2
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import random

def get_cmap(n, name='gist_rainbow'):
    return plt.cm.get_cmap(name, n)

def plot_cams(P, ax, color):
    c = null_space(P).flatten()
    c /= c[-1]
    v = P[2, :3]*0.2
    ax.scatter(c[0], c[1], c[2], color=color, s=10, label="Camera Center")
    ax.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color=color, length=1, linewidth=1.5, label="Principal Axis")
        

def plot_3d_points_and_cameras_with_principal_axes(Xmodel, P_list):
    cmap = get_cmap(len(P_list))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(Xmodel)):
        ax.scatter(Xmodel[i][0], Xmodel[i][1], Xmodel[i][2], color=cmap(i), s=0.1, label="3D Model Points")
        plot_cams(P_list[i], ax, cmap(i))
        
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')
    plt.show()

def plot_3d_points(X):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0], X[1], X[2], color='green', s=1, label="3D Model Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')
    plt.show()



def triangulate_3D_point_DLT(P1, P2, x1, x2):
    A = np.zeros((4, 4))
    
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]  
    
    return X / X[-1]

def triangulate_all_points(P1, P2, x1, x2):
    X = np.zeros((4, x1.shape[1]))  

    for i in range(x1.shape[1]):
        X[:, i] = triangulate_3D_point_DLT(P1, P2, x1[:, i], x2[:, i])

    return X



def pflat(x):
    return x / x[-1]



def estimate_T_DLT(R, xs, Xs):
    n = xs.shape[1]
    A, B = zip(*[
        (
            np.array([[1, 0, -x1], [0, 1, -x2]]),
            np.array([[X3 * x1 - X1], [X3 * x2 - X2]])
        )
        for x1, x2, (X1, X2, X3) in zip(xs[0], xs[1], (R @ Xs).T)
    ])
    A, B = np.vstack(A), np.vstack(B)
    return np.linalg.lstsq(A, B, rcond=None)[0].flatten()


def estimate_T_robust(xs, Xs, R, threshold, max_iterations=80000):
    best_inliers = None
    best_P = None
    max_inliers = 0

    for _ in range(max_iterations):
        indices = np.random.choice(xs.shape[1], 2, replace=False)
        xs_sample = xs[:, indices]
        Xs_sample = Xs[:, indices]

        T_candidate = estimate_T_DLT(R, xs_sample, Xs_sample)
        P_candidate = np.hstack((R, T_candidate.reshape(3,1)))

        Xs_h = np.vstack((Xs, np.ones(Xs.shape[1])))
        xs_projected = P_candidate @ Xs_h
        xs_projected = pflat(xs_projected)
        
        delta = (xs_projected - xs)**2
        errors = delta[0] + delta[1]
        inliers = errors < threshold**2

        num_inliers = np.sum(inliers)
       
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_P = P_candidate
    
    return best_P, best_inliers


def reconstruct_initial_3D(i1,i2,images,epipolar_threshold, K, P1, Rs):
    x1, x2, descriptor = image_points(images[i1], images[i2],True)
    x1_hom  = np.vstack((x1, np.ones(x1.shape[1])))
    x2_hom  = np.vstack((x2, np.ones(x2.shape[1])))
    x1n = normalize_points(x1_hom, K)
    x2n = normalize_points(x2_hom, K)
    
    E_best, inliers = estimate_E_robust(x1n, x2n, K, epipolar_threshold)

    x1_inliers = x1n[:, inliers]
    x2_inliers = x2n[:, inliers]
    

    P2, X0 = best_P(E_best, K, P1, x1_inliers, x2_inliers)
   
    R_world = Rs[i1].T
    
    X0_world = R_world @ X0[:3]
       
    return X0_world, descriptor[inliers]


def best_P(E,K,P1, x1_inliers, x2_inliers):
    P2s = extract_P_from_E(E)
    max_points = -1
    best_solution = None
    index = 0
  
    for i,P2 in enumerate(P2s):
        X = triangulate_all_points(P1, P2, x1_inliers, x2_inliers)
        count = count_points_in_front(P1, X) + count_points_in_front(P2, X)
        if count > max_points:
            index = i
            max_points = count
            best_solution = P2
            best_X = X

    return best_solution, best_X

def extract_P_from_E(E):
    U, _, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    P = [
        np.hstack((U @ W @ Vt, U[:, 2:3])),
        np.hstack((U @ W @ Vt, -U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, -U[:, 2:3]))
    ]
    return P
    
def triangulate_3D_point_DLT(P1, P2, x1, x2):
    A = np.zeros((4, 4))
    
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]  
    
    return X / X[-1]

def triangulate_all_points(P1, P2, x1, x2):
    X = np.zeros((4, x1.shape[1]))  

    for i in range(x1.shape[1]):
        X[:, i] = triangulate_3D_point_DLT(P1, P2, x1[:, i], x2[:, i])

    return X

def count_points_in_front(P, X):
    X_proj = P @ X 
    return np.sum(X_proj[2, :] > 0)

def image_points(img1, img2, return_descriptor=False):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    filtered_matches = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            filtered_matches.append(m)
    
    x1 = np.array([kp1[m.queryIdx].pt for m in filtered_matches]).T
    x2 = np.array([kp2[m.trainIdx].pt for m in filtered_matches]).T

    if return_descriptor:
        des1 = np.array([des1[m.queryIdx] for m in filtered_matches])
        return x1, x2, des1
    
    return x1, x2

def normalize_points(points, K):
    K_inv = np.linalg.inv(K)
    normalized_points = K_inv @ points
    return normalized_points

def enforce_essential(E_approx):
    U, S, Vt = np.linalg.svd(E_approx)
    S = np.array([1,1,0])
    E = U @ np.diag(S) @ Vt
    return E

def estimate_E_DLT(x1s, x2s,K, norm=True):
    if norm == True:
        x1n = normalize_points(x1s,K)
        x2n = normalize_points(x2s,K)
    else:
        x1n = x1s
        x2n = x2s
    
    M = []
    for i in range(x1n.shape[1]):
        x1, y1, _ = x1n[:, i]
        x2, y2, _ = x2n[:, i]
        M.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    M = np.array(M)

    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1]
    E_approx = v.reshape(3, 3)

    min_singular_value = S[-1]
    residual = np.linalg.norm(M @ v)
    
    E = enforce_essential(E_approx)
    
    return E

def compute_point_to_line_distances(l, x):
    l1 = l[0,:]
    l2 = l[1,:]
    l3 = l[2,:]
    x1 = x[0,:]
    x2 = x[1,:]
    
    numerator = np.abs(l1*x1 + l2*x2 + l3)
    denominator = np.sqrt(l1**2 + l2**2)
    distances = numerator / denominator
    
    return distances

def estimate_E_robust(x1, x2, K, eps, max_iterations=80000):

    best_inliers = None
    best_E = None
    max_inliers = 0

    for _ in range(max_iterations):
        indices = np.random.choice(x1.shape[1], 8, replace=False)
        x1_sample = x1[:, indices]
        x2_sample = x2[:, indices]

        E_candidate = estimate_E_DLT(x1_sample, x2_sample, K, norm=False)
        
        l1 = E_candidate.T @ x2
        errors1 = compute_point_to_line_distances(l1,x1)
        
        l2 = E_candidate @ x1
        errors2 = compute_point_to_line_distances(l2,x2)
        
        errors = (errors1**2 + errors2**2) / 2
        
        inliers = errors < (eps**2)
        num_inliers = np.sum(inliers)
       
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_E = E_candidate

    return best_E, best_inliers



def get_dataset_info(dataset):
    if dataset == 1:
        img_names = ["data/1/kronan1.JPG", "data/1/kronan2.JPG"]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 45.0  # from the EXIF data
        init_pair = [1, 2]
        pixel_threshold = 1.0
    elif dataset == 2:
        # Corner of a courtyard
        img_names = [
            "data/2/DSC_0025.JPG", "data/2/DSC_0026.JPG", "data/2/DSC_0027.JPG", "data/2/DSC_0028.JPG", "data/2/DSC_0029.JPG",
            "data/2/DSC_0030.JPG", "data/2/DSC_0031.JPG", "data/2/DSC_0032.JPG", "data/2/DSC_0033.JPG"
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [1, 9]
        pixel_threshold = 1.0
    elif dataset == 3:
        # Smaller gate of a cathedral
        img_names = [
            "data/3/DSC_0001.JPG", "data/3/DSC_0002.JPG", "data/3/DSC_0003.JPG", "data/3/DSC_0004.JPG", "data/3/DSC_0005.JPG",
            "data/3/DSC_0006.JPG", "data/3/DSC_0007.JPG", "data/3/DSC_0008.JPG", "data/3/DSC_0009.JPG", "data/3/DSC_0010.JPG",
            "data/3/DSC_0011.JPG", "data/3/DSC_0012.JPG"
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [5, 8]
        pixel_threshold = 1.0
    elif dataset == 4:
        # Fountain
        img_names = [
            "data/4/DSC_0480.JPG", "data/4/DSC_0481.JPG", "data/4/DSC_0482.JPG", "data/4/DSC_0483.JPG", "data/4/DSC_0484.JPG",
            "data/4/DSC_0485.JPG", "data/4/DSC_0486.JPG", "data/4/DSC_0487.JPG", "data/4/DSC_0488.JPG", "data/4/DSC_0489.JPG",
            "data/4/DSC_0490.JPG", "data/4/DSC_0491.JPG", "data/4/DSC_0492.JPG", "data/4/DSC_0493.JPG"
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [5, 10]
        pixel_threshold = 1.0
    elif dataset == 5:
        # Golden statue
        img_names = [
            "data/5/DSC_0336.JPG", "data/5/DSC_0337.JPG", "data/5/DSC_0338.JPG", "data/5/DSC_0339.JPG", "data/5/DSC_0340.JPG",
            "data/5/DSC_0341.JPG", "data/5/DSC_0342.JPG", "data/5/DSC_0343.JPG", "data/5/DSC_0344.JPG", "data/5/DSC_0345.JPG"
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 45.0  # from the EXIF data
        init_pair = [3, 7]
        pixel_threshold = 1.0
    elif dataset == 6:
        # Detail of the Landhaus in Graz
        img_names = [
            "data/6/DSCN2115.JPG", "data/6/DSCN2116.JPG", "data/6/DSCN2117.JPG", "data/6/DSCN2118.JPG", "data/6/DSCN2119.JPG",
            "data/6/DSCN2120.JPG", "data/6/DSCN2121.JPG", "data/6/DSCN2122.JPG"
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [2, 4]
        pixel_threshold = 1.0
    elif dataset == 7:
        # Building in Heidelberg
        img_names = [
            "data/7/DSCN7409.JPG", "data/7/DSCN7410.JPG", "data/7/DSCN7411.JPG", "data/7/DSCN7412.JPG", "data/7/DSCN7413.JPG",
            "data/7/DSCN7414.JPG", "data/7/DSCN7415.JPG"
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [1, 7]
        pixel_threshold = 1.0
    elif dataset == 8:
        # Relief
        img_names = [
            "data/8/DSCN5540.JPG", "data/8/DSCN5541.JPG", "data/8/DSCN5542.JPG", "data/8/DSCN5543.JPG", "data/8/DSCN5544.JPG",
            "data/8/DSCN5545.JPG", "data/8/DSCN5546.JPG", "data/8/DSCN5547.JPG", "data/8/DSCN5548.JPG", "data/8/DSCN5549.JPG",
            "data/8/DSCN5550.JPG"
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [4, 7]
        pixel_threshold = 1.0
    elif dataset == 9:
        # Triceratops model on a poster
        img_names = [
            "data/9/DSCN5184.JPG", "data/9/DSCN5185.JPG", "data/9/DSCN5186.JPG", "data/9/DSCN5187.JPG", "data/9/DSCN5188.JPG",
            "data/9/DSCN5189.JPG", "data/9/DSCN5191.JPG", "data/9/DSCN5192.JPG", "data/9/DSCN5193.JPG"
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [4, 6]
        pixel_threshold = 1.0
    else:
        raise ValueError("Unknown dataset number")

    focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
    K = np.array([
        [focal_length, 0, im_width / 2],
        [0, focal_length, im_height / 2],
        [0, 0, 1]
    ])

    return K, img_names, init_pair, pixel_threshold



