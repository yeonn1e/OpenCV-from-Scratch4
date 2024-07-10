import numpy as np
import cv2
import time
import random
np.seterr(all='ignore')
np.random.seed(4)

temple1 = cv2.imread('./CV_Assignment_3_Data/temple1.png')
temple2 = cv2.imread('./CV_Assignment_3_Data/temple2.png')

house1 = cv2.imread('./CV_Assignment_3_Data/house1.jpg')
house2 = cv2.imread('./CV_Assignment_3_Data/house2.jpg')

library1 = cv2.imread('./CV_Assignment_3_Data/library1.jpg')
library2 = cv2.imread('./CV_Assignment_3_Data/library2.jpg')

M_temple = np.loadtxt('./CV_Assignment_3_Data/temple_matches.txt')
M_house = np.loadtxt('./CV_Assignment_3_Data/house_matches.txt')
M_library = np.loadtxt('./CV_Assignment_3_Data/library_matches.txt')

# 1-1. Fundamental matrix computation
def compute_avg_reproj_error(_M, _F):
    # compute_avg_reproj_error.py
    N = _M.shape[0]

    X = np.c_[ _M[:,0:2] , np.ones( (N,1) ) ].transpose()
    L = np.matmul( _F , X ).transpose()
    norms = np.sqrt( L[:,0]**2 + L[:,1]**2 )
    L = np.divide( L , np.kron( np.ones( (3,1) ) , norms ).transpose() )
    L = ( np.multiply( L , np.c_[ _M[:,2:4] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error = (np.fabs(L)).sum()

    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = ( np.multiply( L , np.c_[ _M[:,0:2] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error += (np.fabs(L)).sum()

    return error/(N*2)

def compute_F_raw(M):
    # linear system 구성: 포인트 쌍에 대해 방정식 구성
    # N x 9 형태의 행렬 A 구성할 수 있음
    A = np.zeros((M.shape[0], 9))
    for i in range(M.shape[0]):
        xp, yp, x, y = M[i, :]
        A[i, :] = np.array([x*xp, x*yp, x,
                            y*xp, y*yp, y,
                            xp, yp, 1])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1, :]
    F = F.reshape((3, 3))
    return F

def compute_F_norm(M):
    # compute_F_raw에서 정규화, rank 조정, denomalization의 과정 추가
    # - nomalization: center to (0,0) + scaling to -1 ~ +1
    c1 = M[:, :2]
    c2 = M[:, 2:]
    
    m1 = np.mean(c1, axis=0)
    m2 = np.mean(c2, axis=0)
    std1 = np.std(c1 - m1)
    std2 = np.std(c2 - m2)
    

    mat1 = np.array([[1/std1, 0, -m1[0]/std1],
                   [0, 1/std1, -m1[1]/std1],  
                   [0, 0, 1]])
    mat2 = np.array([[1/std2, 0, -m2[0]/std2],
                   [0, 1/std2, -m2[1]/std2],  
                   [0, 0, 1]])
    
    
    # Homogeneous 만들기
    h1 = np.column_stack((c1, np.ones(len(c1))))
    h2 = np.column_stack((c2, np.ones(len(c2))))

    # 업데이트
    new1 = np.matmul(mat1, h1.T)
    new2 = np.matmul(mat2, h2.T)
    c11 = new1.T[:, :2]
    c22 = new2.T[:, :2]
    M_new = np.hstack((c11, c22))

    F = compute_F_raw(M_new)

    #U,S,Vt = np.linalg.svd(F)
    #S[2] = 0
    #F = np.matmul(np.matmul(U, np.diag(S)), Vt)

    # denormalization
    F = mat2.T.dot(F).dot(mat1)

    return F

def compute_F_mine(M):
    """
    ** RANSAC + remove outliers within 5 seconds **
    1. 무작위로 선택된 8 points 설정
    2. 정규화 F 계산
    3. inliers 설정
    4. 가장 작은 average reprojection error 가지는 F 찾기
    """
    start = time.time()
    smallest_err = np.inf
    identified_pts = set()
    inliers = []
    threshold = 0.15

    def exist(pts): # 존재 여부 확인 return boolean
        return pts in identified_pts

    def add(pts):
        identified_pts.add(pts)

    def choose():
        # 1. 무작위로 선택된 8 points 설정
        sample_pts = random.sample(range(len(M)), 8)
        sorted_pts = tuple(sorted(sample_pts))
        return sorted_pts

    while (time.time() - start) < 4.99:  # within 5 sec
        idx = choose()
        if exist(idx):
            continue

        add(idx)

        # 2. 정규화 F 계산
        F = compute_F_norm(M[list(idx), :])

        # 3. inliers 설정
        error = np.zeros((M.shape[0],))
        for i in range(M.shape[0]):
            ones1, ones2 = np.append(M[i, :2], 1), np.append(M[i, 2:], 1)
            err1, err2 = abs((F @ ones1).dot(ones2)), abs((F.T @ ones2).dot(ones1)) 
            error[i] = (err1 + err2)
        inliers = [i for i in range(M.shape[0]) if error[i] <= threshold]
        M_new = np.hstack((M[inliers, :2], M[inliers, 2:]))
        F_new = compute_F_norm(M_new)

        #4. 가장 작은 average reprojection error 가지는 F 찾기
        updated_err = compute_avg_reproj_error(M, F_new)
        if updated_err < smallest_err: # 에러 최솟값으로 업데이트 (RANSAC)
            smallest_err = updated_err
            result = F_new

    if result is None:
        result = compute_F_norm(M)
    
    return result


# temple
F_temple = compute_F_mine(M_temple)
F_t = F_temple.copy() # 계산한 F값을 그대로 다시 사용하기 위함
print("Average Reprojection Errors (temple1.png and temple2.png)")
print("   Raw =", compute_avg_reproj_error(M_temple, compute_F_raw(M_temple)))
print("   Norm =", compute_avg_reproj_error(M_temple, compute_F_norm(M_temple)))
print("   Mine =", compute_avg_reproj_error(M_temple, F_temple))

# house
F_house = compute_F_mine(M_house)
F_h = F_house.copy()
print("\nAverage Reprojection Errors (house1.jpg and house2.jpg)")
print("   Raw =", compute_avg_reproj_error(M_house, compute_F_raw(M_house)))
print("   Norm =", compute_avg_reproj_error(M_house, compute_F_norm(M_house)))
print("   Mine =", compute_avg_reproj_error(M_house, F_house))

# library
F_library = compute_F_mine(M_library)
F_l = F_library.copy()
print("\nAverage Reprojection Errors (library1.jpg and library2.jpg)")
print("   Raw =", compute_avg_reproj_error(M_library, compute_F_raw(M_library)))
print("   Norm =", compute_avg_reproj_error(M_library, compute_F_norm(M_library)))
print("   Mine =", compute_avg_reproj_error(M_library, F_library))


# 1-2.Visualization of epipolar lines
M_temple = np.loadtxt('CV_Assignment_3_Data/temple_matches.txt')
M_house = np.loadtxt('CV_Assignment_3_Data/house_matches.txt')
M_library = np.loadtxt('CV_Assignment_3_Data/library_matches.txt')

def draw_epipolar(M, F, sample1, sample2):
    # 한 이미지의 점에 대해 다른 이미지에서 epipolar line을 구함
    colors = [(0,0,255), (0,255,0), (255,0,0)] # RGB
    img1, img2 = sample1.copy(), sample2.copy()  

    while True:
        IMG1, IMG2 = img1.copy(), img2.copy()
        M_random = M[np.random.choice(len(M), 3), :]  # 랜덤으로 점 선택

        for idx, pts in enumerate(M_random):
            # 점들에 동그라미 그리기
            xp, yp, x, y = pts
            X = np.array([xp, yp, 1]).reshape((3, 1))
            Xp = np.array([x, y, 1]).reshape((3, 1))
            cv2.circle(IMG1, (int(xp), int(yp)), 4, colors[idx], 2)
            cv2.circle(IMG2, (int(x), int(y)), 4, colors[idx], 2)

            # epipolar line 정의
            lp = F.dot(X) # X.T * l = 0
            l = F.T.dot(Xp) # Xp.T * lp = 0

            def draw_epiline(img, line, color):
                a, b, c = line.flatten()
                x, x_ = map(int, [0, img.shape[1]])
                y, y_ = map(int, [-c/b, -(c+a*img.shape[1])/b])
                cv2.line(img, (x, y), (x_, y_), color, 2)
                        
            draw_epiline(IMG1, l, colors[idx])
            draw_epiline(IMG2, lp, colors[idx])

        image = np.hstack((IMG1, IMG2))
        cv2.imshow('visualization of epipolar lines', image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

draw_epipolar(M_temple, F_t, temple1, temple2)
draw_epipolar(M_house, F_h, house1, house2)
draw_epipolar(M_library, F_l, library1, library2)