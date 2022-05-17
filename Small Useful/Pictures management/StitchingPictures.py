import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.
Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.
Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)
    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])


'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).
Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers
Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)
    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print('mask=', M.shape,
        #     'src', src_pts.shape,
        #     'pts1', [len(pts1),len(pts1[0])])

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
   
# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, name, showfig=False):
    
    # Write your codes here
    #im1 = cv2.imread("image1.jpg", 0)
    #im2 = cv2.imread("image2.jpg", 0)
    (x, y) = img1.shape
    
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    # get similar points 
    (M, pts1, pts2, mask1) = getTransform(img2, img1,'homography')
   ### uncomment following if affine
    # (M, pts1, pts2, mask1) = getTransform(img2, img1,'affine')
    # M = np.c_[M, np.ones(3)]
    # matrix = np.ones((3,3))
    # print(matrix)
    # matrix[:-1,:] = M
    # M =matrix 
    (x, y) = img1.shape
       
    # for example: transform img2 to img1's plane
    # first, make some room around img1
    #out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    # then transform im1 with the 3x3 transformation matrix
    out1 = cv2.warpPerspective(img2, M, (img1.shape[1],img1.shape[0]))
    # out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
    if showfig:
        plt.imshow(img1)
        plt.show()
        plt.imshow(out1)
        plt.show()
    output = np.zeros(img1.shape)
    mask = np.ones(img1.shape)

    for i in range(x):
     for j in range(y):
        if img1[i][j]==0 and out1[i][j]==0:
            output[i][j]=0
        elif img1[i][j]==0:
            output[i][j] = out1[i][j]
        elif out1[i][j]==0:
                output[i][j] = (img1[i][j])
        else:
            output[i][j]= (int(int(img1[i][j]) + int(out1[i][j]))/2) 

    cv2.imwrite(name + '.png',output)
    return output

def Laplacian_blending(img1,img2,mask,levels=4):
    
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    # print(GM.shape)
    gp1 = [G1,]
    gp2 = [G2,]
    gpM = [GM,]
    for i in range(levels):
            # print(gp1, len(gp1[0]))
            # print(i)
            G1 = cv2.pyrDown(G1)
            G2 = cv2.pyrDown(G2)
            GM = cv2.pyrDown(GM)
            gp1.append(np.float32(G1))
            gp2.append(np.float32(G2))
            gpM.append(np.float32(GM))
            # print(len(gp1))

            # generate Laplacian Pyramids for A,B and masks
    lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2  = [gp2[levels-1]]
    gpMr = [gpM[levels-1]]
    for i in range(levels-1,0,-1):
    # Laplacian: subtarct upscaled version of lower level from current level
    # to get the high frequencies
     L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
     L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
     lp1.append(L1)
     lp2.append(L2)
     gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    # print('l1', lp1[0].shape,
    #         'l2', lp2[0].shape,
    #         'gm', gpMr[0].shape,
    #         )
    for l1,l2,gm in zip(lp1,lp2,gpMr):
  
     ls = l1 * gm + l2 * (1.0 - gm)
     LS.append(ls)
     
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,levels):
     ls_ = cv2.pyrUp(ls_)
     # print(ls_.dtype, LS[i].dtype)
     ls_ = cv2.add(ls_, np.float32(LS[i]))
    return ls_

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

i=1
image = files[0]
# image = Perspective_warping(image  , files[1], 'output' + str(i))

for file in files[1:]:
    print(i)
    # print(image.dtype)
    name = 'homographyResults'
    file = crop_img(file, 0.96)
    Perspective_warping(image  , file, name + str(i))
    image = cv2.imread(name + str(i) + '.png', 0)
    # print(image.shape)
    
    i += 1
