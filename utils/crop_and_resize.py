import numpy as np
from skimage.measure import block_reduce
import cv2

path="/Users/jonathanberte/devel2/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(path)

def crop_and_resize(img, target_size=64, zoom=1):
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    crop_img=img[0:64, 0:64]
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        crop_img = img[y:(y+h), x:(x+w)]
    img=crop_img
    #laplacian = cv2.Laplacian(img,cv2.CV_64F)
    #sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
    #edges = cv2.Canny(img, 50, 50)
    cv2.imshow('Video', img)
    small_side = int(np.min(img.shape) * zoom)
    reduce_factor = small_side // target_size
    crop_size = target_size * reduce_factor
    mid = np.array(img.shape) // 2
    half_crop = crop_size // 2
    center = img[mid[0]-half_crop:mid[0]+half_crop,
    	mid[1]-half_crop:mid[1]+half_crop]
    print("reduce factor")
    print(reduce_factor)
    if reduce_factor==0:
        reduce_factor=2
    img2=block_reduce(center, (reduce_factor, reduce_factor), np.mean)
    #cv2.imshow('Video', img2)
    #cv2.imwrite('gray2.png', img2)

    return block_reduce(center, (reduce_factor, reduce_factor), np.mean)
