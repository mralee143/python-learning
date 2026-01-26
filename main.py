# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread('glass.jpg', cv2.IMREAD_GRAYSCALE)

# # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.axis('off')          # optional: removes axes
# plt.savefig('output.jpg', dpi=300, bbox_inches='tight')
# plt.close('all')

#Geometric Transformations of Images

# import cv2 as cv

# img = cv.imread('glass.jpg')
# assert img is not None, "File could not be read, check path or os.path.exists()"

# # Method 1: Resize using scale factors
# res1 = cv.resize(
#     img,
#     None,
#     fx=2,
#     fy=2,
#     interpolation=cv.INTER_CUBIC
# )

# # Method 2: Resize using explicit width & height
# height, width = img.shape[:2]
# res2 = cv.resize(
#     img,
#     (2 * width, 2 * height),
#     interpolation=cv.INTER_CUBIC
# )

# # Show results
# cv.imshow("Original", img)
# cv.imshow("Resized (fx, fy)", res1)
# cv.imshow("Resized (width, height)", res2)

# cv.waitKey(0)
# cv.destroyAllWindows()

#Image Thresholding

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
 
# img = cv.imread('glass.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# img = cv.medianBlur(img,5)
 
# ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
 
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
 
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


#Smoothing Images

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
 
# img = cv.imread('glass.jpg')
# assert img is not None, "file could not be read, check with os.path.exists()"
 
# kernel = np.ones((5,5),np.float32)/25
# dst = cv.filter2D(img,-1,kernel)
 
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

#Morphological Transformations
# import cv2 as cv
# import numpy as np

# img = cv.imread('glass.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)

# Image Gradients
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# import os

# # Read image
# img = cv.imread('glass.jpg', cv.IMREAD_GRAYSCALE)

# # Check if image loaded
# if img is None:
#     raise FileNotFoundError("Image not found. Check image path or filename.")

# # Apply filters
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# # Plot results
# plt.figure(figsize=(10, 8))

# plt.subplot(2, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# Canny Edge Detection
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
 
# img = cv.imread('glass.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# edges = cv.Canny(img,100,200)
 
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
# plt.show()

# Image Pyramids
# import cv2 as cv

# img = cv.imread('glass.jpg')

# assert img is not None, "file could not be read, check file path"

# lower_reso = cv.pyrDown(img)

# cv.imshow("Original", img)
# cv.imshow("Lower Resolution", lower_reso)

# cv.waitKey(0)
# cv.destroyAllWindows()
# Contours in OpenCV
# import numpy as np
# import cv2 as cv
 
# im = cv.imread('glass.jpg')
# assert im is not None, "file could not be read, check with os.path.exists()"
# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Contour Features
import numpy as np
import cv2 as cv

img = cv.imread('glass.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv.moments(cnt)
print( M )