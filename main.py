import cv2
import matplotlib.pyplot as plt

img = cv2.imread('glass.jpg', cv2.IMREAD_GRAYSCALE)

# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')          # optional: removes axes
plt.savefig('output.jpg', dpi=300, bbox_inches='tight')
plt.close('all')
