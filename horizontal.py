import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation


def skew_correction(image, delta=1, maxlimit=90):
    def return_score(arr, angle):
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-maxlimit, maxlimit + delta, delta)
    for angle in angles:
        histogram, score = return_score(threshold, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


# read from path -- edit
image = cv2.imread('/home/sylwiabadur/Desktop/OBRAZKI/hm.jpg')
angle, rotated = skew_correction(image)
print("Skew angle: " + str(angle))
plt.subplot(121), plt.imshow(image), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(rotated), plt.title('Skew correction')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('/home/sylwiabadur/Desktop/OBRAZKI/new.jpg',
            rotated)  # write to path -- edit
