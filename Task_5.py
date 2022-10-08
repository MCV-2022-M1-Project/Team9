import cv2 
import numpy as np
import matplotlib.pyplot as plt 

def remove_background(self):
    image = cv2.imread(self)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h=image_hsv[:,:,0]
    s=image_hsv[:,:,1]
    v=image_hsv[:,:,2]

    hist_h, bin_edges = np.histogram(h, bins='auto')
    hist_s, bin_edges = np.histogram(s, bins='auto')
    hist_v, bin_edges = np.histogram(v, bins='auto')

    boundaries = [
    ([110, np.min(s), np.min(v)], [160, np.max(s), np.max(v)])
    ]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)

    plt.subplot(1,4,1)
    plt.imshow(mask,cmap='gray')


    print (np.max(h))
    print (np.min(h))
    print (np.max(s))
    print (np.min(s))
    print (np.max(v))
    print (np.min(v))

    plt.subplot(1,4,2)
    plt.plot(hist_h)
    plt.subplot(1,4,3)
    plt.plot(hist_s)
    plt.subplot(1,4,4)
    plt.plot(hist_v)
    plt.show()


def remove_background_2(self):
    image = cv2.imread(self)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    boundaries = [
	([7, 5, 150], [70, 70, 200])
    ]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        mask_inv=255-mask

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        return mask_inv


def remove_background_3(self):
    image = cv2.imread(self)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    ret, mask = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

    output = cv2.bitwise_and(image, image, mask=mask)

    return mask