from skimage import io
from skimage import filters
from skimage import restoration
from skimage import measure

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


import cv2

def read_image(filename):
    # img = io.imread(filename)
    # img = restoration.denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    # return img
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def color_masks(img):
    # img should be in hsv format

    blue_lb = (100, 190, 100)
    blue_hb = (125, 255, 255)

    green_lb = (40, 190, 50)
    green_hb = (60, 255, 255)

    yellow_lb = (12, 200, 125)
    yellow_hb = (30, 255, 255)

    red_lb = (0, 190, 100)
    red_hb = (12, 255, 255)

    mask = cv2.inRange(hsv_img, green_lb, green_hb)
    result = cv2.bitwise_and(img, img, mask=mask)

    tagged, count = measure.label(mask, neighbors=8, return_num=True)

    plt.subplot(1, 1, 1)
    plt.imshow(tagged)
    plt.show()


if __name__ == '__main__':
    filename = "/ros/catkin_ws/src/hrc_discrim_learning/train/GRE3D3-1.0/GRE3D3-1.0/GRE3D3-stimuli/1.jpg"
    img = read_image(filename)

    io.imshow(img)
    io.show()

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # h, s, v = cv2.split(hsv_img)
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")
    #
    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker='.')
    # axis.set_xlabel('hue')
    # axis.set_ylabel('saturation')
    # axis.set_zlabel('value')
    #
    # plt.show()
    color_masks(hsv_img)
