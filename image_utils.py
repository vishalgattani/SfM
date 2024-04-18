import cv2
import random
import numpy as np

def read_img(path):
    img = cv2.imread(path)
    col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return col

def img2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
