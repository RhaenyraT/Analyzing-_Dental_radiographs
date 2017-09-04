"""
	Task 1 implementations - Building active shape model
"""

import os
import re
import fnmatch
import cPickle as pickle
import cv2
import hashlib
from utils import Timer

import task2

from landmarks import Landmarks
from shape_model import shape_model
from grey_level_model import GreyLevelModel

PYRAMID_LEVELS = 1
LANDMARK_DIR = os.path.join('.', 'Data/Landmarks/original')

def load_file(incisor_idx):
    """
    loads all the landmark files of this incisor
    """

    files = sorted(fnmatch.filter(os.listdir(LANDMARK_DIR), "*-{}.txt".format(str(incisor_idx))), key=lambda x: int(re.search('[0-9]+', x).group()))

    landmarks_incisor_idx = []

    for filename in files:
        landmarks_incisor_idx.append(Landmarks("{}/{}".format(LANDMARK_DIR, filename)))

    return landmarks_incisor_idx


def load_all_landmarks_of(incisor_idx, test_img_idx):
    """
    loads all the original and mirrored landmark files of this incisor
    """

    original_landmarks = load_file(incisor_idx)
    
    del original_landmarks[test_img_idx-1]
    
    mirrored_incisor_idx = {1:4, 2:3, 3:2, 4:1, 5:8, 6:7, 7:6, 8:5}
    mirrored_landmarks = [landmarks_points.mirror_y() for landmarks_points in load_file(mirrored_incisor_idx[incisor_idx])]

    return original_landmarks + mirrored_landmarks

def gauss_pyramid(image, levels):
    """
    Create a gaussian pyramid on a given image.
    """
    output = []
    output.append(image)
    tmp = image
    for _ in range(0, levels):
        tmp = cv2.pyrDown(tmp)
        output.append(tmp)
    return output

def get_gradient_sobel(img):
    """
    Applies the Sobel Operator to given image.
    """
    directory = "Data/Radiographs/Sobel/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fname = hashlib.md5(img).hexdigest() + ".png"
    if not os.path.isfile(directory + fname):
        
        img = cv2.GaussianBlur(img,(3,3),0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        cv2.imwrite(directory + fname, sobel)
        return sobel
    else:
        return cv2.imread(directory + fname, 0)    
    return sobel

def buildASM(incisor_list, test_img_idx, k):
    """
    Build ASM of a given incisor list. 
    Excludes the test image from model construction.
    Args:
        k: No. of pixels on either side of a model point for grey level model
    """
    asm_list = []
    train_imgs = task2.load()
    del train_imgs[test_img_idx-1]
    
    with Timer("Building Active Shape Model"):
    
        for incisor in incisor_list:
            print("..For Incisor "+str(incisor))
            
            train_lms = load_all_landmarks_of(incisor, test_img_idx)
        
            directory = "ASM_Models/test_img_%02d/" %(test_img_idx)
            filename = "incisor_%d.model" %(incisor)
    
            if not os.path.exists(directory+filename):
                asm = ASM(incisor, train_lms, train_imgs, k)
                asm_list.append(asm)
                save_asm(asm, directory, filename)
            else:
                with file(directory+filename, 'rb') as f:
                    asm_list.append( pickle.load(f) )
    
    print("") # just for elegant printing on screen   

    return asm_list

def save_asm(asm, directory, filename):
    """
    saves an ASM. The directory ASM_Models has one sub-directory for each test image.
    """
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    fn = os.path.join(directory, filename)
    with open(fn, 'wb') as f:
        pickle.dump(asm, f, pickle.HIGHEST_PROTOCOL)
    f.close()
      
class ASM(object):
    
    """
    A Class for representing Active Shape Model for a given incisor.

    """
    def __init__(self, incisor_nr, lms, imgs, k):

        self.incisor = incisor_nr
        self.k = k
        
        # shape model
        self.sm = shape_model(lms)
        
        # Gaussian Image Pyramids
        pyramids = [gauss_pyramid(image, PYRAMID_LEVELS) for image in imgs]
        lms_pyramids = [[lm.scaleposition(1.0/2**level) for level in range(0, PYRAMID_LEVELS+1)] for lm in lms]

        # Grey Level Model
        self.glms = []
        for level in range(0, PYRAMID_LEVELS+1):
            images = [task2.enhance(image, skip_amf=True) for image in zip(*pyramids)[level]]
            gimages = [get_gradient_sobel(img) for img in images]
            lms = zip(*lms_pyramids)[level]

            glms = []
            for model_point in range(0, 40):
                glm = GreyLevelModel()
                glm.build(images, gimages, lms, model_point, k)
                glms.append(glm)
            self.glms.append(glms)
        






