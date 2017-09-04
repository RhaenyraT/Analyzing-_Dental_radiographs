"""
	Task 2 implementations - Preprocessing radiographs
"""

import sys
import os
import cv2
import numpy as np
from scipy.ndimage import morphology
import Plots
from utils import Timer

DATA_DIR = "Data/Radiographs/"

def load(indices=range(1, 15), preprocessed = False):
    """
    Loads original or preprocessed radiographs of given indices.
    """

    files = ["%02d.tif" % i if i < 15 else "extra/%02d.tif" % i for i in indices]
    if preprocessed:
        files = [DATA_DIR +"/Preprocessed/"+ f for f in files]
    else:
        files = [DATA_DIR + f for f in files]
    images = [cv2.imread(f) for f in files]

    for index, img in zip(indices, images):
        if img is None:
            raise IOError("%s%02d.tif does not exist" % (DATA_DIR, index,))

    return images         
    
def preprocess_radiographs(skip_amf=True):
    
    images = load()

    preprocessed_directory = DATA_DIR+"/Preprocessed/"
    if not os.path.exists(preprocessed_directory):
        os.makedirs(preprocessed_directory)    
    print("")
    with Timer("Preprocessing radiographs"):
        
        for ind, img in enumerate(images):
            print("..Processing ["+str(ind+1)+"/14]")
            
            fname = "%s%02d.tif" % (preprocessed_directory, ind+1,)
            if not os.path.isfile(fname):
                enhanced_img = enhance(img, skip_amf)
                cv2.imwrite("%s%02d.tif" % (preprocessed_directory, ind+1,), enhanced_img)
            else:
                print("..Radiographs are already preprocessed!")
                break
        
        print("..Preprocessed radiographs are in %s" % preprocessed_directory)
    
    print("") # just for elegant printing on screen   

def enhance(img, skip_amf=True):
        
    img = img.copy()
    
    if not skip_amf:
        img = adaptive_median(img)
        #Plots.plot_image(img, title="Adaptive median filtered")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       

    img = bilateral_filter(img)
    #Plots.plot_image(img, title="Bilateral filtered")

    img = mathematical_morphology(img)
    #Plots.plot_image(img, title="Top-hat and bottom-hat combined")

    img = clahe(img)
    #Plots.plot_image(img, title="CLAHE")
    
    return img

def adaptive_median(image_array, window=3, threshold=5):
    """
    Applies an adaptive median filter to the image
    Source: https://github.com/sarnold/adaptive-median/blob/master/adaptive_median.py
    """
    image_array = image_array.copy()
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    
    def med(target_array, array_length):
        """Computes the median of a sublist.
        """
        sorted_array = np.sort(target_array)
        median = sorted_array[array_length/2]
        return median
    
    # set filter window and image dimensions
    W = 2*window + 1
    ylength, xlength = image_array.shape
    vlength = W*W
    
    # create 2-D image array and initialize window
    filter_window = np.array(np.zeros((W, W)))
    target_vector = np.array(np.zeros(vlength))
    pixel_count = 0
    
    try:
        # loop over image with specified window W
        for y in range(window, ylength-(window+1)):
            update_progress(y/float(ylength))
            for x in range(window, xlength-(window+1)):
                # populate window, sort, find median
                filter_window = image_array[y-window:y+window+1, x-window:x+window+1]
                target_vector = np.reshape(filter_window, ((vlength),))
                # internal sort
                median = med(target_vector, vlength)
                # check for threshold
                if not threshold > 0:
                    image_array[y, x] = median
                    pixel_count += 1
                else:
                    scale = np.zeros(vlength)
                    for n in range(vlength):
                        scale[n] = abs(target_vector[n] - median)
                    scale = np.sort(scale)
                    Sk = 1.4826 * (scale[vlength/2])
                    if abs(image_array[y, x] - median) > (threshold * Sk):
                        image_array[y, x] = median
                        pixel_count += 1
        update_progress(1)
    
    except TypeError:
        print("Error in adaptive median filter function")
        sys.exit(2)
    
    print(pixel_count, "pixel(s) filtered out of", xlength*ylength)
    return image_array


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 175, 175)


def mathematical_morphology(img):
    """
    Performs mathematical morphology - top-hat and bottom-hat
    """
    img_top = top_hat_transform(img)

    img_bottom = bottom_hat_transform(img)

    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)
    return img

def top_hat_transform(img):
    return morphology.white_tophat(img, size=400)


def bottom_hat_transform(img):
    return morphology.black_tophat(img, size=80)

def clahe(img):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)

def update_progress(progress):
    """Displays or updates a console progress bar

    Args:
        progress: Accepts a float between 0 and 1.
                    Any int will be converted to a float.
                    A value under 0 represents a 'halt'.
                    A value at 1 or bigger represents 100%

    Source: http://stackoverflow.com/a/15860757

    """
    bar_length = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0.0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0.0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1.0
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    text = "\r....Percent: [{0}] {1}% {2}".format("#"*block + "-"*(bar_length-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()    