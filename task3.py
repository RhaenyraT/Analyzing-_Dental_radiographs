"""
	Task 3 implementations - initialisation and fitting the model
"""

import numpy as np
import Plots
import task2 
import task1
import auto_init
import manual_init
from landmarks import Landmarks
from landmarks import load_landmarks
import math
from grey_level_model import Profile
from procrustes_analysis import get_align_params
from utils import Timer, medfilt
import cv2

from task1 import PYRAMID_LEVELS
MAX_ITER = 50

save_plots = False
show_plots = False
test_idx = 0

def fit_model(asm_list, incisor_list, test_img_idx, m, auto_estimate=True, save=False, show=False):
            
    global save_plots
    global test_idx
    global show_plots
    save_plots = save
    test_idx = test_img_idx
    show_plots = show
    
    test_img = task2.load([test_img_idx])[0]
    X_init_list = []
    if auto_estimate:
        X_init_list =auto_init.get_estimate(asm_list, incisor_list, test_img_idx)
    else:
        lms_list = []
        for asm in asm_list:
            lms_list.append(asm.sm.mean_shape)
        
        X_init_list=manual_init.get_estimate(lms_list, incisor_list, test_img)
            
        Plots.plot_landmarks_on_image(X_init_list, test_img, title="manual_init",\
                                  show=False, save=False, wait=True, color=(0,255,0))

    with Timer("Fitting Model in Multi Resolution Framework"):
        
        final_fit = []
        for ind,X in enumerate(X_init_list):
            print("..For incisor %d" %(asm_list[ind].incisor))
            pyramid = task1.gauss_pyramid(test_img, PYRAMID_LEVELS)
            X = X.scaleposition(1.0 / 2**(PYRAMID_LEVELS+1))
            level = PYRAMID_LEVELS
            
            for img in reversed(pyramid): 
                #print("..Level %d" %(level))
                X = X.scaleposition(2)
                X = fit_one_level(X, img, asm_list[ind], level, m, MAX_ITER)   
                level -= 1
                
            final_fit.append(X)
    
    print("") # just for elegant printing on screen   
    
    if save_plots or show_plots:
        directory = "Plots/model_fit/test_img_"+str(test_idx)+"/"
        Plots.plot_landmarks_on_image(final_fit, test_img, directory=directory, title="Final_fit",\
                          show=show_plots, save=save_plots, wait=True, color=(0,255,0))
        
    return final_fit
    
    
def fit_one_level(X, test_img, asm, level, m, max_iter):
    """
    Fit the model for one level of the image pyramid.
    Based on: 'Protocol 2' in 'An Introduction to Active Shape Models'
    """
    
    glms = asm.glms[level]
    
    img = task2.enhance(test_img, skip_amf=True)
    gimg = task1.get_gradient_sobel(img)
    
    b = np.zeros(asm.sm.pc_modes.shape[1])
    X_prev = Landmarks(np.zeros_like(X.points))
    
    nb_iter = 1
    n_close = 0
    best_fit = np.inf
    best_Y = None

    total_s = 1
    total_theta = 0

    while (n_close < 21 and nb_iter <= max_iter): 

        Y, n_close, fit_quality = get_best_nearby_match(X, asm, img, gimg, glms, m, level)
        if fit_quality < best_fit:
            best_fit = fit_quality
            best_Y = Y
        Plots.plot_landmarks_on_image([X, Y], test_img, wait=False, title="Fitting incisor %02d" % (asm.incisor,))

        if nb_iter == max_iter:
            Y = best_Y

        b, t, s, theta = update_model_params(X, asm, Y, test_img)

        # constraint the pose and shape parameters    
        b = np.clip(b, -3, 3)
        # t = np.clip(t, -5, 5)
        s = np.clip(s, 0.95, 1.05)
        if total_s * s > 1.20 or total_s * s < 0.8:
            s = 1
        total_s *= s

        theta = np.clip(theta, -math.pi/8, math.pi/8)
        if total_theta + theta > math.pi/4 or total_theta + theta < - math.pi/4:
            theta = 0
        total_theta += theta

        X_prev = X
        X = Landmarks(X.as_vector() + np.dot(asm.sm.pc_modes, b)).T(t, s, theta)
        Plots.plot_landmarks_on_image([X_prev, X], test_img, wait=False,title="Fitting incisor %02d" % (asm.incisor,))

        nb_iter += 1

#    print("....No. of Iterations to converge - %d/%d" %(nb_iter, max_iter))
#    print("....No. of pixels found within central 25%% of the profile - atleast %d/%d" %(n_close, 40))
    return X
    
def get_best_nearby_match(X, asm, img, gimg, glms, m, level):
    """
    Examines a region of the given image around each point X_i to find
    """
    Y = []
    n_close = 0
    profiles = []
    best_pixels = []
    fit_qualities = []

    for ind in range(len(X.points)):

        profile = Profile(img, gimg, X, ind, m)
        profiles.append(profile)

        lowest_costs, best_pixel = np.inf, None
        costs_of_fit = []
        
        for i in range(asm.k, asm.k+2*(m-asm.k)+1):
            subprofile = profile.samples[i-asm.k:i+asm.k+1]
            dist = glms[ind].quality_of_fit(subprofile)
            costs_of_fit.append(dist)
            if dist < lowest_costs:
                lowest_costs = dist
                best_pixel = i

        best_pixels.append(best_pixel)
        fit_qualities.append(lowest_costs)
        best_point = [int(c) for c in profile.points[best_pixel, :]]
            
        if(best_pixel > 3*m/4 and best_pixel < 5*m/4):
            n_close += 1


    # Plot sample profile for 10th model point for instance
    if save_plots or show_plots:
        global test_idx
        if(n_close > 19): 
            Plots.plot_profiles(profile.samples, glms[9].mean_profile, costs_of_fit, level, \
                                test_idx, save=save_plots, show=show_plots)
            
    # applying a median filter to get smooth boundary 
    best_pixels.extend(best_pixels)
    best_pixels = np.rint(medfilt(np.asarray(best_pixels), 5)).astype(int)
    for best, profile in zip(best_pixels, profiles):
        best_point = [int(c) for c in profile.points[best, :]]
        Y.append(best_point)
            
    fit_quality = np.mean(fit_qualities)
    return Landmarks(np.array(Y)), n_close, fit_quality


def update_model_params(X, asm, Y, testimg):
    """
    Updates translation, scale and rotation and the shape
    parameters to best fit the model instance X to a new found image points Y.
    Based on: 'Protocol 1' in 'An Introduction to Active Shape Models'
    """

    b = np.zeros(asm.sm.pc_modes.shape[1])
    b_prev = np.ones(asm.sm.pc_modes.shape[1])
    i = 0
    
    while (np.mean(np.abs(b-b_prev)) >= 1e-14):
        i += 1

        x = Landmarks(X.as_vector() + np.dot(asm.sm.pc_modes, b))

        t, s, theta = get_align_params(x, Y)
        
        y = Y.invT(t, s, theta)

        y1 = Landmarks(y.as_vector() / np.dot(y.as_vector(), X.as_vector().T))

        b_prev = b
        b = np.dot(asm.sm.pc_modes.T, (y1.as_vector()-X.as_vector()))

    return b, t, s, theta


def evaluate_results(test_img_idx, incisor_list,  final_fit_list):
    """
    Uses the Dice Coefficient to evaulate the similarity between the final landmarks 
    given by the model and the ground truth landmarks of the test image
    """
    
    test_img = task2.load([test_img_idx])[0]
    test_lms_list = load_landmarks(test_img_idx, incisor_list) 
    
    dice_scores = []
    height, width, _ = test_img.shape
    
    with Timer("Evaluating Results"):
        print("..Dice similarity score")
        
        for ind, incisor in enumerate(incisor_list):
    
            image1 = np.zeros((height, width), np.uint8)
            image2 = np.zeros((height, width), np.uint8)
    
            X1 = test_lms_list[ind].as_matrix() # X1 - ground truth
    
            cv2.fillPoly(image1, np.int32([X1]), 255)
            
            X2 = final_fit_list[ind].as_matrix() # X2 - best fit

            cv2.fillPoly(image2, np.int32([X2]), 255)
            
            dice = np.sum(image1[image2 == 255])*2.0 / (np.sum(image1) + np.sum(image2))
            print("....for incisor %02d - %.2f" %(incisor, dice))
            dice_scores.append(dice)

    if save_plots or show_plots:
        Plots.plot_results(np.arange(len(dice_scores)), incisor_list, dice_scores, \
                           test_img_idx, show=show_plots, save=save_plots)
    
    
    