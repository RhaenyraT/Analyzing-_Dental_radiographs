"""
A program to test different implementations of the project separately
"""
import cv2
import Plots
import task1
from procrustes_analysis import GPA
from shape_model import shape_model

import task2
import auto_init
import manual_init
import task3

def test_GPA(incisor_list, test_img_idx):
    
    for incisor in incisor_list:
        train_lms = task1.load_all_landmarks_of(incisor, test_img_idx)
        mean_shape, aligned_shapes = GPA(train_lms)
        Plots.plot_procrustes(mean_shape, aligned_shapes, test_img_idx, incisor, show=False, save=True)
    
    
def test_shape_model(incisor_list):
    
    for incisor in incisor_list:
        train_lms = task1.load_all_landmarks_of(incisor, test_img_idx)
        sm = shape_model(train_lms)
        Plots.plot_sm(sm, test_img_idx, incisor, show=False, save=True)
    
    
def test_preprocessing_radiographs(img_idx, skip_amf=True):
    
    img = cv2.imread( "%s%02d.tif" %("Data/Radiographs/", img_idx) )
    directory="Plots/Preprocessed/"
    img = img.copy()
    Plots.save_image(img, "Original.png", directory)

    if not skip_amf:
        img = task2.adaptive_median(img)
        #Plots.plot_image(img, title="Adaptive median filtered")
        Plots.save_image(img, "amf.png", directory)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       

    img = task2.bilateral_filter(img)
    #Plots.plot_image(img, title="Bilateral filtered")
    Plots.save_image(img, "bilateral.png", directory)
        
    img = task2.mathematical_morphology(img)
    #Plots.plot_image(img, title="Top-hat and bottom-hat combined")
    Plots.save_image(img, "math_morph.png", directory)
    
    img = task2.clahe(img)
    #Plots.plot_image(img, title="CLAHE")
    Plots.save_image(img, "clahe.png", directory)


def test_auto_initial_estimate(incisor_list, test_img_idx, k):
    
    asm_list = task1.buildASM(incisor_list, test_img_idx, k)
    X_init_list = auto_init.get_estimate(asm_list, incisor_list, test_img_idx, show_bbox_dist=True, \
                           show_app_models=False, show_finding_bbox=True, \
                           show_autoinit_bbox=False, show_autoinit_lms=False, save=False)
    test_img = task2.load([test_img_idx])[0]
    directory= "Plots/auto_init/test_img_%02d/" %(test_img_idx)
    Plots.plot_landmarks_on_image(X_init_list, test_img, directory=directory, title="auto_init",\
                                  show=False, save=True, wait=True, color=(0,255,0))
    
    
def test_manual_init(incisor_list, test_img_idx):
    
    asm_list = task1.buildASM(incisor_list, test_img_idx, k) 
    lms_list = []
    for asm in asm_list:
        lms_list.append(asm.sm.mean_shape)
        
    test_img = task2.load([test_img_idx])[0]
    X_init_list=manual_init.get_estimate(lms_list, test_img)
        
    Plots.plot_landmarks_on_image(X_init_list, test_img, title="manual_init",\
                              show=True, save=False, wait=True, color=(0,255,0))
   
def test_fit_model(incisor_list, test_img_idx, k, m, auto_estimate=True):
    
    asm_list = task1.buildASM(incisor_list, test_img_idx, k) 

    final_fit_list = task3.fit_model(asm_list, incisor_list, test_img_idx, m, \
                                     auto_estimate=auto_estimate, save=True, show=False) 

    task3.evaluate_results(test_img_idx, incisor_list, final_fit_list)


if __name__ == '__main__':
    
    test_img_idx = 2
    incisor_list = range(1,9)
    #incisor_list = [2]
    k = 10
    m = 15
    
    #test_GPA(incisor_list, test_img_idx)
    #test_shape_model(incisor_list)
    #test_preprocessing_radiographs(test_img_idx, skip_amf=False)
    #test_auto_initial_estimate(incisor_list, test_img_idx, k)
    
    #test_img_idx = 6
    #test_auto_initial_estimate(incisor_list, test_img_idx, k)
    
    #test_manual_init(incisor_list, test_img_idx)
    test_fit_model(incisor_list, test_img_idx, k, m, auto_estimate=False)
    
    