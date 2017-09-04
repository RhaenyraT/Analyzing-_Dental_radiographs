"""
Methods for plotting results.
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

SCREEN_H = 720
SCREEN_W = 1280

def plot_landmarks(lms):
    """
    Visualizes the given landmark points.
    """
    if not isinstance(lms, list):
        lms = [lms]

    max_x, min_x, max_y, min_y = [], [], [], []
    for lm in lms:
        points = lm.as_matrix()
        max_x.append(points[:, 0].max())
        min_x.append(points[:, 0].min())
        max_y.append(points[:, 1].max())
        min_y.append(points[:, 1].min())
    max_x, min_x, max_y, min_y = max(max_x), min(min_x), max(max_y), min(min_y)

    img = np.zeros((int((max_y - min_y) + 20), int((max_x - min_x) + 20)))

    for lm in lms:
        points = lm.as_matrix()
        for i in range(len(points)):
            img[int(points[i, 1] - min_y) + 10, int(points[i, 0] - min_x) + 10] = 1

    cv2.imshow('Landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def plot_procrustes(mean_shape, aligned_shapes, test_img_idx, incisor, show=True, save=False):
    """
    Plots the result of the procrustes analysis.
    """
    img = np.ones((600, 300, 3), np.uint8) * 255

    mean_shape = mean_shape.scale(1000).translate([150, 300])
    points = mean_shape.as_matrix()
    
    for i in range(len(points)):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                 (0, 0, 0), 2) 
        
    # plot perpendicular line passing through center representing axes
    cv2.line(img,(0,300),(300,300),(0,0,0),2)
    cv2.line(img,(150,0),(150,600),(0,0,0),2)
    
    # Plotting scatter points of aligned shapes with respect to mean shape
    for ind, aligned_shape in enumerate(aligned_shapes):
        
        aligned_shape = aligned_shape.scale(1000).translate([150, 300])
        points = aligned_shape.as_matrix()
        
        points = points[1::2] # plotting odd points
        for i in range(len(points)):
            cv2.circle(img, (int(points[i, 0]),int(points[i, 1])), 3, (0,0,0))
    
    if show:
        show_image(img, 'Procrustes result for incisor ' + str(incisor))

    if save:
        directory = "Plots/Procrustes/test_img_"+str(test_img_idx)+"/"
        save_image(img, "incisor_"+str(incisor)+".png", directory)
    
    cv2.destroyAllWindows()
    

def plot_sm(asm, test_img_idx, incisor, show=True, save=False):
    """
    Plots the first four principal components(modes) of the ASM.
    """

    # plotting variance- shows first 4 components explain nearly 98% variance
    f = plt.figure(1)
    plt.plot(asm.variance_explained)
    plt.xlabel("cumsum of eigenvalues")
    plt.ylabel("fraction of variance explained")
    
    # plot between first two PCs to check if there's any non-linear relationship
    g = plt.figure(2)
    plt.plot(asm.pc_modes[:,0], asm.pc_modes[:,1], '+')
    plt.xlabel("b1")
    plt.ylabel("b2")

    if show:
        plt.show()
    if save:
        directory = "Plots/shape_model/test_img_"+str(test_img_idx)+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)   
        f.savefig(directory+"variance.png")
        g.savefig(directory+"Plot_of_shape_parameters.png", bbox_inches='tight')

    # plotting first 4 PCs
    plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 0], test_img_idx, title="incisor"+str(incisor)+"mode1", show=show, save=save)
    plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 1], test_img_idx, title="incisor"+str(incisor)+"mode2", show=show, save=save)
    plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 2], test_img_idx, title="incisor"+str(incisor)+"mode3", show=show, save=save)
    plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 3], test_img_idx, title="incisor"+str(incisor)+"mode4", show=show, save=save)



def plot_mode(mu, pc, test_img_idx, title="Shape Model", show=True, save=False):
    """
    Plot the mean shape +/- nstd times the principal component
    """
    from landmarks import Landmarks

    shapes = [Landmarks(mu-2*pc),
              Landmarks(mu-1*pc),
              Landmarks(mu),
              Landmarks(mu+1*pc),
              Landmarks(mu+2*pc),
              ]
    plot_shapes(shapes, test_img_idx, title, show, save)


def plot_shapes(shapes, test_img_idx, title="Shape Model", show=True, save=False):
    """
    Function to show all of the shapes which are passed to it.
    """
    cv2.namedWindow(title)

    shapes = [shape.scale_to_unit().scale(1000) for shape in shapes]
    
    x_range = 0;
    for shape in shapes:
        x_range += ( int(max(shape.points[:,0])) - int(min(shape.points[:,0])) )

    min_x = int(min([shape.points[:, 0].min() for shape in shapes]))
    max_y = int(max([shape.points[:, 1].max() for shape in shapes]))
    min_y = int(min([shape.points[:, 1].min() for shape in shapes]))

    img = np.ones((max_y-min_y+20, x_range+80, 3), np.uint8)*255
    
    prev_maxx = 0
    for shape in shapes:                        
        points = shape.points
        
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]-min_x+10 + prev_maxx), int(points[i, 1]-min_y+10)), \
                  (int(points[(i + 1) % 40, 0]-min_x+10 + prev_maxx), int(points[(i + 1) % 40, 1]-min_y+10)),
                  (0,0,0), thickness=1, lineType=cv2.LINE_AA)            
        prev_maxx += ( int(max(shape.points[:,0])) - int(min(shape.points[:,0])) )              
            
    if show:
        cv2.imshow(title, img)
        cv2.waitKey()
        
    if save:
        directory = "Plots/shape_model/test_img_"+str(test_img_idx)+"/"
        save_image(img, title+"_PC.png", directory)
        
    cv2.destroyAllWindows()


def draw_bbox(img, lms_list, show=True, save=False, return_bbox=False):
    """
    Draws a bounding box around the given landmark points
    """    
    
    img = img.copy()
    minx = []
    maxx = []
    miny = []
    maxy = []
    
    for ind, lms in enumerate(lms_list):
        x=y=w=h=0
        img1 = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
        points = lms.as_matrix()
        
        for i in range(len(points)):
            cv2.fillPoly(img1, np.int32([points]), 255);

        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(img1,0,255,0)
    
        _, contours, _ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    
        x,y,w,h = cv2.boundingRect(contours[0])
        minx.append(x)
        maxx.append(x+w)
        miny.append(y)
        maxy.append(y+h)
    
    cv2.rectangle(img,(min(minx),min(miny)),(max(maxx),max(maxy)),(0,255,0),2)
    if show:
        show_image(img, "contour")   
    
    if return_bbox:
        return [min(minx),min(miny),max(maxx),max(maxy)]

def plot_jaw_split(img, minimal_points, paths, best_path):
    """Plots a jaw split.
    Args:
        img: The dental radiograph for which the jaw split was computed.
        minimal_points ([(value, x, y)]): The low-intensity points.
        paths ([Path]): Candidate paths for the jaw split.
        best_path (Path): The jaw split
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    map(lambda x: draw_path(img, x, color=(0, 0, 255)), paths)

    draw_path(img, best_path, color=(0, 255, 0))

    for _, x, y in minimal_points:
        cv2.circle(img, (x, y), 1, 150, 10)
    show_image(img,'split')


def draw_path(radiograph, path, color=255):
    for i in range(0, len(path.edges)-1):
        cv2.line(radiograph, path.edges[i], path.edges[i+1], color, 5)
        
    
def plot_autoinit(img, jaw_split, current_window=None, search_region=None,lowest_error_bbox=None, \
                  directory="Plots/auto_init/finding_bboxes/", title="initial_estimate_bbox", \
                  wait=False, show=True, save=True):
    """
    Plots a single step in finding the automatic initial estimate.
    """
    img = img.copy()

    # draw search_region on image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if search_region:
        cv2.rectangle(img, search_region[1], search_region[0], (255, 0, 0), 4)

    # draw bbox of lowest error on image
    if lowest_error_bbox:
        cv2.rectangle(img, lowest_error_bbox[1], lowest_error_bbox[0], (0, 255, 0), 4)

    # draw current sliding window on image
    if current_window:
        cv2.rectangle(img, current_window[1], current_window[0], (0, 0, 255), 3)

    # draw jaw split on image
    draw_path(img, jaw_split, color=(0, 255, 0))

    img = fit_on_screen(img)
    if show:
        cv2.imshow(title, img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            time.sleep(0.025)

    if save:
        save_image(img, title+".png", directory)       
        
def plot_landmarks_on_image(lms_list, img, directory="Plots/", title="Landmarks", \
                            show=True, save=False, wait=True, color=(0,255,0)):
    """
    Visualizes a list of landmarks on a radiograph.
    """
    img = img.copy()

    for ind, lms in enumerate(lms_list):
        points = lms.as_matrix()
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1)%40, 0]), int(points[(i + 1)%40, 1])),
                     color, thickness=3)

    if show:
        img = fit_on_screen(img)
        cv2.imshow(title,img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            time.sleep(0.025)
    if save:
        save_image(img, title+".png", directory) 
        

def plot_profiles(sample_profile, model_profile, cost_of_fit, level, \
                  test_img_idx, save=False, show=False):
    
    titles = [i + str(level) for i in ['sample_profile','model_profile','cost_of_fit']]
    for ind, profile in enumerate([sample_profile, model_profile, cost_of_fit]):
        l = len(profile)
        x = range(-(l-1)/2 , (l+1)/2)
    
        f = plt.figure(1)
        plt.bar(x, profile, align='center', alpha=0.7)

        if show:
            plt.show()

        if save:
            directory = "Plots/model_fit/test_img_"+str(test_img_idx)+"/"
            if not os.path.exists(directory):
                os.makedirs(directory)    

            f.savefig(directory+titles[ind]+".png")
        plt.close(f)

def plot_results(x, x_labels, y, test_img_idx, show=False, save=False):

    f = plt.figure(1)
    plt.bar(x, y, align='center', alpha=0.7)
    plt.xticks(x, x_labels)
    plt.ylabel('Dice Score')

    if show:
        plt.show()

    if save:
        directory = "Plots/model_fit/test_img_"+str(test_img_idx)+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)   
        f.savefig(directory+"results.png")
    plt.close(f)
        
def show_image(img, title="Image"):
    """
    Plots the given image.

    """
    img = fit_on_screen(img)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_image(img, title="plot", directory="Plot/"):
    """
    saves an image in a given directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)    
    cv2.imwrite(directory+title, img)
    
def fit_on_screen(image):
    """
    Rescales the given image such to fit on screen.
    """
    
    scale = min(float(SCREEN_W) / image.shape[1], float(SCREEN_H) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

