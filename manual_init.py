
"""
Methods for finding an initial estimate of the model manually by allowing the user
to drag the mean shape of an incisor to the correct position on the image.
"""

import cv2
import numpy as np
from landmarks import Landmarks
from utils import Timer

tooth = []
tmpTooth = []
dragging = False
start_point = (0, 0)
tmp = np.array([])

def get_estimate(lms_list, incisor_list, img):
    """
    Allows the user to drag the mean shape to correct position after which 
    key 'f' has to be pressed to 'freeze' the initial estimate
    """
    oimgh = img.shape[0]
    img, scale = resize(img, 1200, 800)
    imgh = img.shape[0]
    canvasimg = np.array(img)

    out = []
    with Timer("Finding Initial Estimate manually"):
        for ind, lms in enumerate(lms_list):
            
            global tooth
            global dragging
            global start_point
            global tmpTooth
            global tmp
        
            if(ind > 0):
                canvasimg = np.array(tmp)
                img = np.array(tmp)  
                
            # transform model points to image coord
            points = lms.as_matrix()
            min_x = abs(points[:, 0].min())
            min_y = abs(points[:, 1].min())
            points = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points]
            tooth = points
            pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
            
            cv2.polylines(img, [pimg], True, (0, 255, 0))
            centroid = np.mean(pimg, axis=0)
            cv2.putText(img, "Incisor %d" %(incisor_list[ind]), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # show gui
            cv2.imshow('choose', img)
            cv2.setMouseCallback('choose', __mouse, canvasimg)
            
            while True:
                key = cv2.waitKey(1) & 0xFF 
                if key == ord("f"):
                    out.append(Landmarks(np.array([[point[0]*oimgh, point[1]*oimgh] for point in tooth])))
                    
                    tooth = []
                    tmpTooth = []
                    dragging = False
                    start_point = (0, 0)
                    
                    break
    
    print("") # just for elegant printing on screen   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    return out


def __mouse(ev, x, y, flags, img):
    """
    This method handles the mouse-dragging.
    """
    global tooth
    global dragging
    global start_point

    if ev == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
    elif ev == cv2.EVENT_LBUTTONUP:
        tooth = tmpTooth
        dragging = False
    elif ev == cv2.EVENT_MOUSEMOVE:
        if dragging and tooth != []:
            __move(x, y, img)


def __move(x, y, img):
    """
    Redraws the incisor on the radiograph while dragging.
    """
    global tmpTooth
    global tmp
    imgh = img.shape[0]
    tmp = np.array(img)
    dx = (x-start_point[0])/float(imgh)
    dy = (y-start_point[1])/float(imgh)

    points = [(p[0]+dx, p[1]+dy) for p in tooth]
    tmpTooth = points

    pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
    cv2.polylines(tmp, [pimg], True, (0, 255, 0))
    cv2.imshow('choose', tmp)

def resize(image, width, height):
    
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale