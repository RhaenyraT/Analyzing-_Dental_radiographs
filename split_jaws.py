
"""
Detects the split between upper and lower jaw, creating a path between
low-intensity points based on intensity histograms of the columns of the image.

Based on: https://github.com/probberechts/IncisorSegmentation/blob/master/Code/split_jaws.py 
"""

import math
import cv2
import numpy as np
import scipy.signal
import scipy.fftpack
import Plots
import task2

def get_split(radiograph, interval=50, show=False):
    """Computes a path that indicates the split between the upper and lower jaw.

    Based on histograms of the intensities in the columns of the radiograph, it
    detects the darkest points. A path between these points in the center region
    of the image is considered as the jaw split.

    Args:
        radiograph: The dental radiograph for which the split is computed.
        interval (int): The width of the rows for which histograms are computed.
        show (bool): Whether to visualize the result.

    Returns:
        Path: The estimated jaw split.

    """
    radiograph = radiograph.copy()
    # Transform the image to grayscale format
    img = cv2.cvtColor(radiograph, cv2.COLOR_BGR2GRAY)
    # Top-hat transform image to enhance brighter structures
    img = task2.top_hat_transform(img)

    # Apply a Gaussian filter in the horizontal direction over the inverse
    # of the preprocessed image.
    height, width = img.shape
    mask = 255-img
    filt = gaussian_filter(450, width)
    if width % 2 == 0:
        filt = filt[:-1]
    mask = np.multiply(mask, filt)

    # Create intensity histograms for columns of the image.
    minimal_points = []
    for x in range(interval, width, interval):
        ## generating histogram
        hist = []
        for y in range(int(height*0.4), int(height*0.7), 1):
            hist.append((np.sum(mask[y][x-interval:x+interval+1]), x, y))

        ## smooth the histogram using a Fourier transformation
        fft = scipy.fftpack.rfft([intensity for (intensity, _, _) in hist])
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)

        ## find maxima in the histogram and sort them
        indices = scipy.signal.argrelmax(smoothed)[0]
        minimal_points_width = []
        for idx in indices:
            minimal_points_width.append(hist[idx])
        minimal_points_width.sort(reverse=True)

        ## keep the best 3 local maxima which lie atleast 200 apart from another point
        count = 0
        to_keep = []
        for min_point in minimal_points_width:
            _, _, d = min_point
            if all(abs(b-d) > 150 for _, _, b in to_keep) and count < 4:
                count += 1
                to_keep.append(min_point)
        minimal_points.extend(to_keep)

    # Find pairs of points such that the summed intensities of the pixels
    # along a straight line between both points is minimal
    edges = []

    for _, x, y in minimal_points:
        min_intensity = float('inf')
        min_coords = (-1, -1)
        for _, u, v in minimal_points:
            intensity = _edge_intensity(mask, (x, y), (u, v))
            #print(x,u,intensity,min_intensity,v,y,height)
            if x < u and intensity < min_intensity and abs(v-y) < 0.1*height:
                min_intensity = intensity
                min_coords = (u, v)
        if min_coords != (-1, -1):
            edges.append([(x, y), min_coords])

    # Try to form paths from the found edges
    paths = []
    for edge in edges:
        new_path = True
        # Check if edge can be added to an existing path
        for path in paths:
            if path.edges[-1] == edge[0]:
                new_path = False
                path.extend(edge)
        if new_path:
            paths.append(Path([edge[0], edge[1]]))

    mask2 = mask * (255/mask.max())
    mask2 = mask2.astype('uint8')

    # Trim the outer edges of paths
    map(lambda p: p.trim(mask2), paths)
    # Remove too short paths
    paths = remove_short_paths(paths, width, 0.3)
    # Select the best path
    best_path = sorted([(p.intensity(img) / (p.length()), p) for p in paths])[0][1]

    # Show the result
    if show:
        Plots.plot_jaw_split(mask2, minimal_points, paths, best_path)

    # Return the best candidate
    return best_path


class Path(object):
    """A jaw split is represented by a path.

    Attributes:
        edges ([(int,int)]): A list of points along the path.

    """
    def __init__(self, edges):
        self.edges = edges

    def get_part(self, min_bound, max_bound):
        """Get a part of the path between two horizontal bounds.

        Args:
            min_bound (int): The left bound.
            max_bound (int): The right bound.

        Returns:
            The list of points on the path between the two given bounds.

        """
        edges = []
        for edge in self.edges:
            if edge[0] > min_bound and edge[0] < max_bound:
                edges.append(edge)

        return edges

    def extend(self, edge):
        """Add a new point to the right end of the path.

        Args:
            edge ((int, int)): The point to add to the path.

        """
        self.edges.append(edge[1])

    def intensity(self, radiograph):
        """Return the summed intensities of the pixels along this path.

        Args:
            radiograph: The image on which the intensities are measured.

        Returns:
            The summed intensities of the pixels along this path.

        """
        intensity = 0
        for i in range(0, len(self.edges)-1):
            intensity += _edge_intensity(radiograph, self.edges[i], self.edges[i+1])
        return intensity

    def trim(self, radiograph):
        """Trim the outer edges of the path based on the average intensity of
        the edges.

        Args:
            radiograph: The image on which the intensities are measured.

        """
        # average intensity along the path
        mean_intensity = self.intensity(radiograph) / self.length()
        # trim left outer edges
        while len(self.edges) > 2:
            if mean_intensity > _edge_intensity(radiograph, self.edges[0], self.edges[1]) / \
                    math.hypot(self.edges[1][0]-self.edges[0][0], self.edges[1][1]-self.edges[0][1]):
                del self.edges[0]
            else:
                break
        # trim right outer edges
        while len(self.edges) > 2:
            if mean_intensity > _edge_intensity(radiograph, self.edges[-1], self.edges[-2]) / \
                    math.hypot(self.edges[-1][0]-self.edges[-2][0], self.edges[-1][1]-self.edges[-2][1]):
                del self.edges[-1]
            else:
                break

    def length(self):
        """Get the length of this path.

        Returns:
            The sum of the lenghts of all edges

        """
        return np.sum(np.sqrt(np.sum(np.power(np.diff(self.edges, axis=0), 2), axis=1)))


def remove_short_paths(paths, width, ratio):
    """Remove all paths smaller than width*ratio.
    """
    return filter(lambda p: p.length() >= width*ratio, paths)


def _edge_intensity(radiograph, p1, p2):
    """Get the summed intensities of all pixels along an edge.

    Args:
        radiograph: The image on which the intensities are measured.
        p1 ([int, int]): The first point of the edge.
        p2 ([int, int]): The last point of the edge.

    Returns:
        The summed intensities of all pixels along an edge.

    """
    intensities = createLineIterator(p1, p2, radiograph)
    #print(intensities)
    return sum(intensities)


def gaussian_filter(sigma, filter_length=None):
    """Given a sigma, return a 1-D Gaussian filter.

    Args:
        sigma: float, defining the width of the filter
        filter_length: optional, the length of the filter, has to be odd

    Returns:
        A 1-D numpy array of odd length, containing the symmetric, discrete
        approximation of a Gaussian with sigma Summation of the array-values
        must be equal to one.

    """
    def gaussian_function(sigma, u):
        return 1/(math.sqrt(2*math.pi)*sigma)*math.e**-(u**2/(2*sigma**2))

    if filter_length is None:
        #determine the length of the filter
        filter_length = math.ceil(sigma*5)
        #make the length odd
        filter_length = 2*(int(filter_length)/2) + 1

    #make sure sigma is a float
    sigma = float(sigma)

    #create the filter
    result = np.asarray([gaussian_function(sigma, u) for u in range(-(filter_length//2), filter_length//2 + 1, 1)])
    result = result / result.sum()

    #return the filter
    return result

def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     

    Based on: https://stackoverflow.com/a/32857432/1075896
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1 = np.array([P1[0], P1[1]])
    P2 = np.array([P2[0], P2[1]])
    
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)
 
    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
         itbuffer[:,1] = P1Y
         if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
         else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope =  dX.astype(np.float32)/dY.astype(np.float32) #dX/dY
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope =  dY.astype(np.float32)/dX.astype(np.float32) #dY/dX
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer[:,2]

if __name__ == '__main__':
    # test it on all radiographs
    #imgs = task2.load()[0]
    img = cv2.imread('Data/Radiographs/06.tif')
    #for i in range(0, 1):
    path = get_split(img, 50, False)
    Plots.draw_path(img, path, color=(0, 255, 0))
    Plots.show_image(img)
#    for i in range(0, len(path.edges)-1):
#        print(path.edges[i], path.edges[i+1])