
"""
	Class for an (active) shape model

"""
import numpy as np
from procrustes_analysis import GPA
        
class shape_model(object):
    
    def __init__(self, landmarks):

        # Generalized Procrustes analysis
        mu, Xnew = GPA(landmarks)
        #Plots.plot_procrustes(mean_shape, aligned_shapes, incisor_nr=incisor, show=False, save=True)
    
        # covariance calculation
        XnewVec = landmarks_as_vectors(Xnew)
        S = np.cov(XnewVec, rowvar=0)
    
        self.k = len(mu.points)      # Number of points
        self.mean_shape = mu
        self.covariance = S
        self.aligned_shapes = Xnew
    
        # PCA on shapes
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(-eigvals)   # Ensure descending sort
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        self.scores = np.dot(XnewVec, eigvecs)
        self.mean_scores = np.dot(mu.as_vector(), eigvecs)
        self.variance_explained = np.cumsum(eigvals/np.sum(eigvals))
    
        # Build modes for up to 98% variance
        def extract_useful_pcs(arr):
            for index, item in enumerate(arr):
                if item:
                    return index, item
                
        npcs, _ = extract_useful_pcs(self.variance_explained > 0.99)
        npcs += 1
    
        M = []
        for i in range(0, npcs-1):
            M.append(np.sqrt(eigvals[i]) * eigvecs[:, i])
        self.pc_modes = np.array(M).squeeze().T


def landmarks_as_vectors(landmarks):
    """Converts a list of Landmarks object to vector format.
    Args:
        landmarks: A list of Landmarks objects
    Returns:
        A numpy N x 2*p array where p is the number of landmarks and N the
        number of examples. x coordinates are before y coordinates in each row.
    """
    mat = []
    for lm in landmarks:
        mat.append(lm.as_vector())
    return np.array(mat)