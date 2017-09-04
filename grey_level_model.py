"""
A class for create grey-level models.
"""

import math
import numpy as np
from scipy import linspace, asarray


class GreyLevelModel(object):
    """
    A Class to represent the grey-level structure in the locality
    of the model points of an incisor.

    Based on: An introduction to Active Shape Models (pp. 235-238)
    """

    def __init__(self):
        self.profiles = []
        self.mean_profile = None
        self.covariance = []

    def build(self, images, gimages, models, point_ind, k):

        for ind in range(len(images)):
            self.profiles.append(Profile(images[ind], gimages[ind], models[ind], point_ind, k))

        temp = []
        for profile in self.profiles:
            temp.append(profile.samples)
        temp = np.array(temp)
        
        self.mean_profile = (np.mean(temp, axis=0))
        self.covariance = (np.cov(temp, rowvar=0))

    def quality_of_fit(self, samples):

        return (samples - self.mean_profile).T \
                .dot(self.covariance) \
                .dot(samples - self.mean_profile)

    
class Profile(object):
    """
    A profile of samples normal to the model boundary through a model point.

    Based on: An introduction to Active Shape Models (pp. 235-238)
    """

    def __init__(self, image, grad_image, model, point_ind, k):

        self.image = image
        self.grad_image = grad_image
        self.model_point = model.points[point_ind, :]
        self.k = k
        self.normal = self.compute_normal(model.points[(point_ind-1) % 40, :],
                                              model.points[(point_ind+1) % 40, :])
        self.points, self.samples = self.get_samples()


    def compute_normal(self, p_prev, p_next):

        n1 = get_normal(p_prev, self.model_point)
        n2 = get_normal(self.model_point, p_next)
        n = (n1 + n2) / 2
        return n / np.linalg.norm(n)

    def get_samples(self):

        pos_points, pos_values, pos_grads = self.get_params(-self.normal)
        neg_points, neg_values, neg_grads = self.get_params(self.normal)

        neg_values = neg_values[::-1]  
        neg_grads = neg_grads[::-1]  
        neg_points = neg_points[::-1]  
        points = np.vstack((neg_points, pos_points[1:, :]))
        values = np.append(neg_values, pos_values[1:])
        grads = np.append(neg_grads, pos_grads[1:])

        div = max(sum([math.fabs(v) for v in values]), 1)
        samples = [float(g)/div for g in grads]

        return points, samples

    def get_params(self, direction):

        a = asarray(self.model_point)
        b = asarray(self.model_point + direction*self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k+1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k+1))
        values = self.image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        grad_values = self.grad_image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        return coordinates.T, values, grad_values

def get_normal(p1, p2):

    return np.array([p1[1] - p2[1], p2[0] - p1[0]])