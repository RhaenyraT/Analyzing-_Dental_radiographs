"""
Method for shape alignment using Procrustes Analysis.
"""

import numpy as np
from landmarks import Landmarks


def GPA(landmarks):
    """
    Performs Generalized Procrustes Analysis on the given landmark models.
    Based on: An introduction to Active Shape Models - Appenix A
    """
    aligned_shapes = list(landmarks)

    aligned_shapes = [shape.translate_to_origin() for shape in aligned_shapes]

    x0 = aligned_shapes[0].scale_to_unit()
    mean_shape = x0

    while True:

        for ind, lm in enumerate(aligned_shapes):
            aligned_shapes[ind] = align_shapes(lm, mean_shape)

        new_mean_shape = compute_mean_shape(aligned_shapes)

        new_mean_shape = align_shapes(new_mean_shape, x0)
        new_mean_shape = new_mean_shape.scale_to_unit().translate_to_origin()

        if ((mean_shape.as_vector() - new_mean_shape.as_vector()) < 1e-10).all():
            break

        mean_shape = new_mean_shape

    return mean_shape, aligned_shapes


def align_shapes(x1, x2):
    """
    Aligns two mean centered shapes by scaling, rotatin and translating.
    Based on: An introduction to Active Shape Models - Appenices A & D
    """
    _, s, theta = get_align_params(x1, x2)

    # aligning x1 with x2
    x1 = x1.rotate(theta)
    x1 = x1.scale(s)

    # project into the tangent space by scaling x1 with 1/(x1.x2)
    xx = np.dot(x1.as_vector(), x2.as_vector())
    return Landmarks(x1.as_vector()*(1.0/xx))


def get_align_params(x1, x2):
    """
    Computes the optimal parameters for the alignment of two shapes.
    Based on: An introduction to Active Shape Models - Appenix D
    """

    x1 = x1.as_vector()
    x2 = x2.as_vector()

    l1 = len(x1)//2
    l2 = len(x2)//2

    x1_centroid = np.array([np.mean(x1[:l1]), np.mean(x1[l1:])])
    x2_centroid = np.array([np.mean(x2[:l2]), np.mean(x2[l2:])])
    x1 = [x - x1_centroid[0] for x in x1[:l1]] + [y - x1_centroid[1] for y in x1[l1:]]
    x2 = [x - x2_centroid[0] for x in x2[:l2]] + [y - x2_centroid[1] for y in x2[l2:]]

    norm_x1_sq = (np.linalg.norm(x1)**2)
    a = np.dot(x1, x2) / norm_x1_sq

    b = (np.dot(x1[:l1], x2[l2:]) - np.dot(x1[l1:], x2[:l2])) / norm_x1_sq

    s = np.sqrt(a**2 + b**2)

    theta = np.arctan(b/a)

    # the optimal translation is chosen so as to match their centroids
    t = x2_centroid - x1_centroid

    return t, s, theta


def compute_mean_shape(landmarks):

    mat = []
    for lm in landmarks:
        mat.append(lm.as_vector())
    mat = np.array(mat)
    return Landmarks(np.mean(mat, axis=0))
