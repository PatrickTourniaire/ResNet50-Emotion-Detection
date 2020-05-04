# Imports
import numpy as np
import cv2

class Camera:

    def __init__(self, frame):
        self.frame = frame
        self.size = frame.size
    
    def get_center():
        return ((self.size[0]/2, self.size[0]/2))

    def get_focal_length():
        return self.size[1]

    def get_camera_matrix():
        center = self.get_center()
        matrix = np.array([[self.get_focal_length, 0, center[0]],
                             [0, self.get_focal_length, center[1]],
                             [0, 0, 1]], dtype = "double")
        return matrix

    def get_distance_coeffs():
        return np.zeros((4,1))

