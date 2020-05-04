import numpy as np
import cv2

# Emotions constants
EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Constants for DLIB shape points
NOSE_TIP           = 30
CHIN               = 8
LEFT_EYE_CORNER    = 36
RIGHT_EYE_CORNER   = 45
MOUTH_CORNER_LEFT  = 48
MOUTH_CORNER_RIGHT = 54

# General constants
FRAME_COLOR     = (0, 255, 0)
FRAME_THICKNESS = 3
CIRCLE_DIAMETER = 4
CIRCLE_COLOR    = (255, 0, 0)
LINE_THINKNESS  = 2
CAMERA_INDEX    = -1

# CV2 text constants
FONT                  = cv2.FONT_HERSHEY_SIMPLEX
TOP_LEFT_CORNER       = (0,20)
TOP_UNDER_LEFT_CORNER = (0,40)
FONT_SCALE            = 0.5
FONT_COLOR            = (255,255,255)
LINE_TYPE             = 2

# 3D model points 
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-255.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

# Image points from feed to dlib
def set_image_point(landmarks):

    nose_tip = (landmarks.part(NOSE_TIP).x, landmarks.part(NOSE_TIP).y)
    chin = (landmarks.part(CHIN).x, landmarks.part(CHIN).y)
    left_eye_corner = (landmarks.part(LEFT_EYE_CORNER).x, landmarks.part(LEFT_EYE_CORNER).y)
    right_eye_corner = (landmarks.part(RIGHT_EYE_CORNER).x, landmarks.part(RIGHT_EYE_CORNER).y)
    mouth_corner_left = (landmarks.part(MOUTH_CORNER_LEFT).x, landmarks.part(MOUTH_CORNER_RIGHT).y)
    mouth_corner_right = (landmarks.part(MOUTH_CORNER_RIGHT).x, landmarks.part(MOUTH_CORNER_RIGHT).y)

    return ([nose_tip, chin, left_eye_corner, right_eye_corner, mouth_corner_left, mouth_corner_right])