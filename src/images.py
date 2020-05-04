import cv2
import numpy as np

def prep_image(frame, segment, shape, toGray=True, hasSegment=True):
    # Image properties
    width = shape[1]
    height = shape[2]
    
    if hasSegment:
        # Only get certain segment of image
        frame = frame[segment[2]:segment[3], segment[0]:segment[1]]
    img_emotion = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
    
    if toGray:
        img_emotion = cv2.cvtColor(img_emotion, cv2.COLOR_BGR2GRAY)
    img_emotion = np.array(img_emotion).reshape(shape)

    return img_emotion
