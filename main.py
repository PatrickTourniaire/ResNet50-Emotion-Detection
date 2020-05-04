# Import modules
import numpy as np
import numpy.linalg as la
import cv2
import dlib
import keras
from keras.models import load_model

# Local modules
from src.utils import *
from src.camera import Camera
from src.images import *

# Paths
PATH_FACE_LANDMARKS = "models/shape_predictor_68_face_landmarks.dat"
PATH_EMOTIONS = "models/emotion_detection_model.h5"

# Video stream
video_stream = cv2.VideoCapture(CAMERA_INDEX)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PATH_FACE_LANDMARKS)

# Predict emotions
emotions_model = load_model(PATH_EMOTIONS, compile=False)

# Main loop
while True:
    # Read frame and convert to black and white
    _, frame = video_stream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Camera measurements
    size = frame.shape

    focal_length = size[1]
    center = (size[0]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double")
    distance_coeffs = np.zeros((4,1))

    # 2D image points on face
    image_points = []

    # Find faces and get land marks
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_emotion_shape = (1, 48, 48, 1)
        img_emotion = prep_image(gray, (x1,x2,y1,y2), face_emotion_shape, toGray=False)

        # Predict emotion of face
        emotion_array = emotions_model.predict(img_emotion)
        predicted_emotion = EMOTIONS[np.argmax(emotion_array)]

        cv2.putText(frame,"Emotion " + predicted_emotion, 
                        (x1,y1 - 20), 
                        FONT, 
                        FONT_SCALE,
                        FONT_COLOR,
                        LINE_TYPE)

        # Draw face rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), FRAME_COLOR, FRAME_THICKNESS)

        landmarks = predictor(gray, face)
        points = set_image_point(landmarks)
        image_points = np.array(points, dtype="double")

        for point in points:
            x = point[0]
            y = point[1]

            cv2.circle(frame, (x,y), CIRCLE_DIAMETER, CIRCLE_COLOR, -1)

        if (image_points.size != 0):
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, distance_coeffs)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, distance_coeffs)

            point_nose_start = points[0]
            point_nose_end = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            nose_start2end = (point_nose_end[0] - point_nose_start[0], point_nose_end[1] - point_nose_start[1])
            nose_start2cam = (center[0] - point_nose_start[0], center[1] - point_nose_start[1])
            
            dot_prod = np.dot(nose_start2end, nose_start2cam)
            prod_length = abs(la.norm(nose_start2end)) * abs(la.norm(nose_start2cam)) 
            face_cam_angle = np.arccos(dot_prod/prod_length)
            
            cv2.putText(frame,"Head angle to camera: " + str(face_cam_angle), 
                        TOP_LEFT_CORNER, 
                        FONT, 
                        FONT_SCALE,
                        FONT_COLOR,
                        LINE_TYPE)

            cv2.line(frame, point_nose_start, point_nose_end, CIRCLE_COLOR, LINE_THINKNESS)
            cv2.line(frame, point_nose_start, (int(center[0]), int(center[1])), FRAME_COLOR, LINE_THINKNESS)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
