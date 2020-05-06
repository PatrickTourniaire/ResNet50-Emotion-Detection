
# ResNet 50 Emotion Detection
This project was created for educational purposes to explore the ResNet50 architecture's application in live emotion detection. The project started by exploring a way to measure attention, but pivoted to explore this type of convolutional neural networks. 

## Some Results
| ![Anger](https://i.imgur.com/sjPAvom.png) |![Fear](https://i.imgur.com/0TrXn44.png)  |![Happy](https://i.imgur.com/0e2Ag4N.png) |
|--|--|--|


## Prerequisites

### Data
To run this project you need to download a data set from Kaggle. Firstly, create the data directory by navigating to the project root and running the following;

    mkdir data; cd data 
Then you need to make sure you have the [Kaggle api installed](https://github.com/Kaggle/kaggle-api). When you have that installed you can run the following to get the data set.

    kaggle competitions download -c facial-keypoints-detector
Now to get it all in correct folders run;

    mkdir emotions; unzip facial-keypoints-detector.zip -d emotions/

### Models
You will need the the opencv facial landmark detector model which you can download from [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat).
You also need to run the notebook resnet-50-emotion-detection to get the model to run the project. You can run it in Kaggle if you wish by cloning my notebook [here](https://www.kaggle.com/unityrift/resnet-50-emotion-detection).
 
## Running 

    python3 main.py
If you wish to train the model for yourself or you want to make some changes to it then you need to open the resnet-50-emotion-detection notebook located in the project root. If you want to run it in Kaggle then you can clone my [Kaggle notebook](https://www.kaggle.com/unityrift/resnet-50-emotion-detection).
