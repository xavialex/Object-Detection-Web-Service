# Object Detection Web Service

This project makes use of the SSD Mobilenet model trained in COCO dataset to perform object and people detection from the video signal received by a wired webcam. It then publishes a UI delivered by an HTTP Server that shows the amount of people detected in that frame and that frame with the detection boxes and the likelihoods.

## Prerequisites

You'll need Python 3.X and some dependencies. If you're a conda user, you can create an environment from the ```object_detection_env.yml``` file using the Terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the ```object_detection_env.yml``` file:

    ```conda env create -f object_detection_env.yml```
2. Activate the new environment:
    * Windows: ```activate myenv```
    * macOS and Linux: ```source activate myenv``` 
    
    NOTE: Replace ```myenv``` with the name of the environment.
3. Verify that the new environment was installed correctly:

    ```conda list```

## Use

Launch *people_detector_from_webcam_ws.py* from your favorite IDE or from the command prompt. You'll then see two windows, one with the image that's being captured by the program (video, url, webcam, etc.), and other with the same signal but with the detections (bounding boxes) overlapped into it. You can also open your internet browser and address to 127.0.0.1:8080. A basic UI will appear showing the amount of people detected in that frame and that frame with the detection boxes and the likelihoods. Press the *'q'* button in the resulting floating window to close the program.
