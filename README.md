# Traffic-Sign-Detection-Classification
A program that detects and classifies traffic signs based on images received from a raspberry pi camera.

I used the raspicam node (https://github.com/UbiquityRobotics/raspicam_node) to publish images to the topic raspicam_node/Image. My program then subscribes to this topic and reads in the image data from that topic which is then manipulated such that the yolo detection model and keras classification model are able to find the signs and classify them.
