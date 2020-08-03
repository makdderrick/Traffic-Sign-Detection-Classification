#!/usr/bin/env python
import numpy as np
import pickle
import rospy
from keras.models import load_model
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Altering image for computational efficiency
def grayscale(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	return img
def equalize(img):
	img =cv2.equalizeHist(img)
	return img
def preprocessing(img):
	img = grayscale(img)
	img = equalize(img)
	img = img/255
	return img

# Returns sign name based off of class index
def getClassName(classNo):
	if   classNo == 0: return 'Yield'
	elif classNo == 1: return 'Stop'
	elif classNo == 2: return 'No Entry'
	elif classNo == 3: return 'General Caution'
	elif classNo == 4: return 'Turn Right Ahead'
	elif classNo == 5: return 'Turn Left Ahead'
	elif classNo == 6: return 'Ahead Only'
	elif classNo == 7: return 'Go Straight or Right'
	elif classNo == 8: return 'Go Straight or Left'

# Detects and classifies traffic signs using images from raspicam_node
def callback(img):
	rospy.loginfo("Image obtained")
	try:
		cv_image = bridge.imgmsg_to_cv2(img, "bgr8")

		# Detection
		img_det = cv2.resize(cv_image, None, fx=0.4, fy=0.4)
		height, width, channels = img_det.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img_det, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)

		# Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

					# Cropped Image where the traffic sign is 
					cropped_image = img_det[(y-10):(y+h+10),(x-10):(x+w+10)]

					# Classification
					img = np.asarray(cropped_image)
					img = cv2.resize(img, (32,32))
					img = preprocessing(img)
					img = img.reshape(1, 32, 32, 1)

					#Predictions
					predictions = model.predict(img)
					classIndex = model.predict_classes(img)
					probabilityValue =np.amax(predictions)

					if probabilityValue > threshold:
						cv2.putText(img_det, "CLASS: " , (5, 20), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.putText(img_det, "PROB: ", (5, 35), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.putText(img_det, str(getClassName(classIndex)), (60, 20), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.putText(img_det, str(round(probabilityValue*100,2) )+"%", (60, 35), font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

		
		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draws box around the traffic sign
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				color = colors[class_ids[i]]
				cv2.rectangle(img_det, (x, y), (x + w, y + h), color, 2)
				cv2.putText(img_det, label, (x, y + 30), font, .5, color, 2)
		cv2.imshow("Object Detection", img_det)
		cv2.namedWindow('Object Detection',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Object Detection', 1280,960)
		cv2.waitKey(1)


	except CvBridgeError as e:
		print(e)

def main():
	rospy.init_node('traffic_sign_observer')
	image_subscriber = rospy.Subscriber('/raspicam_node/image', Image, callback)

	rospy.spin()


model = load_model("model.h5")

font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.9

bridge = CvBridge()
classes = ["Traffic sign"]
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


if __name__ == "__main__":
	main()