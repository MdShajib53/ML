# Read a Image/Video from Web Cam using openCV

import cv2 

# Create a Camera Object
cam = cv2.VideoCapture(0)

# Read image from Camera Object

while True:
	success, img = cam.read()

	if not success:
		print("Reading Camera Failed!")


	cv2.imshow("Image Window", img)
	cv2.waitKey(1) # Pause 1ms for read the next img