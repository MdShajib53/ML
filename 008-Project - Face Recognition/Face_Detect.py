#Face Detection

import cv2 

# Create a Camera Object
cam = cv2.VideoCapture(0)

# Model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Read image from Camera Object
while True: 
	success, img = cam.read()
	img = cv2.flip(img, 1)

	if not success:
		print("Reading Camera Failed")


	faces = model.detectMultiScale(img, 1.3, 5)

	for f in faces:
		x,y,w,h = f 
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)


	cv2.imshow("Image Window", img)
	key = cv2.waitKey(1) # Pause 1ms before read the img
	if key==ord('q'): # ord -> gives ascii value of q
		break

# Release Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()
