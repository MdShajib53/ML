# Part-1 : Data Collection and Store

import numpy as np 
import cv2

# Create a Camera Object
cam = cv2.VideoCapture(0)

# Ask the name
fileName = input("Enter the name of the person: ")
dataset_path = "./data/"
offset = 20

# Model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Create a list of save face data
faceData = []
skip = 0



# Read image from Camera Object
while True: 
	success, img = cam.read()
	img = cv2.flip(img, 1)

	if not success:
		print("Reading Camera Failed")

	# Store the gray image
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = model.detectMultiScale(img, 1.3, 5)
	faces = sorted(faces, key=lambda f:f[2]*f[3]) # sort the face with largest boundary box
		
	if len(faces)>0: 

		f = faces[-1] # pick largest face

		x,y,w,h = f 
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

		# Crop and save the largest face
		cropped_face = img[y-offset:y+h+offset, x-offset:x+offset+w]
		cropped_face = cv2.resize(cropped_face,(100,100))

		skip+=1
		if skip%20==0:
			faceData.append(cropped_face)
			print("Saved so far: ", skip/20)


	cv2.imshow("Image Window", img)
	cv2.imshow("Cropped Face", cropped_face)
	key = cv2.waitKey(1) & 0xFF # Pause 1ms before read the img
	if key==ord('q'): # ord -> gives ascii value of q
		break

# Write the faceData on the Disk
faceData = np.asarray(faceData)
print(faceData.shape)
m = faceData.shape[0]
faceData = np.reshape(faceData, (m, -1))

print(faceData.shape)

# Save on the Disk as np array
filepath = dataset_path+fileName+".npy"
np.save(filepath, faceData)
print("Data Saved Successfully"+filepath)


# Release Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()


