# Part-2 : Data Preparation

import numpy as np
import cv2
import os


dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}
offset = 20

classId = 0

for f in os.listdir(dataset_path):
	if f.endswith(".npy"):

		nameMap[classId] = f[:-4] # shajib.npy--> shajib
		# X-value
		dataItem = np.load(dataset_path + f)
		m = dataItem.shape[0]
		faceData.append(dataItem)

		# Y-value
		target = classId * np.ones((m,))
		classId+=1
		labels.append(target)

XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1,1))

print(XT.shape)
print(yT.shape)
print(nameMap)




# Part-3: KNN-Algorithm

def dist(p, q):
	return np.sqrt(np.sum((p-q)**2))

def knn(X,y,Xt,k=11):
	m = X.shape[0]
	dlist = []

	for i in range(m):
		d = dist(X[i],Xt)
		dlist.append((d,y[i][0]))

	dlist = sorted(dlist)
	dlist = np.array(dlist[:k])
	labels = dlist[:,1]

	labels, cnts = np.unique(labels, return_counts=True)
	idx = cnts.argmax()
	pred = labels[idx]

	return int(pred)


# Part-4:  Prediction 

# Create a Camera Object
cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


# Read image from Camera Object
while True: 
	success, img = cam.read()
	img = cv2.flip(img, 1)

	if not success:
		print("Reading Camera Failed")

	faces = model.detectMultiScale(img, 1.3, 5)
	
	# Render a box around each face and predicts its name
	for f in faces:
		x,y,w,h = f 
		print(f)
		
		# Crop and save the largest face
		cropped_face = img[y-offset:y+h+offset, x-offset:x+offset+w]
		cropped_face = cv2.resize(cropped_face,(100,100))

		# Predict the name using KNN
		classPredicted = knn(XT, yT, cropped_face.flatten())
		namePredicted = nameMap.get(classPredicted, "Unknown")

		# Display Name and Box
		cv2.putText(img, namePredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2,cv2.LINE_AA)
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

	cv2.imshow("Predication Window", img)

	key = cv2.waitKey(1) & 0xFF # Pause 1ms before read the img
	if key==ord('q'): # ord -> gives ascii value of q
		break

# Release Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()

