# import the necessary packages
import numpy as np
import argparse
import pickle
import cv2
import os
import imutils

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(['E:/Face recognition/opencv-face-recognition/opencv-face-recognition/face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['E:/Face recognition/opencv-face-recognition/opencv-face-recognition/face_detection_model',
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# initialize the total number of faces processed
total = 0
knownNames=[]
knownEmbeddings=[]

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch('E:/Face recognition/opencv-face-recognition/opencv-face-recognition/openface_nn4.small2.v1.t7')

# loop over the image paths
for i in os.listdir('E:/Face recognition/opencv-face-recognition/opencv-face-recognition/Sub_Data'):
	# extract the person name from the image path
	for no,j in enumerate(os.listdir(os.path.sep.join(['E:\Face recognition\opencv-face-recognition\opencv-face-recognition\Sub_Data',i]))):
        
		image = cv2.imread(os.path.sep.join(['E:\Face recognition\opencv-face-recognition\opencv-face-recognition\Sub_Data',i,j]))
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                    1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

# loop over the detections
		for i1 in range(0, detections.shape[2]):
            	
			confidence = detections[0, 0, i1, 2]

			if confidence > 0.6:
	
				box = detections[0, 0, i1, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

		
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

	
				if fW < 20 or fH < 20:
					continue
                
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec=embedder.forward()
                
				name=i

			# add the name of the person + corresponding face
			# embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total+=1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open('embedding', "wb")
f.write(pickle.dumps(data))
f.close()