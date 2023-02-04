import numpy as np
import cv2
# from sklearn.externals 

import os
# import sqlite3
import numpy as np
from PIL import Image
from EmployeeAttendance.settings import BASE_DIR

detector = cv2.CascadeClassifier(BASE_DIR+'/Employee_attendance/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


class FaceRecognition:    

    def faceDetect(self, Entry1):
        face_id = Entry1
        # Load the Caffe face detection model
        modelFile = "Employee_attendance/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "Employee_attendance/deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        # Start the video capture
        cap = cv2.VideoCapture(0)

        face_count = 0

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            # Check if the frame is valid
            if frame is None:
                break
            
            # Create a 4D blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

            # Pass the blob through the network and get the detections
            net.setInput(blob)
            detections = net.forward()

            # Loop over the detections and draw a rectangle around the face
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7: # increase the confidence threshold
                    x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                    y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                    x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                    y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    face = frame[y1:y2, x1:x2]
                    
                    # Check if the region of interest is within the frame
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        face_count += 1
                        cv2.imwrite(BASE_DIR+'/Employee_attendance/dataset/User.' + str(face_id) + '.' + str(face_count) + ".jpg", face)

            # Show the output frame
            cv2.imshow("Face Detection", frame) 
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif face_count >= 30: # Take 30 face sample and stop video
                 break

    
    def trainFace(self):
        # Path for face image database
        path = BASE_DIR+'/Employee_attendance/dataset'

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                face_id = int(os.path.split(imagePath)[-1].split(".")[1])
                print("face_id",face_id)
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(face_id)

            return faceSamples,ids

        print ("\n Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.save(BASE_DIR+'/Employee_attendance/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))


    def recognizeFace(self):
        recognizer.read(BASE_DIR+'/Employee_attendance/trainer/trainer.yml')
        cascadePath = BASE_DIR+'/Employee_attendance/haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX

        confidence = 0
        cam = cv2.VideoCapture(0)

        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)

        while True:

            ret, img =cam.read()

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                face_id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                # Check if confidence is less then 100 ==> "0" is perfect match 
                if (confidence < 100):
                    name = 'Detected'
                else:
                    name = "Unknown"
                
                cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            
            cv2.imshow('Detect Face',img) 

            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            if confidence > 50:
                break

        print("\n Exiting Program")
        cam.release()
        cv2.destroyAllWindows()
        print(face_id)
        return face_id