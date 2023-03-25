import numpy as np
import cv2
import os
from PIL import Image
from EmployeeAttendance.settings import BASE_DIR
import face_recognition

face_recognizer = cv2.dnn.readNetFromTorch(
    BASE_DIR+'/Employee_attendance/openface.nn4.small2.v1.t7')

# Load the Caffe face detection model
modelFile = "Employee_attendance/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "Employee_attendance/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

trained_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
trained_face_recognizer.read(
    BASE_DIR+'/Employee_attendance/trainer/trainer.yml')


class FaceRecognition:

    def faceDetect(self, Entry1):
        print("IN dace detection")
        emp_id = Entry1
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
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [
                                         104, 117, 123], False, False)

            # Pass the blob through the network and get the detections
            net.setInput(blob)
            detections = net.forward()

            # Loop over the detections and draw a rectangle around the face
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # increase the confidence threshold
                    x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                    y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                    x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                    y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    face = frame[y1:y2, x1:x2]

                    # Check if the region of interest is within the frame
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        face_count += 1
                        cv2.imwrite(BASE_DIR+'/Employee_attendance/dataset/User.' +
                                    str(emp_id) + '.' + str(face_count) + ".jpg", face)

            # Show the output frame
            cv2.imshow("Face Detection", frame)
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif face_count >= 30:  # Take 30 face sample and stop video
                break

    def trainFace(self):
        print("In train face")
        # Path for face image database
        path = BASE_DIR+'/Employee_attendance/dataset'

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                img = cv2.imread(imagePath)

                # Detect faces in image
                blob = cv2.dnn.blobFromImage(cv2.resize(
                    img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                # Extract face features and labels
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array(
                            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                        print("box=", box)
                        (x, y, w, h) = box.astype("int")

                        face = img[y:y+h, x:x+w]
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)

                        face_recognizer.setInput(faceBlob)
                        vec = face_recognizer.forward()

                        faceSamples.append(vec.flatten())
                        ids.append(
                            int(os.path.split(imagePath)[-1].split(".")[1]))

            return faceSamples, ids

            # Train face recognition model
        print("\n Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        print(faces, ids)
        trained_face_recognizer.train(faces, cv2.ml.ROW_SAMPLE, np.array(ids))

        # Save the model
        trained_face_recognizer.save(
            BASE_DIR+'/Employee_attendance/trainer/trainer.xml')
        print("model saved")

        # Print the number of faces trained
        print("\n {0} faces trained. Exiting Program".format(
            len(np.unique(ids))))

    def recognizeFace(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        confidence = 0
        cam = cv2.VideoCapture(0)

        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        THRESHOLD = 0.6
        while True:

            ret, img = cam.read()

            # Detect faces using the face detection network
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            best_match_id = None

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    # Get the bounding box of the face
                    box = detections[0, 0, i, 3:7] * np.array(
                        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    (x, y, w, h) = box.astype("int")

                    # Check if face is large enough to be recognized
                    if w >= minW and h >= minH:
                        # Recognize the face using the face recognition model
                        face = img[y:y + h, x:x + w]
                        faceBlob = cv2.dnn.blobFromImage(cv2.resize(face, (96, 96)), 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        face_recognizer.setInput(faceBlob)
                        vec = face_recognizer.forward()

                        # Compare with dataset
                        dataset_path = BASE_DIR+'/Employee_attendance/dataset'
                        min_distance = float("inf")
                        for filename in os.listdir(dataset_path):
                            if filename.endswith('.jpg'):
                                # Load image and compute embedding
                                image = cv2.imread(
                                    os.path.join(dataset_path, filename))
                                imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (96, 96)), 1.0 / 255,
                                                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                                face_recognizer.setInput(imageBlob)
                                imageVec = face_recognizer.forward()

                                # Compute Euclidean distance between embeddings
                                distance = np.linalg.norm(vec - imageVec)

                                # Update closest match
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match_id = os.path.splitext(
                                        filename)[0].split('.')[1]
                                print(best_match_id)

                        # Draw the bounding box and label
                        if min_distance < THRESHOLD:
                            cv2.rectangle(
                                img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(img, best_match_id, (x+5, y-5),
                                        font, 1, (255, 255, 255), 2)
                            cv2.putText(img, str(min_distance),
                                        (x+5, y+h-5), font, 1, (255, 255, 255), 2)
                            cv2.imshow('Recognizing Face', img)

                            # Exit on pressing 'q'
            if cv2.waitKey(1) == ord('q'):
                break

            cam.release()
            cv2.destroyAllWindows()
            return best_match_id
