import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "attendance_img"
images = []
classNames = []

imagesList = os.listdir(path)

for imgName in imagesList:
    image = cv2.imread(f'{path}/{imgName}')
    images.append(image)
    classNames.append(os.path.splitext(imgName)[0])

print(classNames)

def doEncondings(images):
    encodeImages = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeImages.append(encodeImg)
    return encodeImages

def fillAttendance(name):
    with open('attendace.csv','r+') as f:
        checkData = f.readlines();
        checkedNames = []
        for data in checkData:
            entry = data.split(',')
            checkedNames.append(entry[0])
        if name not in checkedNames:
            timeRaw = datetime.now()
            time = timeRaw.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{time}')

encodedImages = doEncondings(images)
print("DONE WITH ENCODING")

cam = cv2.VideoCapture(0)
while True:
    success, img = cam.read()
    imgShrink = cv2.resize(img,(0,0),None,0.25,0.25)
    imgShrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesLocation = face_recognition.face_locations(imgShrink)
    encodeFaces = face_recognition.face_encodings(imgShrink,facesLocation)

    for encodeFace, faceLocation in zip(encodeFaces,facesLocation):
        matches = face_recognition.compare_faces(encodedImages, encodeFace)
        faceDistance = face_recognition.face_distance(encodedImages, encodeFace)
        print(faceDistance)
        bestMatchIndex = np.argmin(faceDistance)

        if matches[bestMatchIndex]:
            detectedPerson = classNames[bestMatchIndex].upper()
            print(detectedPerson)
            y1,x2,y2,x1 = faceLocation
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img, detectedPerson, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
            fillAttendance(detectedPerson)
    
    cv2.imshow("CCTV",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    