import cv2
import time
from PIL import Image
import os
from datetime import datetime

print("Capture Session Init...")

jetsonOnBoardCam = ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)1280, height=(int)720, '
            'format=(string)NV12, framerate=(fraction)15/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw, width=(int){}, height=(int){}, '
            'format=(string)BGRx ! '
            'videoconvert ! appsink').format(1280, 720)

#captureSession = cv2.VideoCapture(jetsonOnBoardCam)
captureSession = cv2.VideoCapture(1)
faceClassifier = cv2.CascadeClassifier('trainData/haarcascade_frontalface_default.xml')

currentFrameNo = 0
forgetThreshold = 5
screenshotThreshold = 7

recordedFaces = []

moveThreshold = (
    captureSession.get(cv2.CAP_PROP_FRAME_WIDTH) / 10,
    captureSession.get(cv2.CAP_PROP_FRAME_HEIGHT) / 10
)

lastFPSShow = time.time()
lastUUID = 0

screenshotBaseDir = "screenshots/"+datetime.now().strftime("%Y%m%d-%H%M")+"/"
os.mkdir(screenshotBaseDir)

while(True):
    start_time = time.time()
    ret, currentFrame = captureSession.read()

    #currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
    grayScaleFrame = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
    
    faceList = faceClassifier.detectMultiScale(grayScaleFrame, 1.3, 5)
    
    faceRecognitionFrame = currentFrame

    currentFaces = []
    for recordedFace in recordedFaces:
        recordedFace['seen']['prevSeen'] = False

    for x,y,width,height in faceList:
        currentFace = {
            "x": x,
            "y": y,
            "width": width,
            "hight": height,
            "seen": {
                "uuid": lastUUID,
                "prevSeen": False,
                "screenshotCount": 0,
                "seenFrames": 0,
                "forgetValue": 0
            }
        }
        currentFaces.append(currentFace)

        for recordedFace in recordedFaces:
            if abs(recordedFace['x'] - x) <= moveThreshold[0] and abs(recordedFace['y'] - y) <= moveThreshold[1]:
                recordedFace['x'] = x
                recordedFace['y'] = y
                recordedFace['width'] = width
                recordedFace['height'] = height

                recordedFace['seen']['prevSeen'] = True
                recordedFace['seen']['seenFrames'] += 1
                recordedFace['seen']['forgetValue'] = 0

                if recordedFace['seen']['seenFrames'] > screenshotThreshold:
                    screenshot = Image.fromarray(cv2.cvtColor(currentFrame,cv2.COLOR_BGR2RGB))
                    faceCrop = screenshot.crop((x,y,x+width,y+height))

                    if not os.path.isdir(screenshotBaseDir+str(recordedFace['seen']['uuid'])):
                        os.mkdir(screenshotBaseDir+str(recordedFace['seen']['uuid'])+"/")

                    faceCrop.save(screenshotBaseDir+str(recordedFace['seen']['uuid'])+"/"+str(recordedFace['seen']['screenshotCount'])+".jpg")
                    recordedFace['seen']['screenshotCount'] += 1
                break
        else:
            recordedFaces.append(currentFace)
            lastUUID += 1

        #faceRecognitionFrame = cv2.rectangle(faceRecognitionFrame, (x,y), (x+width, y+width), (255,0,0), 2)
        
    for recordedFace in recordedFaces:
        if recordedFace['seen']['prevSeen'] == False:
            recordedFace['seen']['forgetValue'] += 1
        
        if recordedFace['seen']['forgetValue'] > forgetThreshold:
            recordedFaces.remove(recordedFace)


    #cv2.imshow("Current Input", faceRecognitionFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if time.time() - lastFPSShow > 1:
        print("Update: ", 1.0 / (time.time() - start_time), "fps")
        print(recordedFaces)
        lastFPSShow = time.time()

    currentFrameNo += 1

captureSession.release()
cv2.destroyAllWindows()
