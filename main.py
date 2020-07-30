import cv2
import time
import os

from PIL import Image
from datetime import datetime
from pyfiglet import Figlet
import numpy as np

from Face import Face
from WebcamConnect import VideoStream
from WebcamConnect.Resolution import Resolution

# === CONFIG ===
laboratory_camera = 'rtsp://192.9.45.64:554/profile2/media.smp'

jetson_onboard_camera = ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)1280, height=(int)720, '
            'format=(string)NV12, framerate=(fraction)15/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw, width=(int){}, height=(int){}, '
            'format=(string)BGRx ! '
            'videoconvert ! appsink').format(*Resolution.HD)

device_cam = 1
use_cuda = True

# === RESOURCE ===

camera_to_use = device_cam
screenshot_base_directory = "screenshots/"+datetime.now().strftime("%Y%m%d-%H%M")+"/"
classifier_xml = "TrainData/cuda/haarcascade_frontalface_default.xml"

# === LOGIC ===

def main():
    figlet = Figlet()
    
    print(figlet.renderText("PRML"))
    print("Facial Recognition - Dataset collection")
    print()

    print("Setting up directory...")

    if not os.path.exists(screenshot_base_directory):
        os.mkdir(screenshot_base_directory)
    
    webcam = VideoStream(camera_to_use)

    webcam.connect()
    
    print("Wait until the connection...")

    while not webcam.isConnected():
        pass

    print("Connected!")

    Face.set_original_resolution(*webcam.get_origin_resolution())
    
    while (classification_session(webcam)):
        pass
        

def classify_faces(frame, downscale = 1):
    face_classifier = cv2.CascadeClassifier(classifier_xml)

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if downscale:
        grayscale_frame = cv2.resize((frame.shape[1] / downscale, frame.shape[0] / downscale))
    
    detected_faces = face_classifier.detectMultiScale(grayscale_frame, 1.3, 5)

    for face in detected_faces:
        face.x = (int) (face.x * downscale)
        face.y = (int) (face.y * downscale)
        face.width = (int) (face.width * downscale)
        face.height = (int) (face.height * downscale)

    return detected_faces


face_list = []
face_uuid = 1

def classification_session(webcam: VideoStream):
    global face_list, face_uuid

    cycle_start = time.time()

    current_frame = webcam.getFrame(Resolution.FullHD)
    user_show_frame = np.copy(current_frame)

    user_show_frame = cv2.cvtColor(user_show_frame, cv2.COLOR_RGB2BGR)

    detected_faces = classify_faces(current_frame, 2)

    for face_metadata in detected_faces:
        x, y, width, height = face_metadata

        this_face_uuid = 0

        bigger_side = width if width > height else height
        
        font_size_multiplier = ( bigger_side / Face.origin_height )

        for face in face_list:
            face: Face = face

            if face.process_frame(x, y, width, height):
                color = (0,255,0) if face.should_capture() else (0,0,255)

                cv2.rectangle(user_show_frame, (x,y), (x+width, y+height), color, 2)
                cv2.putText(user_show_frame, "Face ID: {}".format(face.uuid), (x, y+height+(int)(5 * font_size_multiplier)), cv2.FONT_HERSHEY_DUPLEX, 0.15 * font_size_multiplier, color)
                break

        else:
            face_list.append(Face(face_uuid, x, y, width, height))

            cv2.rectangle(user_show_frame, (x,y), (x+width, y+height), (0,0,255), 2)
            cv2.putText(user_show_frame, "Face ID: {}".format(face_uuid), (x, y+height+(int)(5 * font_size_multiplier)), cv2.FONT_HERSHEY_DUPLEX, 0.15 * font_size_multiplier, (0,0,255))
            print("New Face: Face ID: {} @ {}".format(face_uuid, datetime.now()))

            face_uuid += 1
            

    for face in face_list:
        if face.should_delete():
            face_list.remove(face)
            print("Deleted Face: Face ID: {}, Capture Count: {}".format(face.uuid, face.screenshot_count))
        
        if face.should_capture():
            image = Image.fromarray(current_frame)
            face.screenshot(image, screenshot_base_directory)

        if not face.was_seen:
            face.forget()

        face.reset_was_seen()

    fps = 1.0 / (time.time() - cycle_start)

    cv2.putText(user_show_frame, "{:8.4f} fps".format(fps), (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (134,67,0))
    cv2.imshow("Classified Data", user_show_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True


if __name__ == "__main__":
    main()

