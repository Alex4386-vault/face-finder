import torch
import torchvision
import time
import sys

import cv2

import numpy as np

from facenet_pytorch import MTCNN
from datetime import datetime
from PIL import Image

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

device_cam = 0
user_viewport = (854,480)

use_cuda = False

# === RESOURCE ===

camera_to_use = device_cam
screenshot_base_directory = "screenshots/"+datetime.now().strftime("%Y%m%d-%H%M")+"/"
classifier_xml = "TrainData/cuda/haarcascade_frontalface_default.xml"

head_less = "--headless" in sys.argv
force_use_cuda = "--cuda" in sys.argv
debug_mode = "--debug" in sys.argv

DEVICE = "cpu"


# headless
if head_less:
    print("Running in headless mode!")

# create screenshot dir
if not os.path.exists(screenshot_base_directory):
    os.mkdir(screenshot_base_directory)

# === Cuda ===

if torch.cuda.is_available():
    if force_use_cuda:
        print("PyTorch detected CUDA, and executed with --cuda flag. Trying to use CUDA.")
        DEVICE = "cuda"
    else:
        if use_cuda:
            print("Source code default was set to use CUDA. Trying to use CUDA.")
            DEVICE = "cuda"
        else:
            print("PyTorch detected CUDA, but source code default was forcing CPU. continuing with CPU.")

else:
    if force_use_cuda:
        print("Warning! PyTorch did not detect CUDA, but executed with --cuda flag. Forcing PyTorch to use CUDA, expect some errors.")
        DEVICE = "cuda"
    else:
        if use_cuda:
            print("Warning! PyTorch did not detect CUDA, source code default wanted to use CUDA. ignoring source code default, using CPU.")
        else:
            print("Using CPU...")

torch.device(DEVICE)


# === VideoStream ===

webcam = VideoStream(camera_to_use)

webcam.connect()

print("Wait until the connection...")

while not webcam.isConnected():
    pass

print("Connected!")

Face.set_original_resolution(*webcam.get_origin_resolution())

# === Neural Network ===
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# === GLOBAL VARIABLES ===
face_list = []
face_uuid = 1

while 1:
    cycle_start = time.time()

    frame = webcam.getFrame()
    raw_faces = mtcnn.detect(Image.fromarray(frame))
    already_found_faces = []

    detected_faces = [[ int(i) for i in  face.tolist() ] for face in raw_faces[0]] if raw_faces[0] is not None else []

    #webcam_width, webcam_height = webcam.get_origin_resolution()

    if not head_less:
        user_show_frame = np.copy(frame)
        user_show_frame = cv2.cvtColor(user_show_frame, cv2.COLOR_RGB2BGR)

    # == Face capture logic ==
    for face_metadata in detected_faces:
        x, y, width, height = face_metadata

        this_face_uuid = 0

        bigger_side = width if width > height else height
        
        font_size_multiplier = ( bigger_side / Face.origin_height )
        font_scaler = 2

        for face in face_list:
            face: Face = face

            if face.process_frame(x, y, width, height):
                if face.uuid in already_found_faces:
                    continue

                color = (0,255,0) if face.should_capture() else (0,0,255)
                already_found_faces.append(face.uuid)
                
                if not head_less:
                    cv2.rectangle(user_show_frame, (x,y), (x+width, y+height), color, 2)
                    cv2.putText(user_show_frame, "Face ID: {} (Capture: {})".format(face.uuid, face.screenshot_count), (x, y+height+(int)(5 * font_scaler * font_size_multiplier + 5)), cv2.FONT_HERSHEY_DUPLEX, 0.15 * font_scaler * font_size_multiplier, color)
                break

        else:
            face_list.append(Face(face_uuid, x, y, width, height))

            if not head_less:
                cv2.rectangle(user_show_frame, (x,y), (x+width, y+height), (0,0,255), 2)
                cv2.putText(user_show_frame, "Face ID: {} (Prep)".format(face_uuid), (x, y+height+(int)(5 * font_scaler * font_size_multiplier + 5)), cv2.FONT_HERSHEY_DUPLEX, 0.15 * font_scaler * font_size_multiplier, (0,0,255))

            print()
            print("New Face: Face ID: {} @ {}".format(face_uuid, datetime.now()))

            face_uuid += 1
            

    for face in face_list:
        if face.should_delete():
            face_list.remove(face)
            print()
            print("Deleted Face: Face ID: {}, Capture Count: {}".format(face.uuid, face.screenshot_count))

            if face_uuid - 1 == face.uuid and face.screenshot_count == 0:
                face_uuid -= 1
                print("Terminate: Reverting Face ID to "+str(face_uuid))
        
        if face.should_capture():
            if face.seen_frames - face.screenshot_threshold == 1:
                if face.screenshot_count == 0:
                    print("Capturing Face: Face ID: {}".format(face.uuid), flush=True)
                else:
                    print("Recapturing Face: Face ID: {}, Capture Count: {}".format(face.uuid, face.screenshot_count), flush=True)

            image = Image.fromarray(frame)
            face.screenshot(image, screenshot_base_directory)
            

        if not face.was_seen:
            face.forget()

        face.reset_was_seen()

    # == Face capture logic end

    cycle_end = time.time()
    fps = 1.0 / (cycle_end - cycle_start)


    if not head_less:
        user_show_frame = cv2.resize(user_show_frame, user_viewport)
        cv2.rectangle(user_show_frame, (5,5), (150,25), (255,255,255), cv2.FILLED)
        cv2.putText(user_show_frame, "{:8.4f} fps".format(fps), (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0))

        cv2.imshow("OpenCV Console (press 'q' to terminate)", user_show_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting!", flush=True)
            break


    print("\b"*12, end='', flush=True)
    print("{:8.4f} fps".format(fps), end='', flush=True)

