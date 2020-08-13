import torch
import torchvision
import time
import sys

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

use_cuda = True

# === RESOURCE ===

camera_to_use = device_cam
screenshot_base_directory = "screenshots/"+datetime.now().strftime("%Y%m%d-%H%M")+"/"
classifier_xml = "TrainData/cuda/haarcascade_frontalface_default.xml"

head_less = "--headless" in sys.argv
force_use_cuda = "--cuda" in sys.argv
debug_mode = "--debug" in sys.argv

DEVICE = "cpu"

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

while True:
    cycle_start = time.time()

    frame = webcam.getFrame()
    mtcnn.detect(Image.fromarray(frame))

    cycle_end = time.time()

    fps = 1.0 / (time.time() - cycle_start)
    print(fps,"FPS")

