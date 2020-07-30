import cv2
from .Resolution import Resolution

class VideoStream():
    '''The VideoStream handler class for retrieving frames from the webcam'''
    connected = False

    def __init__(self, videoStream):
        '''Initializes VideoStream with VideoStream Location'''
        self.videoStream = videoStream
        self.vc = None

    def connect(self):
        '''Connect and create VideoCapture via provided VideoStream Location'''
        self.vc = cv2.VideoCapture(self.videoStream)

        while not self.vc.isOpened():
            pass

        self.connected = True

    def disconnect(self):
        '''Disconnect from VideoStream and close VideoCapture'''
        self.vc.release()
        self.vc = None
        self.connected = False

    def isConnected(self):
        '''Check whether this VideoStream is connected'''
        return self.connected

    def getFrame(self, resolution=None):
        '''
        get CurrentFrame from the VideoCapture  
        (not safe, should be checked with `isConnected` beforehand)
        '''
        _, tmpFrame = self.vc.read()

        tmpFrame = cv2.cvtColor(tmpFrame, cv2.COLOR_BGR2RGB)

        if resolution is not None:
            tmpFrame = cv2.resize(tmpFrame, resolution)

        return tmpFrame

    def get_origin_resolution(self):
        if self.isConnected():
            origin_width = self.vc.get(cv2.CAP_PROP_FRAME_WIDTH) / 10
            origin_height = self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT) / 10

            return (origin_width, origin_height)
        else:
            raise ConnectionError("The Webcam is not connected, yet.")
