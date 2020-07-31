from PIL import Image
import os

# THERE ARE NO FACES

class Face():
    
    screenshot_threshold = 7
    forget_threshold = 15

    origin_width = 1280
    origin_height = 720

    @classmethod
    def set_original_resolution(cls, width, height):
        cls.origin_height = height
        cls.origin_width


    def __init__(self, uuid, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.uuid = uuid
        self.was_seen = False
        self.forget_value = 0
        self.seen_frames = 0
        self.screenshot_count = 0

    def was_seen(self):
        return self.was_seen

    def get_move_threshold(self):
        this_width_threshold = self.width / 2
        input_width_threshold = self.origin_width / 10

        this_height_threshold = self.height / 2
        input_height_threshold = self.origin_height / 10

        x = this_width_threshold if this_width_threshold > input_width_threshold else input_width_threshold
        y = this_height_threshold if this_height_threshold > input_height_threshold else input_height_threshold

        return x, y
    
    def in_threshold(self, x, y, width, height):
        move_threshold = self.get_move_threshold()
        return (
            abs(self.x - x) <= move_threshold[0] and abs(self.y - y) <= move_threshold[1] and
            abs(self.width - width) <= move_threshold[0] and abs(self.height - height) <= move_threshold[1]
        )

    def update_features(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_image(self, original_frame: Image) -> Image:
        return original_frame.crop((self.x, self.y, self.x + self.width, self.y + self.height))

    def save_image(self, original_frame: Image, save_location: str):
        image = self.get_image(original_frame)
        image.save(save_location)

    def process_frame(self, x, y, width, height):
        if self.in_threshold(x, y, width, height):
            self.update_features(x, y, width, height)
            
            if not self.was_seen:
                self.was_seen = True
                self.forget_value = 0

            self.seen_frames += 1
            return True

        return False
    
    def forget(self):
        self.was_seen = False
        self.forget_value += 1
        self.seen_frames = 0
    
    def reset_was_seen(self):
        self.was_seen = False

    def should_delete(self):
        return self.forget_value > self.forget_threshold and not self.was_seen

    def should_capture(self):
        return self.seen_frames > self.screenshot_threshold and self.was_seen

    def screenshot(self, original_frame: Image, save_folder: str):
        self.screenshot_count += 1

        uuid_dir = os.path.join(save_folder, str(self.uuid))

        if not os.path.exists(uuid_dir):
            os.mkdir(uuid_dir)

        save_location = os.path.join(uuid_dir, str(self.screenshot_count)+".jpg")
        self.save_image(original_frame, save_location)

