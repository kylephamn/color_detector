import cv2
import numpy as np
import requests
import threading
import queue
import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.label import Label

# API endpoint
API_URL = "https://www.thecolorapi.com/id"


class ColorDetectorWidget(Widget):
    tracking = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(ColorDetectorWidget, self).__init__(**kwargs)

        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Image widget to display video feed
        self.image_widget = Image(size=self.size, pos=self.pos)
        self.add_widget(self.image_widget)

        # Bind size and position to adjust when the window size changes
        self.bind(size=self.update_image_size, pos=self.update_image_size)

        # Add toggle button for color tracking
        self.toggle_button = Button(text="Enable Tracking", size_hint=(1, 0.1))
        self.toggle_button.bind(on_press=self.toggle_tracking)

        # Add label to display color information
        self.color_label = Label(text="", size_hint=(1, 0.1), halign='left', valign='middle')
        self.color_label.bind(size=self.color_label.setter('text_size'))

        # Add widgets to layout
        self.layout = BoxLayout(orientation='vertical')
        self.layout.add_widget(self.image_widget)
        self.layout.add_widget(self.toggle_button)
        self.layout.add_widget(self.color_label)
        self.add_widget(self.layout)

        # Initialize variables
        self.frame = None
        self.frame_no_overlay = None
        self.hsv_value = None
        self.lower_bound = None
        self.upper_bound = None
        self.cache = {}  # Cache for API results
        self.api_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Start the video capture thread
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Start the API processing thread
        self.api_thread = threading.Thread(target=self.process_api_queue, daemon=True)
        self.api_thread.start()

        # Schedule the update method
        Clock.schedule_interval(self.update, 1.0 / 30)  # 30 FPS

        # Bind touch events
        Window.bind(on_touch_down=self.on_touch_down)

    def update_image_size(self, *args):
        self.image_widget.size = self.size
        self.image_widget.pos = self.pos

    def toggle_tracking(self, instance):
        self.tracking = not self.tracking
        if self.tracking:
            self.toggle_button.text = "Disable Tracking"
            # Clear any displayed color info
            self.color_label.text = ''
            self.hsv_value = None
            self.lower_bound = None
            self.upper_bound = None
        else:
            self.toggle_button.text = "Enable Tracking"
            # Clear tracking data
            self.hsv_value = None
            self.lower_bound = None
            self.upper_bound = None

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                continue
            # Put the frame in the queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Drop frame if the queue is full
                pass
            time.sleep(0.01)  # Small delay to yield control

    def update(self, dt):
        try:
            frame = self.frame_queue.get_nowait()
            self.frame_no_overlay = frame.copy()

            if self.tracking and self.hsv_value is not None:
                frame = self.track_color(frame)

            # Convert frame to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            texture.flip_vertical()
            self.image_widget.texture = texture

        except queue.Empty:
            pass

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return

        x, y = touch.pos

        # Map touch coordinates to frame coordinates
        frame_x = int(x * self.width / self.image_widget.width)
        frame_y = int((self.image_widget.height - y) * self.height / self.image_widget.height)

        # Ensure coordinates are within the frame
        if frame_x >= self.width or frame_y >= self.height or frame_x < 0 or frame_y < 0:
            return

        # Get pixel value from the frame
        if self.frame_no_overlay is None:
            return

        bgr_pixel = self.frame_no_overlay[frame_y, frame_x]
        b, g, r = map(int, bgr_pixel)

        if self.tracking:
            # Set up color tracking
            self.set_tracking_color(bgr_pixel)
        else:
            # Use cached result if available
            rgb_key = (r, g, b)
            if rgb_key in self.cache:
                color_name = self.cache[rgb_key]
                self.update_color_info(color_name, r, g, b)
            else:
                # Put the API request in the queue
                self.api_queue.put((rgb_key,))
                # Store the touch position for displaying the color info
                self.touch_position = touch.pos

    def set_tracking_color(self, bgr_pixel):
        # Convert BGR to HSV
        hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)
        self.hsv_value = hsv_pixel[0][0]
        sensitivity = 15  # Adjust sensitivity as needed

        # Set the lower and upper HSV range for the selected color
        h = self.hsv_value[0]
        self.lower_bound = np.array([max(h - sensitivity, 0), 50, 50])
        self.upper_bound = np.array([min(h + sensitivity, 179), 255, 255])

        print(f"Tracking HSV Value: {self.hsv_value}")
        print(f"Lower Bound: {self.lower_bound}")
        print(f"Upper Bound: {self.upper_bound}")

    def track_color(self, frame):
        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for the selected color
        mask = cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)

        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Optionally, draw the center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        return frame

    def update_color_info(self, color_name, r, g, b):
        text = f'{color_name} (R:{r}, G:{g}, B:{b})'
        self.color_label.text = text
        print(text)  # Optional: Print to console for debugging

    def process_api_queue(self):
        while not self.stop_event.is_set():
            try:
                rgb_key = self.api_queue.get(timeout=0.1)
                r, g, b = rgb_key[0]
                color_name = self.get_color_name(r, g, b)
                self.cache[rgb_key[0]] = color_name
                # Update the color info on the main thread
                Clock.schedule_once(lambda dt: self.update_color_info(color_name, r, g, b), 0)
                self.api_queue.task_done()
            except queue.Empty:
                continue

    def get_color_name(self, r, g, b):
        params = {'rgb': f'rgb({r},{g},{b})'}
        try:
            response = requests.get(API_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            color_name = data.get('name', {}).get('value', 'Unknown')
            return color_name
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return "Unknown"

    def on_stop(self):
        self.stop_event.set()
        self.capture_thread.join()
        self.api_thread.join()
        self.capture.release()

    def on_parent(self, widget, parent):
        if not parent:
            # Widget is being removed from the widget tree
            self.on_stop()


class ColorDetectorApp(App):
    def build(self):
        return ColorDetectorWidget()


# Run the application
if __name__ == "__main__":
    ColorDetectorApp().run()
