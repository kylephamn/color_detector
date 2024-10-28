import cv2
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
import time

# API endpoint
API_URL = "https://www.thecolorapi.com/id"

class ColorDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a canvas to display the video feed
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Add toggle button for color tracking
        self.tracking = False
        self.toggle_button = tk.Button(window, text="Enable Tracking", command=self.toggle_tracking)
        self.toggle_button.pack()

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_mouse_click)

        # Initialize variables
        self.photo = None
        self.color_info = ''
        self.delay = 15  # milliseconds
        self.cache = {}  # Cache for API results
        self.api_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Variables for color tracking
        self.hsv_value = None
        self.lower_bound = None
        self.upper_bound = None
        self.tracking_frame = None

        # Start the video capture thread
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Start the API processing thread
        self.api_thread = threading.Thread(target=self.process_api_queue, daemon=True)
        self.api_thread.start()

        # Start the video update loop
        self.update_video()

        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def toggle_tracking(self):
        self.tracking = not self.tracking
        if self.tracking:
            self.toggle_button.config(text="Disable Tracking")
            # Clear any displayed color info
            self.color_info = ''
            self.canvas.delete('color_info')
        else:
            self.toggle_button.config(text="Enable Tracking")
            # Clear tracking data
            self.hsv_value = None
            self.lower_bound = None
            self.upper_bound = None
            self.tracking_frame = None

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Resize frame for faster processing
            # frame = cv2.resize(frame, (self.width // 2, self.height // 2))
            # Put the frame in the queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Drop frame if the queue is full
                pass
            time.sleep(0.01)  # Small delay to yield control

    def update_video(self):
        if not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.02)
                self.frame_no_overlay = frame.copy()

                if self.tracking and self.hsv_value is not None:
                    frame = self.track_color(frame)

                # Convert the image to PhotoImage
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
                # Display the image
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Display color info if available
                if self.color_info and not self.tracking:
                    text, x, y = self.color_info
                    self.display_color_info(text, x, y)

            except queue.Empty:
                pass

            # Schedule the next frame update
            self.window.after(self.delay, self.update_video)

    def on_mouse_click(self, event):
        x, y = event.x, event.y

        # Ensure coordinates are within the frame
        if x >= self.width or y >= self.height:
            return

        # Get pixel value from the frame
        try:
            bgr_pixel = self.frame_no_overlay[int(y), int(x)]
        except IndexError:
            return  # Clicked outside the frame

        b, g, r = map(int, bgr_pixel)

        if self.tracking:
            # Set up color tracking
            self.set_tracking_color(bgr_pixel)
        else:
            # Use cached result if available
            rgb_key = (r, g, b)
            if rgb_key in self.cache:
                color_name = self.cache[rgb_key]
                self.update_color_info(color_name, r, g, b, x, y)
            else:
                # Put the API request in the queue
                self.api_queue.put((rgb_key, x, y))

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

    def update_color_info(self, color_name, r, g, b, x, y):
        text = f'{color_name} (R:{r}, G:{g}, B:{b})'
        self.color_info = (text, x, y)
        print(text)  # Optional: Print to console for debugging

    def process_api_queue(self):
        while not self.stop_event.is_set():
            try:
                rgb_key, x, y = self.api_queue.get(timeout=0.1)
                r, g, b = rgb_key
                color_name = self.get_color_name(r, g, b)
                self.cache[rgb_key] = color_name
                # Update the color info on the main thread
                self.window.after(0, self.update_color_info, color_name, r, g, b, x, y)
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

    def display_color_info(self, text, x, y):
        # Adjust position to avoid going off-screen
        offset_x = 20
        offset_y = -30
        text_x = x + offset_x
        text_y = y + offset_y

        # Create a text object to measure its size
        font = ("Arial", 12)
        text_id = self.canvas.create_text(0, 0, text=text, font=font, anchor='nw')
        bbox = self.canvas.bbox(text_id)
        self.canvas.delete(text_id)
        box_width = bbox[2] - bbox[0] + 10
        box_height = bbox[3] - bbox[1] + 10

        # Adjust position if it's too close to the edges
        if text_x + box_width > self.width:
            text_x = self.width - box_width
        if text_x < 0:
            text_x = 0
        if text_y - box_height < 0:
            text_y = y + 30

        # Draw rectangle and text
        # Clear previous overlays
        self.canvas.delete('color_info')
        self.canvas.create_rectangle(
            text_x, text_y - box_height, text_x + box_width, text_y,
            fill='black', outline='', tags='color_info'
        )
        self.canvas.create_text(
            text_x + 5, text_y - box_height + 5,
            text=text, fill='white', font=font, anchor='nw', tags='color_info'
        )

    def on_closing(self):
        self.stop_event.set()
        self.capture_thread.join()
        self.api_thread.join()
        self.cap.release()
        self.window.destroy()

# Run the application
if __name__ == "__main__":
    try:
        ColorDetectorApp(tk.Tk(), "Color Detector with Tracking")
    except Exception as e:
        print(f"An error occurred: {e}")
