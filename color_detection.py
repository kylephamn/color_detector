import cv2
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk

# API endpoint
API_URL = "https://www.thecolorapi.com/id"

# Function to get color name from API
def get_color_name(r, g, b):
    params = {'rgb': f'rgb({r},{g},{b})'}
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()
        color_name = data.get('name', {}).get('value', 'Unknown')
        return color_name
    except Exception as e:
        print(f"API request failed: {e}")
        return "Unknown"

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a canvas to display the video feed
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.mouse_click)

        self.color_info = ''
        self.delay = 15  # milliseconds
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_no_overlay = frame.copy()
            # Convert the image to PhotoImage
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

            # Display the image
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Display color info if available
            if self.color_info:
                text, x, y = self.color_info
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

                # Draw rectangle
                rect_id = self.canvas.create_rectangle(text_x, text_y - box_height, text_x + box_width, text_y, fill='black', outline='')
                # Draw text
                text_id = self.canvas.create_text(text_x + 5, text_y - box_height + 5, text=text, fill='white', font=font, anchor='nw')

        self.window.after(self.delay, self.update)

    def mouse_click(self, event):
        x = event.x
        y = event.y

        # Ensure coordinates are within the frame
        if x >= self.width or y >= self.height:
            return

        # Get pixel value from the frame
        bgr_pixel = self.frame_no_overlay[int(y), int(x)]
        b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])

        # Get color name from API
        color_name = get_color_name(r, g, b)
        self.color_info = (f'{color_name} (R:{r}, G:{g}, B:{b})', x, y)

        print(self.color_info[0])  # Optional: Print to console for debugging

    def on_closing(self):
        self.cap.release()
        self.window.destroy()

# Run the application
App(tk.Tk(), "Color Detector")
