# # Reflection padding 
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Load the image
# image_path = 'bA1.jpg'
# img = cv2.imread(image_path)
# img_pil=Image.open("C:/Users/JAINAM/Downloads/d360/image_video_enhancer/input_images/bA1.jpg")
# # Apply reflection padding (50 pixels on all sides as an example)
# top, bottom, left, right = 60, 60, 60, 60
# padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

# # Display the padded image
# plt.imshow(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Hide the axis

# padded_img=cv2.resize(padded_img,(635,635))
# # Optionally save the image
# cv2.imwrite('padded_image_now.jpg', padded_img)

import cv2
import numpy as np

# Load the video
video_path = 'imaged_100-A-SD-A60_video.mp4'  # Specify the input video path
cap = cv2.VideoCapture(video_path)

# Get the video details (frame width, height, fps, etc.)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the padding size
top, bottom, left, right = 60, 60, 60, 60

# Define the output video codec and create a VideoWriter object
output_size = (635, 635)  # New size after padding and resizing
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, output_size)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is returned, exit the loop

    # Apply reflection padding to the frame
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REFLECT)
    
    # Resize the padded frame to the desired size (635x635 in this case)
    resized_frame = cv2.resize(padded_frame, output_size)

    # Write the processed frame to the output video
    out.write(resized_frame)

# Release video capture and writer objects
cap.release()
out.release()

# Optionally, close any OpenCV windows (if you are displaying the frames in between)
cv2.destroyAllWindows()
