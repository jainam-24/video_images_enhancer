# #content aware padding

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = 'C:/Users/JAINAM/Downloads/d360/image_video_enhancer/bA1.jpg'
# img = cv2.imread(image_path)

# # Apply constant padding (for simplicity) which will be filled in later
# top, bottom, left, right = 100, 100, 100, 100
# padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# # Create a mask where the padded area will be inpainted (1 for areas to be filled)
# mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.uint8)
# mask[:top, :] = 1  # Top padding
# mask[-bottom:, :] = 1  # Bottom padding
# mask[:, :left] = 1  # Left padding
# mask[:, -right:] = 1  # Right padding

# # cv2.INPAINT_TELEA is a method for content-aware 
# # inpainting that fills the padded regions based on the surrounding content of the original image.
# # Apply inpainting to fill the padded areas based on the original image
# inpainted_img = cv2.inpaint(padded_img, mask, 10, cv2.INPAINT_TELEA)

# # Save the inpainted image
# cv2.imwrite('content_aware_padded_image.jpg', inpainted_img)

# # Display the inpainted image
# plt.imshow(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

import cv2
import numpy as np

# Input and output video paths
input_video_path = 'imaged_100-A-SD-A60_video.mp4'
output_video_path = 'output_video_with_padding.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get the video properties (frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the padding (you can adjust these values)
top, bottom, left, right = 100, 100, 100, 100

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                      (frame_width + left + right, frame_height + top + bottom))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Apply constant padding to each frame
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Create a mask for the padded areas
    mask = np.zeros((padded_frame.shape[0], padded_frame.shape[1]), dtype=np.uint8)
    mask[:top, :] = 1  # Top padding
    mask[-bottom:, :] = 1  # Bottom padding
    mask[:, :left] = 1  # Left padding
    mask[:, -right:] = 1  # Right padding
    
    # Inpaint the padded areas
    inpainted_frame = cv2.inpaint(padded_frame, mask, 10, cv2.INPAINT_TELEA)
    
    # Write the inpainted frame to the output video
    out.write(inpainted_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
