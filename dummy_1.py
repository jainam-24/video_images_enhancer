# Reflection padding with selective boundary blurring
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = 'bA1.jpg'
img = cv2.imread(image_path)
# img_pil = Image.open("C:/Users/JAINAM/Downloads/d360/image_video_enhancer/input_images/bA1.jpg")

# Define padding sizes
reflection_padding1 = 50  # First round of reflection padding
reflection_padding2 = 50  # Second round of reflection padding

# Step 1: Apply the first reflection padding (50 pixels on all sides)
padded_img = cv2.copyMakeBorder(img, reflection_padding1, reflection_padding1, reflection_padding1, reflection_padding1, cv2.BORDER_REFLECT)

# Step 2: Apply a second reflection padding (20 pixels on all sides)
padded_img = cv2.copyMakeBorder(padded_img, reflection_padding2, reflection_padding2, reflection_padding2, reflection_padding2, cv2.BORDER_REFLECT)

# Create a mask to define the boundary region where we want to blur (e.g., 10 pixels from the boundary)
blur_size = reflection_padding1+reflection_padding2-20  # Width of the boundary to blur
height, width, _ = padded_img.shape

# Define regions for blurring: top, bottom, left, right edges
mask = np.zeros_like(padded_img)

# Top boundary
mask[:blur_size, :] = 1
# Bottom boundary
mask[-blur_size:, :] = 1
# Left boundary
mask[:, :blur_size] = 1
# Right boundary
mask[:, -blur_size:] = 1

# Step 3: Apply Gaussian blur to the regions defined by the mask
blurred_region = cv2.GaussianBlur(padded_img, (7, 7), 2)

# Step 4: Combine the original image with the blurred regions using the mask
padded_img = np.where(mask == 1, blurred_region, padded_img)

# Display the final padded image
plt.imshow(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide the axis    

# Step 5: Resize the padded image to 635x635 pixels
padded_img = cv2.resize(padded_img, (635, 635))

# Optionally save the image
cv2.imwrite('padded_image_now_blurred.jpg', padded_img)
