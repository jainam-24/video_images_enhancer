import cv2
import os
import numpy as np
from glob import glob
import re


def natural_key(string):
    """
    A helper function for natural sorting that splits strings into numeric and non-numeric parts.
    This allows the sorting function to compare numbers in the filenames naturally.
    """
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', string)]

def process_images(image_folder, rgb_offset, output_folder):
    """
    Process 256 images by applying RGB enhancement.
    
    Parameters:
    image_folder (str): Path to the folder containing 256 images.
    rgb_offset (tuple): A tuple of (R, G, B) values to be added to each pixel.
    output_folder (str): Path to save the enhanced images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image paths (assuming common image extensions)
    image_paths = glob(os.path.join(image_folder, '*.[jp][pn]g'))
    
    # Sort images by natural order (i.e., image_1.jpg, image_2.jpg, ..., image_10.jpg)
    image_paths = sorted(image_paths, key=natural_key)[:256]
    
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        enhanced_image = enhance_image(image, rgb_offset)
        
        # Save the enhanced image
        output_path = os.path.join(output_folder, f"enhanced_{idx+1}.png")
        cv2.imwrite(output_path, enhanced_image)
        print(f"Saved enhanced image: {output_path}")

def enhance_image(image, rgb_offset):
    """
    Enhance the image by adding an RGB offset to each pixel.
    
    Parameters:
    image (np.ndarray): The input image in BGR format (as returned by OpenCV).
    rgb_offset (tuple): A tuple of (R, G, B) values to be added to each pixel.
    
    Returns:
    np.ndarray: The enhanced image with RGB values modified.
    """
    # Convert image from BGR to RGB (since OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create an array of the RGB offset and add it to the image
    offset = np.array(rgb_offset).reshape((1, 1, 3))
    enhanced_image = np.clip(image_rgb + offset, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for OpenCV to handle the output correctly
    return cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)


def process_video(video_path, rgb_offset, output_video_path):
    """
    Process a video by applying RGB enhancement to each frame.
    
    Parameters:
    video_path (str): Path to the input video file.
    rgb_offset (tuple): A tuple of (R, G, B) values to be added to each pixel.
    output_video_path (str): Path to save the enhanced video.
    """
    # Check if the video file can be opened
    if not os.path.exists(video_path):
        print(f"Error: Input video file {video_path} does not exist.")
        return
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not output_video.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return
    
    # Process each frame of the video
    frame_count = 0
    while True:
        ret, frame = video.read()
        
        if not ret:
            # If no frame is returned, we've either reached the end of the video or there's an issue
            print(f"Finished processing video. Total frames processed: {frame_count}/{total_frames}")
            break
        
        # Enhance the frame
        enhanced_frame = enhance_image(frame, rgb_offset)
        
        # Write the enhanced frame to the output video
        output_video.write(enhanced_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames out of {total_frames}")
    
    # Release video capture and writer objects
    video.release()
    output_video.release()
    
    print(f"Video processing complete. Saved to {output_video_path}")

def main(input_type, input_path, rgb_offset, output_path):
    """
    Main function to handle either image or video processing.
    
    Parameters:
    input_type (str): Either 'images' or 'video'.
    input_path (str): Path to the folder of images or the video file.
    rgb_offset (tuple): A tuple of (R, G, B) values to be added to each pixel.
    output_path (str): Path to save the processed images or video.
    """
    if input_type == 'images':
        print("Processing images...")
        process_images(input_path, rgb_offset, output_path)
    elif input_type == 'video':
        print("Processing video...")
        process_video(input_path, rgb_offset, output_path)
    else:
        print("Invalid input type. Choose either 'images' or 'video'.")

if __name__ == "__main__":
    
    user_input=input("Enter 1 for images Or Enter 2 for videos : ")
    
    if user_input=="1":
        
        input_type = 'images' 
        
        # Path to input (folder for images or video file)
        input_path = './input_images'  # or './input_video.mp4'
        
        # RGB offset to add (R, G, B)
        rgb_offset = (20, 20, 20)
        
        # Output folder or file
        output_path = './output_images'  # or './output_video.mp4'
        
        main(input_type, input_path, rgb_offset, output_path)
        print("process completed, check images in output images folder ")
    
    elif user_input=="2":
        
        input_type = 'video' 
        # Input video path (ensure it is a valid video file, e.g., .mp4 or .avi)
        input_path = 'input_video/diamond.mp4'
        
        # RGB offset to add (R, G, B)
        rgb_offset = (-10, -10, -10)
        
        # Output path for the processed video (ensure it has the correct extension)
        output_path = './output_video/output_video.mp4'
        
        main(input_type, input_path, rgb_offset, output_path)
        print("process completed, check video in output video folder ")
        