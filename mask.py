import cv2
import numpy as np
import os


def extract_images(npy_file, output_folder):
    images = np.load(npy_file)

    os.makedirs(output_folder, exist_ok=True)

    for i, image in enumerate(images):
        # Construct filename
        filename = os.path.join(output_folder, f"image_{i}.png")
        
        # Save image: convert to uint8 if needed (for displayable format)
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
        
        cv2.imwrite(filename, image)

def create_mask(image_path):
    # Step 1: Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Edge detection with Canny or Sobel filter
    # For Canny
    #edges = cv2.Canny(blurred, 50, 150)
    # For Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.sqrt(sobelx**2 + sobely**2)
    edges = np.uint8(np.clip(edges, 0, 255))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw the contour on
    mask = np.zeros_like(image, dtype=np.uint8)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour on the mask
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(image, [main_contour], -1, (255, 0, 0), thickness=1)  # Green outline

        # Fit an ellipse to the contour for smoother edges
        if len(main_contour) >= 5:  # Fit requires at least 5 points
            ellipse = cv2.fitEllipse(main_contour)
            mask = np.zeros_like(image, dtype=np.uint8)  # Reset mask
            cv2.ellipse(mask, ellipse, 255, thickness=cv2.FILLED)

    # Save or display the mask
    cv2.imwrite('mask_test.png', mask)
    cv2.imwrite('countered_image.png', image)
    cv2.imshow('Original with Contours', image)
    cv2.imshow('Mask', mask)

    # Wait for a key press and close the image windows
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #
#
    ## Release all OpenCV resources and close the program
    #cv2.waitKey(1)  # This extra call is sometimes needed to ensure cleanup
    #cv2.destroyAllWindows()  # Just to make sure all windows are closed


    #return mask


def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            try:
                # Create mask for each image
                mask = create_mask(image_path)

                # Save the mask to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, mask)
                print(f"Mask saved for {filename} at {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
img = "./data/allSlices1/slice_001_pred.png"
mask = create_mask('./data/allSlices1/slice_001_pred.png')


# To visualize using OpenCV
#cv2.imshow('Mask', mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#exit()





#npy_file = "./logs/lamino_chip_4096rays/eval/epoch_01500/image_pred.npy"
#png_files = "./data/allSlices"
#output_folder = "data/masks"
##extract_images(npy_file, png_files)
#process_images(png_files, output_folder)
## Example usage