import os
import nibabel as nib
import numpy as np
from PIL import Image

def convert_nii_to_png(nii_path, output_path):
    # Load the NIfTI file
    nifti_file = nib.load(nii_path)
    image_data = nifti_file.get_fdata()

    # Select the middle slice
    middle_slice_index = image_data.shape[2] // 2
    slice_2d = image_data[:, :, middle_slice_index]

    # Normalize the slice
    slice_normalized = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
    image = Image.fromarray((slice_normalized * 255).astype(np.uint8))

    # Resize the image to 256x256
    image_resized = image.resize((256, 256))

    # Convert to grayscale
    image_gray = image_resized.convert('L')

    # Save the resized grayscale image
    image_gray.save(output_path)

def stitch_images(img_path1, img_path2, output_path):
    # Load the images
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # Create a new image with double the width
    dst = Image.new('L', (img1.width + img2.width, img1.height))

    # Paste the two images side-by-side
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))

    # Save the stitched image
    dst.save(output_path)

def process_brain_data(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing {folder}...")

            # Define file paths
            ct_path = os.path.join(folder_path, 'ct.nii.gz')
            mr_path = os.path.join(folder_path, 'mr.nii.gz')
            ct_png_path = os.path.join(folder_path, 'ct_temp.png')
            mr_png_path = os.path.join(folder_path, 'mr_temp.png')

            # Convert and save images temporarily
            convert_nii_to_png(ct_path, ct_png_path)
            convert_nii_to_png(mr_path, mr_png_path)

            # Stitch images and save in output folder
            stitched_path = os.path.join(output_dir, f"{folder}_stitched.png")
            stitch_images(ct_png_path, mr_png_path, stitched_path)

            # Clean up temporary files
            os.remove(ct_png_path)
            os.remove(mr_png_path)
            
# Main execution
root_directory = '/home/sushmitha/Downloads/output_brain/pelvis'  # Change this to your actual directory path
output_directory = '/home/sushmitha/Downloads/output_brain'  # Change this to your desired output directory
process_brain_data(root_directory, output_directory)
