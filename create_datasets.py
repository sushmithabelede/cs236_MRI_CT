import os
import shutil
import random

# Set the paths for the source and destination folders

# source_folder = '/home/mattias/CS236/data/T2_slices'
# train_folder = '/home/mattias/PyTorch-CycleGAN/datasets/medical/train/A'
# test_folder = '/home/mattias/PyTorch-CycleGAN/datasets/medical/test/A'

source_folder = '/home/mattias/CS236/data/CT_slices_registered_clean'
train_folder = '/home/mattias/PyTorch-CycleGAN/datasets/medical/train/B'
test_folder = '/home/mattias/PyTorch-CycleGAN/datasets/medical/test/B'

# Create train and test directories if they don't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Get all file names in the source folder
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Shuffle files and split into train and test sets (70% train, 30% test)
random.shuffle(files)
split_index = int(0.7 * len(files))
train_files = files[:split_index]
test_files = files[split_index:]

# Copy files to the respective train and test folders
for file in train_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

for file in test_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

print(f"Copied {len(train_files)} files to {train_folder} and {len(test_files)} files to {test_folder}.")
