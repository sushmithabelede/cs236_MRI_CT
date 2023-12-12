#To preprocess, we drew inspiration from https://github.com/AntoninDuval/CycleGAN

import cv2
import matplotlib.pyplot as plt
import PIL
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

CT_DATA_FOLDER = '/Users/maay/Desktop/ICME/Q4/CS236/project/CycleGAN-master/notebooks/data/CT_slices'
CT_DATA_TGT_FOLDER = '/Users/maay/Desktop/ICME/Q4/CS236/project/CycleGAN-master/notebooks/data/CT_slices_registered_clean'


def remove_artefact(img_ref, img, threshold = 200):
  """
  remove artefact on CT image
  """
  img_shape = np.array(img_ref.shape)
  mask = binary_erosion((img_ref > threshold).astype(np.uint8), iterations=4)
  mask = binary_dilation(mask, iterations=35)
  mask = binary_fill_holes(mask)

  result = cv2.bitwise_and(img, img, mask = mask.astype(np.uint8))
  return result


ref = cv2.imread(f'{CT_DATA_FOLDER}/slide_0.jpg', cv2.IMREAD_GRAYSCALE)

for file in tqdm([f for f in listdir(CT_DATA_FOLDER) if isfile(join(CT_DATA_FOLDER, f))]):
    current = cv2.imread(f'{CT_DATA_FOLDER}/{file}', cv2.IMREAD_GRAYSCALE)
    r = remove_artefact(ref, current)
    j = PIL.Image.fromarray(r)
    j.save(f'{CT_DATA_TGT_FOLDER}/{file}')



# for path in tqdm(image_paths_T2):
#     img = np.asarray(Image.open(path).convert('L'))
#     img_nii = nib.Nifti1Image(img, affine=np.eye(4))
#     new_path = join(split(path)[0]+'_nii', split(path)[1])[:-3]+'nii'
#     nib.save(img_nii, new_path)
# dir_path_T2 = '/Users/maay/Desktop/ICME/Q4/CS236/project/CycleGAN-master/notebooks/data/T2_slices'
# image_paths_T2 = sorted([join(dir_path_T2, file) for file in listdir(dir_path_T2) if isfile(join(dir_path_T2, file))])
