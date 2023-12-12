
################## FOR PSNR, FID and SSIM ##################### import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
import os

def get_inception_features(model, img):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
    return pred.squeeze().cpu().numpy()

def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def process_folder(folder_path):
    real_images = []
    fake_images = []

    # Load Inception model for FID
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    # Collect real and fake images
    for file in os.listdir(folder_path):
        if file.endswith("_real_B.png"):
            real_images.append(file)
        elif file.endswith("_fake_B.png"):
            fake_images.append(file)

    real_features = []
    fake_features = []
    psnr_values = []
    ssim_values = []

    # Extract features from images and calculate PSNR and SSIM
    for real_img, fake_img in zip(real_images, fake_images):
        real_path = os.path.join(folder_path, real_img)
        fake_path = os.path.join(folder_path, fake_img)

        real_image = Image.open(real_path).convert('RGB')
        fake_image = Image.open(fake_path).convert('RGB')

        real_features.append(get_inception_features(inception_model, real_image))
        fake_features.append(get_inception_features(inception_model, fake_image))

        real_np = np.array(real_image)
        fake_np = np.array(fake_image)

        psnr_values.append(calculate_psnr(real_np, fake_np))
        ssim_values.append(ssim(real_np, fake_np, multichannel=True))

    # Convert lists to numpy arrays
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)

    # Calculate FID
    fid_value = calculate_fid(real_features, fake_features)

    # Calculate average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return fid_value, avg_psnr, avg_ssim

# Path to the folder containing images
folder_path = '/home/sushmitha/Downloads/test_mri_ct'
fid_value, avg_psnr, avg_ssim = process_folder(folder_path)
print(f"FID: {fid_value}, Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")