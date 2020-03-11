import math
import numpy as np
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

def psnr_fn(image, reconstructed_image):
    rmse = math.sqrt(np.mean((image - reconstructed_image) ** 2))
    max_value = image.max()
    return 20 * math.log10(max_value / rmse)

def ssim_fn(image, reconstructed_image):
    image = image.transpose(1,2,0)
    reconstructed_image = reconstructed_image.transpose(1,2,0)
    return ssim(image, reconstructed_image, multichannel=True)