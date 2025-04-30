import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve

# 设置路径
rbc_tracks_csv_path = '/home/denga/code/output/rbc_tracks.csv'  # RBC 轨迹数据路径
z_range_path = '/home/denga/code/output/major_vessel_z_range.csv'  # 主要大血管Z范围路径
output_image_path = '/home/denga/code/output/simulation_sample_image.png'  # 输出图片路径

# PSF 参数
def generate_psf_kernel(psf_size_pixels, sigma_pixels):
    ax = np.arange(-psf_size_pixels // 2 + 1., psf_size_pixels // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_pixels**2))
    kernel /= np.sum(kernel)
    return kernel

# 计算 PSF
pixel_size_um = 7.33
psf_size_um = 21.0
psf_size_pixels = int(psf_size_um / pixel_size_um)
if psf_size_pixels % 2 == 0:
    psf_size_pixels += 1
sigma_um = psf_size_um / 2.355
sigma_pixels = sigma_um / pixel_size_um
psf_kernel = generate_psf_kernel(psf_size_pixels, sigma_pixels)

# 加载数据
z_range_df = pd.read_csv(z_range_path)
major_vessel_z_min = z_range_df['major_vessel_z_min_mm'].iloc[0]
major_vessel_z_max = z_range_df['major_vessel_z_max_mm'].iloc[0]

rbc_tracks_df = pd.read_csv(rbc_tracks_csv_path)

# 提取第一时间步数据
data = rbc_tracks_df[rbc_tracks_df['time'] == rbc_tracks_df['time'].min()]
data_filtered = data[(data['z'] >= major_vessel_z_min - 0.0002) & (data['z'] <= major_vessel_z_max + 0.0002)]

# 设置图像尺寸
camera_fov_mm = 7.39
fov_size_um = camera_fov_mm * 1000
num_pixels = int(fov_size_um / pixel_size_um)
x_min = rbc_tracks_df['x'].min() * 1000
x_max = rbc_tracks_df['x'].max() * 1000
y_min = rbc_tracks_df['y'].min() * 1000
y_max = rbc_tracks_df['y'].max() * 1000

# 坐标转换函数
def world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels_x, num_pixels_y):
    x_pixel = ((x_um - x_min) / (x_max - x_min) * (num_pixels_x - 1)).astype(int)
    y_pixel = ((y_um - y_min) / (y_max - y_min) * (num_pixels_y - 1)).astype(int)
    return x_pixel, y_pixel

# 生成图像
image = np.zeros((num_pixels, num_pixels), dtype=np.float32)
if not data_filtered.empty:
    x_um = data_filtered['x'].values * 1000
    y_um = data_filtered['y'].values * 1000
    x_pixel, y_pixel = world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels, num_pixels)
    valid_indices = (x_pixel >= 0) & (x_pixel < num_pixels) & (y_pixel >= 0) & (y_pixel < num_pixels)
    x_pixel = x_pixel[valid_indices]
    y_pixel = y_pixel[valid_indices]
    np.add.at(image, (y_pixel, x_pixel), 1.0)

# 应用 PSF
image = fftconvolve(image, psf_kernel, mode='same')
image = image / image.max() if image.max() > 0 else image

# 添加噪声
noise = np.random.normal(loc=0.0, scale=0.05, size=image.shape)
image = image + noise
image = (image / image.max() * 255).astype(np.uint8)

# 保存图像
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Simulated RBC Image with Noise and PSF')
plt.savefig(output_image_path, bbox_inches='tight')
plt.show()

print(f"图像已保存到 {output_image_path}")
