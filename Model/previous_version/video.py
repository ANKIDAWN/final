import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve
from tqdm import tqdm
import imageio

# ========================== 设置输入和输出路径 ==========================
rbc_tracks_csv_path = '/home/denga/code/output/rbc_tracks.csv'  # RBC 轨迹数据路径
z_range_path = '/home/denga/code/output/major_vessel_z_range.csv'  # 主要大血管Z范围路径
output_directory = '/home/denga/code/persoff/data/network/output/imaging_simulation'  # 图像输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 摄像机和显微镜参数
camera_frame_rate = 400  # 帧率，fps
camera_fov_mm = 7.39     # 视场（正方形边长），mm
pixel_size_um = 7.33     # 像素尺寸，μm
batch_size = 1000        # 每个视频包含的帧数

# 动态生成背景噪声

def generate_dynamic_noise(shape, intensity=0.05):
    """
    生成动态噪声
    :param shape: 图像形状
    :param intensity: 噪声强度
    :return: 动态噪声
    """
    noise = np.random.normal(loc=0.0, scale=intensity, size=shape)
    return np.clip(noise, 0, 1)  # 限制到 [0, 1]

# PSF 参数
# 使用高斯函数模拟 PSF

def generate_psf_kernel(psf_size_pixels, sigma_pixels):
    """
    生成基于高斯函数的 PSF 内核
    :param psf_size_pixels: PSF 的尺寸（像素），应为奇数
    :param sigma_pixels: 高斯函数的标准差（像素）
    :return: 归一化的 PSF 内核
    """
    ax = np.arange(-psf_size_pixels // 2 + 1., psf_size_pixels // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_pixels**2))
    kernel /= np.sum(kernel)
    return kernel

# 计算 PSF 内核
psf_size_um = 21.0  # PSF 尺寸，μm
psf_size_pixels = int(psf_size_um / pixel_size_um)
if psf_size_pixels % 2 == 0:
    psf_size_pixels += 1  # 确保 PSF 尺寸为奇数
sigma_um = psf_size_um / 2.355  # 将 FWHM 转换为标准差
sigma_pixels = sigma_um / pixel_size_um

psf_kernel = generate_psf_kernel(psf_size_pixels, sigma_pixels)
print(f"PSF 内核尺寸：{psf_kernel.shape}")

# ========================== 加载数据 ==========================
# 加载主要大血管的 Z 坐标范围
print("\n=== 读取主要大血管的Z坐标范围 ===")
z_range_df = pd.read_csv(z_range_path)
major_vessel_z_min = z_range_df['major_vessel_z_min_mm'].iloc[0]
major_vessel_z_max = z_range_df['major_vessel_z_max_mm'].iloc[0]

# 加载 RBC 轨迹数据
print("\n=== 读取 RBC 轨迹数据 ===")
rbc_tracks_df = pd.read_csv(rbc_tracks_csv_path)

# 获取所有时间步
time_steps = sorted(rbc_tracks_df['time'].unique())
simulation_time = time_steps[-1] - time_steps[0]
num_frames = int(simulation_time * camera_frame_rate) + 1
sampled_times = np.linspace(time_steps[0], time_steps[-1], num_frames)
print(f"按照 {camera_frame_rate} fps，总共生成 {num_frames} 帧图像。")

# 坐标转换函数

def world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels_x, num_pixels_y):
    x_pixel = ((x_um - x_min) / (x_max - x_min) * (num_pixels_x - 1)).astype(int)
    y_pixel = ((y_um - y_min) / (y_max - y_min) * (num_pixels_y - 1)).astype(int)
    return x_pixel, y_pixel

# 设置图像的像素数量
fov_size_um = camera_fov_mm * 1000  # 转换为 μm
num_pixels = int(fov_size_um / pixel_size_um)
num_pixels_x = num_pixels
num_pixels_y = num_pixels
x_min = rbc_tracks_df['x'].min() * 1000  # 转换为 μm
x_max = rbc_tracks_df['x'].max() * 1000
y_min = rbc_tracks_df['y'].min() * 1000
y_max = rbc_tracks_df['y'].max() * 1000

# ========================== 分批生成视频 ==========================
batch_count = (num_frames + batch_size - 1) // batch_size  # 向上取整
for batch_idx in range(batch_count):
    start_frame = batch_idx * batch_size
    end_frame = min(start_frame + batch_size, num_frames)
    output_video_path = os.path.join(output_directory, f'imaging_simulation_batch_{batch_idx:03d}.mp4')

    with imageio.get_writer(output_video_path, fps=camera_frame_rate, codec='libx264', quality=8) as writer:
        for frame_idx, t in enumerate(tqdm(sampled_times[start_frame:end_frame], desc=f'Batch {batch_idx+1}/{batch_count}')):
            data = rbc_tracks_df[rbc_tracks_df['time'] == t]
            data_filtered = data[(data['z'] >= major_vessel_z_min - 0.0002) & (data['z'] <= major_vessel_z_max + 0.0002)]

            image = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float32)
            if not data_filtered.empty:
                x_um = data_filtered['x'].values * 1000  # 转换为 μm
                y_um = data_filtered['y'].values * 1000
                x_pixel, y_pixel = world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels_x, num_pixels_y)
                valid_indices = (x_pixel >= 0) & (x_pixel < num_pixels_x) & (y_pixel >= 0) & (y_pixel < num_pixels_y)
                x_pixel = x_pixel[valid_indices]
                y_pixel = y_pixel[valid_indices]
                np.add.at(image, (y_pixel, x_pixel), 1.0)

            image = fftconvolve(image, psf_kernel, mode='same')
            image = image / image.max() if image.max() > 0 else image

            noise = generate_dynamic_noise(image.shape, intensity=0.05)
            image = image + noise  # 加入背景噪声
            image = (image / image.max() * 255).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)

            writer.append_data(image)

    print(f"批次 {batch_idx+1}/{batch_count} 的视频已保存到 {output_video_path}")

print("\n=== 所有批次视频生成完毕 ===")
