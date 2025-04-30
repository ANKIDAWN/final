import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve
from tqdm import tqdm
import imageio

# ======================== 可视化代码开始 ========================

# 设置输入和输出路径
rbc_tracks_csv_path = 'output/rbc_tracks.csv'  # 请根据实际路径调整
output_directory = 'output/imaging_simulation'  # 图像输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 摄像机和显微镜参数
camera_frame_rate = 400  # 帧率，fps
camera_fov_mm = 7.39     # 视场（正方形边长），mm
pixel_size_um = 7.33     # 像素尺寸，μm

# 计算像素数量
fov_size_um = camera_fov_mm * 1000  # 转换为 μm
num_pixels = int(fov_size_um / pixel_size_um)
print(f"图像分辨率：{num_pixels} x {num_pixels} 像素")

# PSF 参数
# 您可以使用理论 PSF 或从原始数据中提取实际 PSF
# 这里我们使用高斯函数模拟 PSF

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
psf_size_um = 1.0  # 假设 PSF 的尺寸为 1 μm（您可以根据需要调整）
psf_size_pixels = int(psf_size_um / pixel_size_um)
if psf_size_pixels % 2 == 0:
    psf_size_pixels += 1  # 确保 PSF 尺寸为奇数
sigma_um = psf_size_um / 2.355  # 将 FWHM 转换为标准差
sigma_pixels = sigma_um / pixel_size_um

psf_kernel = generate_psf_kernel(psf_size_pixels, sigma_pixels)
print(f"PSF 内核尺寸：{psf_kernel.shape}")

# ======================== 读取 RBC 轨迹数据 ========================

print("=== 读取 RBC 轨迹数据 ===")
try:
    rbc_tracks_df = pd.read_csv(rbc_tracks_csv_path)
    print(f"成功读取 {len(rbc_tracks_df)} 条轨迹数据。")
except FileNotFoundError:
    print(f"未找到文件：{rbc_tracks_csv_path}。请确保文件路径正确。")
    exit(1)

# 获取所有时间步
time_steps = sorted(rbc_tracks_df['time'].unique())
print(f"总共有 {len(time_steps)} 个时间步。")

# 根据摄像机帧率进行采样
simulation_time = time_steps[-1] - time_steps[0]
num_frames = int(simulation_time * camera_frame_rate) + 1
sampled_times = np.linspace(time_steps[0], time_steps[-1], num_frames)
print(f"按照 {camera_frame_rate} fps，总共生成 {num_frames} 帧图像。")

# ======================== 模拟显微镜成像过程 ========================

print("\n=== 模拟显微镜成像过程 ===")

# 初始化图像序列列表
image_sequence = []

# 将坐标转换为像素坐标
def world_to_pixel(x_um, y_um, fov_size_um, num_pixels):
    """
    将世界坐标（μm）转换为像素坐标
    :param x_um: X 坐标，μm
    :param y_um: Y 坐标，μm
    :param fov_size_um: 视场大小，μm
    :param num_pixels: 图像像素数
    :return: 像素坐标（整数）
    """
    x_pixel = ((x_um + fov_size_um / 2) / fov_size_um) * num_pixels
    y_pixel = ((y_um + fov_size_um / 2) / fov_size_um) * num_pixels
    return int(x_pixel), int(y_pixel)

# 遍历每个采样时间点
for frame_idx, t in tqdm(enumerate(sampled_times), total=num_frames, desc='生成图像序列', ascii=True, ncols=80):
    # 提取当前时间的 RBC 数据
    data = rbc_tracks_df[rbc_tracks_df['time'] == t]
    # 初始化图像
    image = np.zeros((num_pixels, num_pixels), dtype=np.float32)
    # 对于每个 RBC，计算其在图像中的位置并添加到图像中
    for idx, row in data.iterrows():
        x_um = row['x'] * 1e3  # 假设坐标单位为 mm，转换为 μm
        y_um = row['y'] * 1e3
        x_pixel, y_pixel = world_to_pixel(x_um, y_um, fov_size_um, num_pixels)
        # 检查坐标是否在图像范围内
        if 0 <= x_pixel < num_pixels and 0 <= y_pixel < num_pixels:
            # 在图像中添加一个点
            image[y_pixel, x_pixel] += 1.0  # 累积强度，可根据需要调整
    # 将图像与 PSF 进行卷积
    image = fftconvolve(image, psf_kernel, mode='same')
    # 归一化图像
    image /= image.max()
    # 添加到图像序列
    image_sequence.append(image)
    # 可选：保存每一帧图像
    # plt.imsave(os.path.join(output_directory, f"frame_{frame_idx:05d}.png"), image, cmap='gray')

# ======================== 生成视频或 GIF 动画 ========================

print("\n=== 生成模拟视频 ===")

# 将图像序列保存为 GIF 或 MP4
output_video_path = os.path.join(output_directory, 'imaging_simulation.mp4')  # 或者 'imaging_simulation.gif'

# 将图像序列转换为 uint8 类型
image_sequence_uint8 = [(img * 255).astype(np.uint8) for img in image_sequence]

# 保存为 MP4 视频
imageio.mimsave(output_video_path, image_sequence_uint8, fps=camera_frame_rate)

print(f"模拟视频已保存到 {output_video_path}")

# ======================== 可视化代码结束 ========================
