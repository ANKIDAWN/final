import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve
from tqdm import tqdm
import imageio


# 设置输入和输出路径
rbc_tracks_csv_path = '/home/denga/code/output/rbc_tracks.csv'  # RBC 轨迹数据路径
z_range_path = '/home/denga/code/output/major_vessel_z_range.csv'  # 主要大血管Z范围路径
output_directory = '/home/denga/code/persoff/data/network/output/imaging_simulation'  # 图像输出目录
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
psf_size_um = 21.0  # PSF 尺寸，μm，调整为21 μm以覆盖3像素
psf_size_pixels = int(psf_size_um / pixel_size_um)
if psf_size_pixels % 2 == 0:
    psf_size_pixels += 1  # 确保 PSF 尺寸为奇数
sigma_um = psf_size_um / 2.355  # 将 FWHM 转换为标准差
sigma_pixels = sigma_um / pixel_size_um

psf_kernel = generate_psf_kernel(psf_size_pixels, sigma_pixels)
print(f"PSF 内核尺寸：{psf_kernel.shape}")


print("\n=== 读取主要大血管的Z坐标范围 ===")
try:
    z_range_df = pd.read_csv(z_range_path)
    major_vessel_z_min = z_range_df['major_vessel_z_min_mm'].iloc[0]
    major_vessel_z_max = z_range_df['major_vessel_z_max_mm'].iloc[0]
    print(f"主要大血管的 Z 坐标范围：{major_vessel_z_min} - {major_vessel_z_max} mm")
except FileNotFoundError:
    print(f"未找到文件：{z_range_path}。请确保文件路径正确。")
    exit(1)
except KeyError:
    print(f"文件 {z_range_path} 缺少必要的列。")
    exit(1)

print("\n=== 读取 RBC 轨迹数据 ===")
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

print("\n=== 模拟显微镜成像过程 ===")

# 初始化视频写入器
output_video_path = os.path.join(output_directory, 'imaging_simulation.mp4')  # 或者 'imaging_simulation.gif'
try:
    writer = imageio.get_writer(output_video_path, fps=camera_frame_rate, codec='libx264', quality=8)
except ValueError as e:
    print(f"视频写入器初始化失败：{e}")
    print("请确保已安装必要的编解码器插件，例如 FFmpeg。")
    print("尝试运行：pip install imageio[ffmpeg]")
    exit(1)

# 计算坐标范围
x_values_um = rbc_tracks_df['x'].values * 1000  # 假设原始单位为 mm，转换为 μm
y_values_um = rbc_tracks_df['y'].values * 1000

x_min = x_values_um.min()
x_max = x_values_um.max()
y_min = y_values_um.min()
y_max = y_values_um.max()

# 坐标转换函数
def world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels_x, num_pixels_y):
    """
    将世界坐标（μm）转换为像素坐标
    :param x_um: X 坐标，μm
    :param y_um: Y 坐标，μm
    :param x_min: X 最小值，μm
    :param x_max: X 最大值，μm
    :param y_min: Y 最小值，μm
    :param y_max: Y 最大值，μm
    :param num_pixels_x: X 方向像素数
    :param num_pixels_y: Y 方向像素数
    :return: 像素坐标（整数）
    """
    x_pixel = ((x_um - x_min) / (x_max - x_min) * (num_pixels_x - 1)).astype(int)
    y_pixel = ((y_um - y_min) / (y_max - y_min) * (num_pixels_y - 1)).astype(int)
    return x_pixel, y_pixel

# 设置图像的像素数量
num_pixels_x = num_pixels
num_pixels_y = num_pixels
fov_size_um_x = x_max - x_min
fov_size_um_y = y_max - y_min

print(f"图像分辨率：{num_pixels_x} x {num_pixels_y} 像素")
print(f"视场大小：{fov_size_um_x} μm x {fov_size_um_y} μm")

# 遍历每个采样时间点并写入视频
for frame_idx, t in enumerate(tqdm(sampled_times, desc='生成图像序列', ascii=True, ncols=80)):
    # 提取当前时间的 RBC 数据
    data = rbc_tracks_df[rbc_tracks_df['time'] == t]

    # 过滤在指定深度范围内的 RBC（±200 nm）
    data_filtered = data[
        (data['z'] >= major_vessel_z_min - 0.0002) &  # 200 nm = 0.0002 mm
        (data['z'] <= major_vessel_z_max + 0.0002)
    ]

    # 初始化图像
    image = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float32)

    if not data_filtered.empty:
        # 将RBC位置转换为图像坐标
        x_um = data_filtered['x'].values * 1000  # 转换为 μm
        y_um = data_filtered['y'].values * 1000

        x_pixel, y_pixel = world_to_pixel(x_um, y_um, x_min, x_max, y_min, y_max, num_pixels_x, num_pixels_y)

        # 过滤出在图像范围内的像素
        valid_indices = (x_pixel >= 0) & (x_pixel < num_pixels_x) & (y_pixel >= 0) & (y_pixel < num_pixels_y)
        x_pixel = x_pixel[valid_indices]
        y_pixel = y_pixel[valid_indices]

        # 使用numpy的索引加速累加
        np.add.at(image, (y_pixel, x_pixel), 1.0)

    # 将图像与 PSF 进行卷积
    image = fftconvolve(image, psf_kernel, mode='same')

    # 归一化图像
    if image.max() > 0:
        image /= image.max()

    # 将图像转换为uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # 写入视频
    try:
        writer.append_data(image_uint8)
    except Exception as e:
        print(f"写入帧 {frame_idx} 时发生错误：{e}")
        writer.close()
        exit(1)

    # 释放内存
    del image, image_uint8, data, data_filtered

    # 每隔一定帧数打印一次进度（可选）
    if frame_idx % 1000 == 0 and frame_idx > 0:
        progress_percentage = (frame_idx / num_frames) * 100
        print(f"生成进度：{progress_percentage:.1f}%，当前帧：{frame_idx}/{num_frames}")

# 关闭视频写入器
writer.close()
print(f"模拟视频已保存到 {output_video_path}")

