#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐帧输出 PNG，同时深度单位保持 米，一并加上 200 nm 的 axial PSF 衰减。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import imageio
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks",    type=Path,
        default=Path(__file__).parent.parent/"output"/"rbc_tracks.csv")
    p.add_argument("--zrange",    type=Path,
        default=Path(__file__).parent.parent/"output"/"major_vessel_z_range.csv")
    p.add_argument("--out",       type=Path,
        default=Path(__file__).parent.parent/"output"/"imaging_frames")
    p.add_argument("--fps",       type=int,   default=400)
    p.add_argument("--fov-mm",    type=float, default=7.39)
    p.add_argument("--px-size-um",type=float, default=7.33)
    p.add_argument("--psf-um",    type=float, default=21.0)
    p.add_argument("--sigma-z-nm",type=float, default=200.0,
                   help="轴向 PSF σ, 单位 nm")
    p.add_argument("--noise",     type=float, default=0.05)
    return p.parse_args()

def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1) 轨迹
    df = pd.read_csv(args.tracks)
    df = df.drop_duplicates(subset=["time","rbc_id"], keep="first")
    # x,y 转 μm
    df["x_um"] = df["x"] * 1e6
    df["y_um"] = df["y"] * 1e6
    # z 还是 m
    df["z_m"]  = df["z"]

    # 2) 读大血管深度范围（CSV 中列名 mm，但值是 m）
    zr = pd.read_csv(args.zrange)
    # 如果它真的是 mm → m，请乘 1e-3；如果它已经是 m，就直接用：
    z_min = zr.major_vessel_z_min_mm.iloc[0] * 1e-3
    z_max = zr.major_vessel_z_max_mm.iloc[0] * 1e-3
    # 焦面中心
    z_focus = 0.5*(z_min + z_max)

    # 3) 时间对齐
    sim_times = np.sort(df["time"].unique())
    t0, t1    = sim_times[0], sim_times[-1]
    n_frames  = int((t1-t0)*args.fps) + 1
    cam_times = np.linspace(t0, t1, n_frames)

    # pivot & 内插 x_um, y_um, z_m
    px = df.pivot(index="time", columns="rbc_id", values="x_um") \
           .reindex(sim_times).interpolate()
    py = df.pivot(index="time", columns="rbc_id", values="y_um") \
           .reindex(sim_times).interpolate()
    pz = df.pivot(index="time", columns="rbc_id", values="z_m")  \
           .reindex(sim_times).interpolate()

    X = px.reindex(cam_times, method="nearest") \
          .interpolate(method="index").values
    Y = py.reindex(cam_times, method="nearest") \
          .interpolate(method="index").values
    Z = pz.reindex(cam_times, method="nearest") \
          .interpolate(method="index").values

    # 4) 横向 PSF
    pixel_fov_um = args.fov_mm * 1e3
    num_px       = int(pixel_fov_um / args.px_size_um)
    ksz = int(args.psf_um / args.px_size_um); ksz += 1 - ksz%2
    sigma_px = (args.psf_um/2.355) / args.px_size_um
    ax = np.arange(-ksz//2+1, ksz//2+1)
    xx,yy = np.meshgrid(ax,ax)
    kernel = np.exp(-(xx**2+yy**2)/(2*sigma_px**2))
    kernel /= kernel.sum()

    # 5) 坐标映射 μm→像素
    Xmin, Xmax = X.min(), X.max()
    Ymin, Ymax = Y.min(), Y.max()
    def to_pixel(x,y):
        ix = ((x-Xmin)/(Xmax-Xmin)*(num_px-1)).astype(int)
        iy = ((y-Ymin)/(Ymax-Ymin)*(num_px-1)).astype(int)
        return ix, iy

    # 6) 轴向 PSF σ（m）
    sigma_z = args.sigma_z_nm * 1e-9

    # 7) 渲染
    for fi, _t in enumerate(tqdm(cam_times, desc="Saving frames", ncols=80)):
        img = np.zeros((num_px,num_px),dtype=np.float32)

        Xi, Yi, Zi = X[fi], Y[fi], Z[fi]
        # 计算每个细胞的权重
        w = np.exp(-((Zi - z_focus)**2) / (2*sigma_z**2))

        ix, iy = to_pixel(Xi, Yi)
        valid = (ix>=0)&(ix<num_px)&(iy>=0)&(iy<num_px)
        # 加权累加
        np.add.at(img, (iy[valid], ix[valid]), w[valid])

        img = fftconvolve(img, kernel, mode="same")
        if img.max()>0: img /= img.max()
        img += np.random.normal(scale=args.noise, size=img.shape)
        img = np.clip(img, 0, 1)
        frame = (img*255).astype(np.uint8)

        fname = args.out/f"frame_{fi:05d}.png"
        imageio.imwrite(str(fname), frame)

    print("✅ All frames saved →", args.out)

if __name__=="__main__":
    main()

