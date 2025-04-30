
"""
Inverse model & boundary condition tuning

加载反演模型相关 CSV:
  - edge_parameters.csv      : 基础参数
  - edge_target.csv          : 目标流量/压力
  - edge_target_BC_tuning.csv: 调优因子
运行调优算法，给图中每条边写入 'target_pressure' 属性，
并导出调优后的边表 CSV。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import igraph as ig
from typing import Union

__all__ = [
    "tune_boundary_conditions",
]


def tune_boundary_conditions(
    g: ig.Graph,
    inverse_model_dir: Union[Path, str],
    out_edges_csv: Union[Path, str],
) -> ig.Graph:
    """
    加载 data/inverse_model 下的 CSV，执行边界条件调优。

    Parameters
    ----------
    g : ig.Graph
        要调优的血管网络，需有 'edge_id' 属性
    inverse_model_dir : Path or str
        包含 edge_parameters.csv, edge_target.csv, edge_target_BC_tuning.csv 的目录
    out_edges_csv : Path or str
        调优后边属性导出 CSV 路径

    Returns
    -------
    g : ig.Graph
        在 g.es 中新增 'target_pressure' 属性
    """
    inv_dir = Path(inverse_model_dir)
    # 读取 CSV
    params_file = inv_dir / 'edge_parameters.csv'
    target_file = inv_dir / 'edge_target.csv'
    tuning_file = inv_dir / 'edge_target_BC_tuning.csv'
    for f in (params_file, target_file, tuning_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"缺少反演模型文件: {f}")

    df_params = pd.read_csv(params_file)
    df_target = pd.read_csv(target_file)
    df_tune = pd.read_csv(tuning_file)

    # 合并 DataFrame，按 edge_id
    df = df_params.merge(df_target, on='edge_id').merge(df_tune, on='edge_id')

    # 假设调优后压力 = 原目标压力 * 调优因子
    if 'target_pressure' not in df.columns or 'tuning_factor' not in df.columns:
        raise ValueError("反演模型 CSV 中需包含 'target_pressure' 和 'tuning_factor' 列")
    df['adjusted_pressure'] = df['target_pressure'] * df['tuning_factor']

    # 写入图的边属性
    # 确保按 edge_id 顺序
    df_sorted = df.sort_values('edge_id')
    if len(df_sorted) != g.ecount():
        raise ValueError(
            f"边数不匹配: 图有 {g.ecount()} 条边, CSV 有 {len(df_sorted)} 条记录"
        )
    g.es['target_pressure'] = df_sorted['adjusted_pressure'].tolist()

    # 导出调优后的边表
    out_path = Path(out_edges_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 构造 DataFrame
    data = {'edge_id': list(range(g.ecount()))}
    for attr in g.es.attribute_names():
        data[attr] = g.es[attr]
    out_df = pd.DataFrame(data)
    out_df.to_csv(out_path, index=False)
    print(f"[inverse_model] Tuned edges CSV → {out_path}")

    return g