"""
tracking_model/phase_separation.py

Phase-separation (RBC 分流概率) 统一实现，支持 “flow” / “logit” 模式
"""
from __future__ import annotations
import numpy as np
import igraph as ig

__all__ = [
    "assign_rbc_probabilities",
    "rbc_split_fraction",
    "branch_prob",
]

# ----------------------------------------------------------------------
# 公用小工具
# ----------------------------------------------------------------------
def _calc_A_B_X0(D_f: float, D_a: float, D_b: float, H: float = 0.4):
    "Eq.(2–4) in Pries 1996 -> μ-law"""
    # 将 m 转为 µm
    D_f, D_a, D_b = D_f * 1e6, D_a * 1e6, D_b * 1e6
    num = D_a**2 / D_b**2 - 1
    denom = D_a**2 / D_b**2 + 1 or 1e-6
    A = -13.29 * (num / denom) * (1 - H) / D_f
    B = 1 + 6.98 * (1 - H) / D_f
    X0 = 0.964 * (1 - H) / D_f
    return A, B, X0


def _logit_model(F_QB: float, A: float, B: float, X0: float):
    num = F_QB - X0
    denom = 1 - 2 * X0 or 1e-6
    logit = np.log(num / denom)
    logit_F_QE = A + B * logit
    return np.exp(logit_F_QE) / (1 + np.exp(logit_F_QE))


# ----------------------------------------------------------------------
# 早期版本：直接写 outs['rbc_probability'] = ...
# ----------------------------------------------------------------------
def branch_prob(g: ig.Graph, H: float = 0.4):
    """
    迭代所有分叉节点（outdegree ≥2）计算 RBC 分流概率。
    基于 logit_model 经验公式
    """
    for v in g.vs:
        outs = g.es.select(_source=v.index)
        ins = g.es.select(_target=v.index)
        if len(outs) < 2:
            continue
        # 父管直径
        D_parent = np.mean(ins['diameter']) if ins else np.mean(outs['diameter'])
        Qs = np.abs(outs['flow_rate'])
        if Qs.sum() == 0:
            outs['rbc_probability'] = np.ones(len(outs)) / len(outs)
            continue
        probs = []
        for e in outs:
            D_a = e['diameter']
            D_b = np.mean([ed['diameter'] for ed in outs if ed != e])
            A, B, X0 = _calc_A_B_X0(D_parent, D_a, D_b, H)
            F_QB = abs(e['flow_rate']) / Qs.sum()
            F_QE = _logit_model(F_QB, A, B, X0)
            probs.append(F_QE)
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()
        outs['rbc_probability'] = probs


# ----------------------------------------------------------------------
# 推荐实现：assign_rbc_probabilities
# 支持 mode="flow" | "logit"
# ----------------------------------------------------------------------
def assign_rbc_probabilities(
    g: ig.Graph,
    mode: str = "flow",
    H_D: float = 0.4,
) -> None:
    """
    为所有出边写入 g.es['rbc_probability']，总和=1。
    mode:
      "flow"  -> 按 flow_rate 比例分配
      "logit" -> 前两条用 logit_model，其他为0
    """
    # initialize
    g.es['rbc_probability'] = [0.0] * g.ecount()
    # 对每个节点的出边
    for v in g.vs:
        outs = g.es.select(_source=v.index)
        n = len(outs)
        if n == 0:
            continue
        # 流量驱动模式
        if mode == "flow":
            flows = np.abs(outs['flow_rate'])
            total = flows.sum()
            if total <= 0:
                probs = np.ones(n) / n
            else:
                probs = flows / total
        # logit 模式
        elif mode == "logit":
            if n == 1:
                probs = np.array([1.0])
            elif n == 2:
                # 用经验公式分配
                ins = g.es.select(_target=v.index)
                D_parent = np.mean(ins['diameter']) if ins else np.mean(outs['diameter'])
                D_alpha, D_beta = outs[0]['diameter'], outs[1]['diameter']
                A, B, X0 = _calc_A_B_X0(D_parent, D_alpha, D_beta, H_D)
                F_QE = _logit_model(0.5, A, B, X0)
                probs = np.array([F_QE, 1 - F_QE])
            else:
                # 三分叉及以上，均分
                probs = np.ones(n) / n
        else:
            raise ValueError(f"未知 mode: {mode}")
        # 归一化写入
        probs = probs / probs.sum()
        for e, p in zip(outs, probs):
            e['rbc_probability'] = float(p)


def rbc_split_fraction(
    g: ig.Graph, mode: str = "flow", H: float = 0.4
):
    """
    旧接口兼容: rbc_split_fraction(g, H) => assign_rbc_probabilities(g, 'logit', H)
    """
    if mode == "default":
        # 兼容老调用：默认行为为 logit
        assign_rbc_probabilities(g, mode="logit", H_D=H)
    else:
        assign_rbc_probabilities(g, mode=mode, H_D=H)
    return g