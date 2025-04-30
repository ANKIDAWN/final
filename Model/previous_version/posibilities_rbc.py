import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm

# 文件路径（请根据您的实际路径修改）
mvn_data_path = r'C:\path\to\MVN1_excel.csv'
output_file_path = r'C:\path\to\rbc_tracks.csv'

# 1. 加载 MVN 数据
mvn_df = pd.read_csv(mvn_data_path)
mvn_df.columns = ['diameters', 'pBC', 'index', 'pressure', 'x', 'y', 'z']
mvn_df.reset_index(drop=True, inplace=True)

# 为每个标记点分配唯一的节点 ID（使用行号作为 node_id）
mvn_df['node_id'] = mvn_df.index

# 创建节点 DataFrame
nodes = mvn_df[['node_id', 'x', 'y', 'z']].copy()
nodes.set_index('node_id', inplace=True)

# 2. 创建边列表并计算边属性
edges = []
edge_attributes = {'diameter': [], 'pressure_diff': [], 'length': []}

# 按照 'index' 分组，代表每个血管段
for vessel_id, group in mvn_df.groupby('index'):
    group = group.sort_values('node_id').reset_index(drop=True)
    node_ids = group['node_id'].tolist()
    pressures = group['pressure'].tolist()
    diameters = group['diameters'].tolist()
    coords = group[['x', 'y', 'z']].values

    for i in range(len(node_ids) - 1):
        source_id = node_ids[i]
        target_id = node_ids[i + 1]

        edges.append((source_id, target_id))

        # 计算边属性
        diameter = (diameters[i] + diameters[i + 1]) / 2
        pressure_diff = abs(pressures[i] - pressures[i + 1])
        length = np.linalg.norm(coords[i + 1] - coords[i])

        edge_attributes['diameter'].append(diameter)
        edge_attributes['pressure_diff'].append(pressure_diff)
        edge_attributes['length'].append(length)

# 3. 创建图形并分配属性
g = ig.Graph(edges=edges, directed=True)

# 分配顶点属性
g.vs['x'] = nodes['x'].values
g.vs['y'] = nodes['y'].values
g.vs['z'] = nodes['z'].values

# 分配边属性
g.es['diameter'] = edge_attributes['diameter']
g.es['pressure_diff'] = edge_attributes['pressure_diff']
g.es['length'] = edge_attributes['length']

# 4. 定义必要的函数
def calculate_mu_rel(diameter, H_D):
    D = diameter * 1e6  # 将直径转换为微米
    mu_045 = 220 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * D**0.645)
    C = (0.8 + np.exp(-0.075 * D)) * (-1 + 1 / (1 + 1e-11 * D**12)) + 1 / (1 + 1e-11 * D**12)
    mu_rel = 1 + (mu_045 - 1) * ((1 - H_D)**C - 1) / ((1 - 0.45)**C - 1)
    return mu_rel

def calculate_A_B_X0(diameter_parent, diameter_branch_alpha, diameter_branch_beta, H_D=0.4):
    D_alpha = diameter_branch_alpha * 1e6  # 转换为微米
    D_beta = diameter_branch_beta * 1e6
    D_F = diameter_parent * 1e6
    numerator = D_alpha**2 / D_beta**2 - 1
    denominator = D_alpha**2 / D_beta**2 + 1
    if denominator == 0:
        denominator = 1e-6  # 避免除以零
    A = -13.29 * (numerator / denominator) * (1 - H_D) / D_F
    B = 1 + 6.98 * (1 - H_D) / D_F
    X_0 = 0.964 * (1 - H_D) / D_F
    return A, B, X_0

def logit_model(F_QB, A, B, X_0):
    numerator = F_QB - X_0
    denominator = 1 - 2 * X_0
    if denominator == 0:
        denominator = 1e-6  # 避免除以零
    logit_F_QB = np.log(numerator / denominator)
    logit_F_QE = A + B * logit_F_QB
    F_QE = np.exp(logit_F_QE) / (1 + np.exp(logit_F_QE))
    return F_QE

def calculate_rbc_velocity(flow_rate, diameter):
    radius = diameter / 2
    cross_section_area = np.pi * radius**2
    if cross_section_area == 0:
        cross_section_area = 1e-6  # 避免除以零
    velocity = flow_rate / cross_section_area
    return velocity

# 5. 计算流量和红细胞速度
viscosity = 0.0035  # Pa·s

def calculate_flow_rate(diameter, pressure_diff, length):
    radius = diameter / 2
    if length == 0:
        length = 1e-6  # 避免除以零
    flow_rate = (np.pi * radius**4 * pressure_diff) / (8 * viscosity * length)
    return flow_rate

flow_rates = [calculate_flow_rate(e['diameter'], e['pressure_diff'], e['length']) for e in g.es]
g.es['flow_rate'] = flow_rates

rbc_velocities = [calculate_rbc_velocity(e['flow_rate'], e['diameter']) for e in g.es]
g.es['rbc_velocity'] = rbc_velocities

# 6. 计算红细胞通过概率
# 初始化红细胞概率
g.es['rbc_probability'] = [1.0] * g.ecount()

# 识别分叉节点（度大于 2 的节点）
degree = g.degree(mode='all')
branch_nodes = [v.index for v, d in zip(g.vs, degree) if d > 2]

# 计算红细胞概率
for node in branch_nodes:
    out_edges = g.es.select(_source=node)
    in_edges = g.es.select(_target=node)
    out_degree = len(out_edges)
    in_degree = len(in_edges)

    if out_degree >= 2:
        # 获取出边的直径
        diameters = np.array(out_edges['diameter'])
        parent_diameter = np.mean([e['diameter'] for e in in_edges]) if in_edges else np.mean(diameters)

        # 确保至少有两条出边
        if len(diameters) >= 2:
            diameter_branch_alpha = diameters[0]
            diameter_branch_beta = diameters[1]

            A, B, X_0 = calculate_A_B_X0(parent_diameter, diameter_branch_alpha, diameter_branch_beta)
            F_QB_alpha = 0.5  # 假设流量均分
            F_QE_alpha = logit_model(F_QB_alpha, A, B, X_0)
            probabilities = [F_QE_alpha, 1 - F_QE_alpha] + [0.0] * (out_degree - 2)
            # 分配概率给出边
            for idx, edge in enumerate(out_edges):
                edge['rbc_probability'] = probabilities[idx]
        else:
            # 如果出边少于 2 条，平均分配概率
            for edge in out_edges:
                edge['rbc_probability'] = 1.0 / out_degree

# 7. 模拟红细胞在网络中的运动
# 设置模拟参数
rbc_count_per_timestep = 10
time_step = 0.1
total_time = 10
time_steps = np.arange(0, total_time, time_step)

# 入口和出口节点
entry_nodes = [v.index for v in g.vs if v.indegree() == 0]
exit_nodes = [v.index for v in g.vs if v.outdegree() == 0]

# 计算入口节点的流入概率
entry_flows = np.array([sum([abs(e['flow_rate']) for e in g.es.select(_source=v.index)]) for v in g.vs.select(entry_nodes)])
total_inflow = np.sum(entry_flows)
if total_inflow == 0:
    entry_probabilities = np.ones(len(entry_nodes)) / len(entry_nodes)
else:
    entry_probabilities = entry_flows / total_inflow

# 初始化红细胞位置
rbc_positions = []
rbc_tracks = []
rbc_id_counter = 0

# 模拟循环，带进度条
for t in tqdm(time_steps, desc='模拟进度', ascii=True, ncols=80, mininterval=0.1):
    # 添加新红细胞
    num_new_rbc = (rbc_count_per_timestep * entry_probabilities).astype(int)
    for node, count in zip(entry_nodes, num_new_rbc):
        for _ in range(count):
            rbc_positions.append({
                'id': rbc_id_counter,
                'edge_index': None,  # 尚未进入任何边
                'position': np.array([g.vs[node]['x'], g.vs[node]['y'], g.vs[node]['z']]),
                'current_node': node,
                'path': [node]
            })
            rbc_id_counter += 1

    # 更新红细胞位置
    new_rbc_positions = []
    for rbc in rbc_positions:
        if rbc['current_node'] is not None:
            # 在节点上，决定下一个边
            current_node = rbc['current_node']
            if current_node in exit_nodes:
                continue  # 红细胞离开网络
            out_edges = g.es.select(_source=current_node)
            if len(out_edges) == 0:
                continue  # 无出边，无法前进
            probabilities = np.array([e['rbc_probability'] for e in out_edges])
            total_prob = np.sum(probabilities)
            if total_prob == 0:
                probabilities = np.ones(len(probabilities)) / len(probabilities)
            else:
                probabilities /= total_prob
            next_edge = np.random.choice(out_edges, p=probabilities)
            rbc['edge_index'] = next_edge.index
            rbc['edge_position'] = 0.0  # 边上的位置从 0 开始
            rbc['current_node'] = None  # 离开节点，进入边
        else:
            # 在边上，更新位置
            edge = g.es[rbc['edge_index']]
            velocity = edge['rbc_velocity']
            distance = velocity * time_step
            edge_length = edge['length']
            rbc['edge_position'] += distance
            if rbc['edge_position'] >= edge_length:
                # 到达边的末端，进入下一个节点
                rbc['current_node'] = edge.target
                rbc['position'] = np.array([g.vs[rbc['current_node']]['x'], g.vs[rbc['current_node']]['y'], g.vs[rbc['current_node']]['z']])
                rbc['path'].append(rbc['current_node'])
                rbc['edge_index'] = None
            else:
                # 更新红细胞在边上的位置
                source_pos = np.array([g.vs[edge.source]['x'], g.vs[edge.source]['y'], g.vs[edge.source]['z']])
                target_pos = np.array([g.vs[edge.target]['x'], g.vs[edge.target]['y'], g.vs[edge.target]['z']])
                direction = target_pos - source_pos
                direction_length = np.linalg.norm(direction)
                if direction_length == 0:
                    continue  # 避免除以零
                direction /= direction_length
                rbc['position'] = source_pos + direction * rbc['edge_position']
        # 记录轨迹
        rbc_tracks.append({
            'rbc_id': rbc['id'],
            'time': t,
            'x': rbc['position'][0],
            'y': rbc['position'][1],
            'z': rbc['position'][2]
        })
        # 将红细胞加入新的列表
        new_rbc_positions.append(rbc)
    # 更新红细胞位置列表
    rbc_positions = new_rbc_positions

# 8. 保存红细胞轨迹数据
rbc_tracks_df = pd.DataFrame(rbc_tracks)
rbc_tracks_df.to_csv(output_file_path, index=False)

# 9. 可视化（可选）
# 您可以使用 VTK 或其他可视化工具来展示红细胞在网络中的运动
# 以下是使用 matplotlib 进行简单的 2D 可视化的示例

import matplotlib.pyplot as plt

# 读取轨迹数据
rbc_tracks_df = pd.read_csv(output_file_path)

# 绘制红细胞轨迹的示例
plt.figure(figsize=(8, 6))
for rbc_id in rbc_tracks_df['rbc_id'].unique():
    rbc_data = rbc_tracks_df[rbc_tracks_df['rbc_id'] == rbc_id]
    plt.plot(rbc_data['x'], rbc_data['y'], linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('红细胞轨迹示例')
plt.show()
