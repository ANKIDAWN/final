import pandas as pd
import numpy as np

# 文件路径
edge_file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_simulated_edge_data.csv'
vertex_file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_simulated_vertex_data.csv'

# 读取连接数据
edge_df = pd.read_csv(edge_file_path)

# 读取顶点的3D坐标和压力数据，并进行数据处理
vertex_df = pd.read_csv(vertex_file_path, header=None)
vertex_df = vertex_df[pd.to_numeric(vertex_df[3], errors='coerce').notnull()]
vertex_df[3] = vertex_df[3].astype(float)

# 获取压力数据
vertex_pressure = vertex_df[3].values

# 假设血液的粘度 μ 和全血比容 H_T
mu = 3.5e-3  # 3.5 mPa·s
H_T = 0.3  # 全血比容

# Hagen-Poiseuille定律计算流量 Q = π * d^4 * |ΔP| / (128 * μ * L)
def calculate_flow_rate(diameter, delta_p, length, mu_rel):
    return (np.pi * diameter**4 * abs(delta_p)) / (128 * mu * mu_rel * length)

# 计算相对粘度 mu_rel（根据Fahraeus-Lindqvist效应，需要一个经验公式）
def calculate_mu_rel(diameter):
    # 假设使用某个经验公式来计算相对粘度 mu_rel
    if diameter < 0.01:  # 直径小于0.01米的情况
        return 1 + (1.3 - 1) * np.exp(-diameter / 0.001)
    else:
        return 1

# Fahraeus效应计算RBC速度
def calculate_rbc_velocity(H_D, v_Bulk):
    return (H_D / H_T) * v_Bulk

# 遍历edge_df，计算每条边的流量和RBC速度
flow_rates = []
rbc_velocities = []
flow_directions = []

for _, row in edge_df.iterrows():
    vertex_1 = int(row['vertex_1'])
    vertex_2 = int(row['vertex_2'])
    diameter = row['diameter']
    length = row['length']
    v_Bulk = row['rbc_velocity']  # 使用已有的RBC速度计算
    
    # 计算压差
    delta_p = vertex_pressure[vertex_1] - vertex_pressure[vertex_2]
    
    # 确定流动方向
    if delta_p >= 0:
        flow_direction = (vertex_1, vertex_2)  # 正向
    else:
        flow_direction = (vertex_2, vertex_1)  # 反向
        assert true, "negative pressure"
    
    flow_directions.append(flow_direction)
    
    # 计算相对粘度 mu_rel
    mu_rel = calculate_mu_rel(diameter)
    
    # 计算流量，使用绝对值的压差
    flow_rate = calculate_flow_rate(diameter, delta_p, length, mu_rel)
    flow_rates.append(flow_rate)
    
    # 计算RBC速度
    v_RBC = calculate_rbc_velocity(0.4, v_Bulk)  # 假设H_D为0.4
    rbc_velocities.append(v_RBC)

# 将计算得到的流量、RBC速度和流动方向添加到DataFrame中
edge_df['calculated_flow_rate'] = flow_rates
edge_df['calculated_rbc_velocity'] = rbc_velocities
edge_df['flow_direction'] = flow_directions

# 计算RBC进入不同分支的概率
probabilities = []
for vertex_1, group in edge_df.groupby('vertex_1'):
    total_flow = group['calculated_flow_rate'].sum()
    
    # 防止总流量为零或负数
    if total_flow <= 0:
        print(f"警告: Vertex {vertex_1} 的总流量小于或等于零，将跳过此节点的计算。")
        probabilities.extend([0] * len(group))
        continue
    
    branch_probabilities = []
    
    if len(group) == 2:
        # 二分叉情况，根据流量比例计算概率
        for flow_rate in group['calculated_flow_rate']:
            probability = max(0, min(flow_rate / total_flow, 1))
            branch_probabilities.append(probability)
    else:
        # 三分叉及以上情况，均分概率
        branch_probabilities = [1/len(group)] * len(group)
    
    # 确保概率和为1，如果有误差，进行调整
    total_prob = sum(branch_probabilities)
    if total_prob != 1.0:
        branch_probabilities = [p / total_prob for p in branch_probabilities]

    probabilities.extend(branch_probabilities)

# 将概率添加到DataFrame中
edge_df['RBC_probability'] = probabilities

# 保存结果到新的CSV文件
output_file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_RBC_probability_by_flowrate.csv'
edge_df.to_csv(output_file_path, index=False)

print(f"RBC流向概率已计算完成并保存在: {output_file_path}")

# 将点代号和对应的概率添加到列表中
for prob in branch_probabilities:
    vertex_probabilities.append((vertex_1, prob))

# 将点代号和概率保存为DataFrame
vertex_prob_df = pd.DataFrame(vertex_probabilities, columns=['vertex_1', 'RBC_probability'])

# 保存结果到新的CSV文件
probability_output_file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_vertex_probabilities.csv'
vertex_prob_df.to_csv(probability_output_file_path, index=False)

print(f"点代号和概率表格已保存为: {probability_output_file_path}")