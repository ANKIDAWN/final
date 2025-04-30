import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
import vtk
import os

# ======================== 步骤 1: 设置输入和输出路径 ========================

# 输入文件路径（请根据您的目录结构调整）
node_data_path = '/home/denga/code/persoff/data/network/network_simulated_vertex_data.csv'
edge_data_path = '/home/denga/code/persoff/data/network/network_simulated_edge_data.csv'
mvn_data_path = '/home/denga/code/MVN1_excel.csv'

# 输出目录设置
output_base_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)
print(f"输出目录：{output_base_dir}")

# 输出文件路径
output_csv_path = os.path.join(output_base_dir, 'rbc_tracks.csv')
vessel_output_filename = os.path.join(output_base_dir, 'vessel_network.vtp')
rbc_output_directory = os.path.join(output_base_dir, 'rbc_positions')
if not os.path.exists(rbc_output_directory):
    os.makedirs(rbc_output_directory)

# ======================== 步骤 2: 加载节点数据 ========================

print("\n=== 加载节点数据 ===")
node_df = pd.read_csv(node_data_path)
node_df.columns = node_df.columns.str.strip()
required_node_columns = ['x', 'y', 'z', 'pressure']
missing_node_columns = [col for col in required_node_columns if col not in node_df.columns]
if missing_node_columns:
    raise ValueError(f"节点数据缺少以下列：{missing_node_columns}")

# 如果 node_id 列不存在，使用索引生成 node_id
if 'node_id' not in node_df.columns:
    print("节点数据中缺少 'node_id' 列，将使用索引作为 'node_id'。")
    node_df['node_id'] = node_df.index

# 确保数值列为数值类型
numeric_columns = ['x', 'y', 'z', 'pressure']
for col in numeric_columns:
    node_df[col] = pd.to_numeric(node_df[col], errors='coerce')

# 移除包含缺失值的行
node_df.dropna(subset=numeric_columns + ['node_id'], inplace=True)
node_df.reset_index(drop=True, inplace=True)

# 确保 node_id 为整数
node_df['node_id'] = node_df['node_id'].astype(int)

# ======================== 步骤 3: 加载边数据 ========================

print("\n=== 加载边数据 ===")
edge_df = pd.read_csv(edge_data_path)
edge_df.columns = edge_df.columns.str.strip().str.lower()
print("边数据的列名：", edge_df.columns.tolist())

required_edge_columns = ['vertex_1', 'vertex_2', 'flow_rate', 'rbc_velocity', 'diameter', 'length', 'ht']
missing_edge_columns = [col for col in required_edge_columns if col not in edge_df.columns]
if missing_edge_columns:
    raise ValueError(f"边数据缺少以下列：{missing_edge_columns}")

# 确保数值列为数值类型
for col in required_edge_columns:
    edge_df[col] = pd.to_numeric(edge_df[col], errors='coerce')

# 移除包含缺失值的行
edge_df.dropna(subset=required_edge_columns, inplace=True)
edge_df.reset_index(drop=True, inplace=True)

# 确保 vertex_1 和 vertex_2 为整数
edge_df['vertex_1'] = edge_df['vertex_1'].astype(int)
edge_df['vertex_2'] = edge_df['vertex_2'].astype(int)

# 为每条边分配唯一的 edge_id
edge_df['edge_id'] = edge_df.index

# ======================== 步骤 4: 加载 MVN 数据 ========================

print("\n=== 加载 MVN 数据 ===")
mvn_df = pd.read_csv(mvn_data_path)
mvn_df.columns = mvn_df.columns.str.strip()
print("MVN 数据的列名：", mvn_df.columns.tolist())

required_mvn_columns = ['diameters', 'pBC', 'index', 'pressure', 'coords:0', 'coords:1', 'coords:2']
missing_mvn_columns = [col for col in required_mvn_columns if col not in mvn_df.columns]
if missing_mvn_columns:
    raise ValueError(f"MVN 数据缺少以下列：{missing_mvn_columns}")

# 确保数值列为数值类型
for col in required_mvn_columns:
    mvn_df[col] = pd.to_numeric(mvn_df[col], errors='coerce')

# 移除包含缺失值的行
mvn_df.dropna(subset=required_mvn_columns, inplace=True)
mvn_df.reset_index(drop=True, inplace=True)

# 确保 index 为整数
mvn_df['index'] = mvn_df['index'].astype(int)

# 为每条边构建坐标列表
print("\n=== 构建边的坐标字典 ===")
edge_coords_dict = {}
for edge_id, group in mvn_df.groupby('index'):
    group_sorted = group.sort_values('index')
    coords = list(zip(group_sorted['coords:0'], group_sorted['coords:1'], group_sorted['coords:2']))
    edge_coords_dict[edge_id] = coords

# 检查边的数量是否匹配
if len(edge_coords_dict) != len(edge_df):
    print(f"警告：MVN 数据中的边数量 ({len(edge_coords_dict)}) 与边数据中的边数量 ({len(edge_df)}) 不一致。只处理匹配的边。")

# ======================== 步骤 5: 创建节点 ID 到索引的映射 ========================

print("\n=== 创建节点 ID 到索引的映射 ===")
unique_node_ids = np.unique(np.concatenate([
    node_df['node_id'].values.astype(int),
    edge_df['vertex_1'].values.astype(int),
    edge_df['vertex_2'].values.astype(int)
]))
node_id_to_index = {int(node_id): int(idx) for idx, node_id in enumerate(unique_node_ids)}
print(f"总唯一节点数：{len(unique_node_ids)}")

# 为节点分配索引
node_df['index'] = node_df['node_id'].map(node_id_to_index).astype(int)

# 为边分配源节点和目标节点的索引
edge_df['source'] = edge_df['vertex_1'].map(node_id_to_index).astype(int)
edge_df['target'] = edge_df['vertex_2'].map(node_id_to_index).astype(int)

# 检查是否有无法映射的顶点
if edge_df['source'].isnull().any() or edge_df['target'].isnull().any():
    missing_vertex_ids = []
    if edge_df['source'].isnull().any():
        missing_vertex_ids.extend(edge_df[edge_df['source'].isnull()]['vertex_1'].unique().tolist())
    if edge_df['target'].isnull().any():
        missing_vertex_ids.extend(edge_df[edge_df['target'].isnull()]['vertex_2'].unique().tolist())
    raise ValueError(f"以下顶点 ID 无法映射到索引：{missing_vertex_ids}")

# ======================== 步骤 6: 构建有向图 ========================

print("\n=== 构建有向图 ===")
# 根据流量方向确定边的方向
edges = []
for idx, row in edge_df.iterrows():
    source = int(row['source'])
    target = int(row['target'])
    flow_rate = row['flow_rate']
    if flow_rate >= 0:
        edges.append((source, target))
    else:
        edges.append((target, source))

# 创建 igraph 图
num_vertices = len(unique_node_ids)
g = ig.Graph(edges=edges, directed=True, n=num_vertices)
print(f"图的顶点数：{g.vcount()}，边数：{g.ecount()}")

# ======================== 步骤 7: 分配节点和边的属性 ========================

print("\n=== 分配节点和边的属性 ===")
# 分配节点属性
node_index_to_row = node_df.set_index('index')
g.vs['node_id'] = [node_index_to_row.loc[idx]['node_id'] if idx in node_index_to_row.index else None for idx in range(g.vcount())]
g.vs['x'] = [node_index_to_row.loc[idx]['x'] if idx in node_index_to_row.index else 0.0 for idx in range(g.vcount())]
g.vs['y'] = [node_index_to_row.loc[idx]['y'] if idx in node_index_to_row.index else 0.0 for idx in range(g.vcount())]
g.vs['z'] = [node_index_to_row.loc[idx]['z'] if idx in node_index_to_row.index else 0.0 for idx in range(g.vcount())]

# 分配边属性
g.es['flow_rate'] = edge_df['flow_rate'].values
g.es['rbc_velocity'] = edge_df['rbc_velocity'].values
g.es['diameter'] = edge_df['diameter'].values
g.es['length'] = edge_df['length'].values
g.es['ht'] = edge_df['ht'].values
g.es['edge_id'] = edge_df['edge_id'].values

# 分配边的坐标
for e in g.es:
    edge_id = e['edge_id']
    if edge_id in edge_coords_dict:
        e['coords'] = edge_coords_dict[edge_id]
    else:
        # 如果 MVN 数据中缺少坐标，使用源和目标节点的坐标
        source = e.source
        target = e.target
        coords = [
            (g.vs[source]['x'], g.vs[source]['y'], g.vs[source]['z']),
            (g.vs[target]['x'], g.vs[target]['y'], g.vs[target]['z'])
        ]
        e['coords'] = coords

# ======================== 步骤 8: 识别入口和出口节点 ========================

print("\n=== 识别入口和出口节点 ===")
entry_nodes = []
exit_nodes = []
for v in g.vs:
    in_degree = v.indegree()
    out_degree = v.outdegree()
    if in_degree == 0 and out_degree > 0:
        entry_nodes.append(v.index)
    elif in_degree > 0 and out_degree == 0:
        exit_nodes.append(v.index)
print(f"识别到的入口节点数量：{len(entry_nodes)}")
print(f"识别到的出口节点数量：{len(exit_nodes)}")


# ======================== 步骤 9: 单位转换和 RBC 生成速率计算 ========================

print("\n=== 单位转换和 RBC 生成速率计算 ===")
# 根据您的实际数据单位，修改以下变量
flow_rate_unit = 'μL/min'  # 假设流量单位为微升每分钟
diameter_unit = 'μm'       # 假设直径单位为微米

# 设置流量转换因子
flow_rate_conversion_factor = 1e-2 / 60.0  # 提高流量单位换算

# 设置直径转换因子
if diameter_unit == 'μm':
    diameter_conversion_factor = 1e-4  # 1 μm = 1e-4 cm
else:
    print("未知的 diameter 单位，请设置正确的转换因子。")
    diameter_conversion_factor = 1.0  # 默认不转换

# 筛选有效的入口血管
entry_vessels = []
entry_vessel_diameters = []
entry_vessel_velocities = []
entry_vessel_flow_rates = []

for node_index in entry_nodes:
    out_edges = g.es.select(_source=node_index)
    for e in out_edges:
        entry_vessels.append(e)
        diameter = e['diameter']  # 单位为 μm
        velocity = abs(e['rbc_velocity'])  # 确保速度为正
        flow_rate = e['flow_rate']
        entry_vessel_diameters.append(diameter)
        entry_vessel_velocities.append(velocity)
        entry_vessel_flow_rates.append(flow_rate)

# 确认 entry_vessel_diameters 是否有值
if not entry_vessel_diameters:
    raise ValueError("entry_vessel_diameters 列表为空，请检查入口血管的提取是否正确。")

# 应用转换因子
entry_vessel_diameters_cm = np.abs(np.array(entry_vessel_diameters)) * diameter_conversion_factor
entry_vessel_velocities_cm_per_s = np.abs(np.array(entry_vessel_velocities))  # 确保速度为正
entry_vessel_volumetric_flow_rates = np.abs(np.array(entry_vessel_flow_rates)) * flow_rate_conversion_factor  # cm³/s

# 打印转换后的直径和速度（仅打印一次）
print("\n--- 入口血管的直径和速度（cm 和 cm/s）示例 ---")
for i in range(min(5, len(entry_vessel_diameters_cm))):
    print(f"入口血管 {i}: 直径 = {entry_vessel_diameters_cm[i]:.6e} cm, 速度 = {entry_vessel_velocities_cm_per_s[i]:.6e} cm/s")

# 筛选有效的入口血管
valid_indices = np.where(entry_vessel_volumetric_flow_rates > 1e-6)[0]  # 流量大于阈值
print("\n入口血管筛选前的数量:", len(entry_vessels))
print("有效的入口血管数量:", len(valid_indices))

if len(valid_indices) == 0:
    print("没有找到有效的入口血管，将使用所有入口血管参与模拟。")
    valid_indices = np.arange(len(entry_vessels))  # 使用所有入口血管

# 过滤有效的入口血管
entry_vessels = [entry_vessels[i] for i in valid_indices]
entry_vessel_diameters_cm = entry_vessel_diameters_cm[valid_indices]
entry_vessel_velocities_cm_per_s = entry_vessel_velocities_cm_per_s[valid_indices]
entry_vessel_volumetric_flow_rates = entry_vessel_volumetric_flow_rates[valid_indices]

# 打印检查入口血管数据
print("\n--- 初始入口血管数据检查 ---")
print(f"有效入口血管数量: {len(entry_vessels)}")
print("流量 (cm³/s):", entry_vessel_volumetric_flow_rates[:5])  # 打印前5个流量
print("直径 (cm):", entry_vessel_diameters_cm[:5])             # 打印前5个直径
print("速度 (cm/s):", entry_vessel_velocities_cm_per_s[:5])    # 打印前5个速度s

# ======================== 步骤 10: 计算 RBC 生成速率 ========================

print("\n=== 计算 RBC 生成速率 ===")
# RBC 浓度（个/cm³），缩放因子
# 增加 RBC 生成浓度
N_RBC_per_cm3 = 1e21  # 从 5e9 增加到 1e12
entry_vessel_rbc_generation_rates = entry_vessel_volumetric_flow_rates * N_RBC_per_cm3
scaling_factor = 1.0  # 调整缩放因子
N_RBC_per_cm3_scaled = N_RBC_per_cm3 * scaling_factor
print(f"调整后的 RBC 浓度（个/cm³）：{N_RBC_per_cm3_scaled}")

# 计算 RBC 生成速率（个/s）
entry_vessel_rbc_generation_rates = entry_vessel_volumetric_flow_rates * N_RBC_per_cm3_scaled  # cells/s

# 打印 RBC 生成速率示例和统计（仅打印一次）
print("\n--- RBC 生成速率（个/s）示例 ---")
print(entry_vessel_rbc_generation_rates[:5])

print("\n--- RBC 生成速率统计 ---")
print(f"最小生成速率: {entry_vessel_rbc_generation_rates.min():.6e} cells/s")
print(f"最大生成速率: {entry_vessel_rbc_generation_rates.max():.6e} cells/s")
print(f"平均生成速率: {entry_vessel_rbc_generation_rates.mean():.6e} cells/s")

# 打印每个入口血管的 RBC 生成速率（仅打印前5个）
print("\n--- 每个入口血管的 RBC 生成速率 (cells/s) ---")
for idx, rate in enumerate(entry_vessel_rbc_generation_rates[:5]):
    print(f"入口血管 {idx}, 生成速率: {rate:.6e} cells/s")

# ======================== 步骤 11: 初始化模拟参数 ========================

print("\n=== 初始化模拟参数 ===")
# 累计 RBC 计数
cumulative_rbc = np.zeros(len(entry_vessels))

# 模拟参数
time_step = 10      # 增加时间步长，秒
total_time = 500  # 延长总模拟时间，秒
time_steps = np.arange(0, total_time, time_step)
rbc_positions = []
rbc_tracks = []
rbc_id_counter = 0
total_rbc_count = 1000000  # 总 RBC 数量上限

# ======================== 步骤 12: 定义导出 RBC 位置到 VTK 的函数 ========================

def create_time_array(time_value):
    """创建 VTK 中的时间数组"""
    time_array = vtk.vtkDoubleArray()
    time_array.SetName("TimeValue")
    time_array.SetNumberOfTuples(1)
    time_array.SetValue(0, time_value)
    return time_array

def export_rbc_positions_to_vtk(rbc_positions, time_step_index, time_value, output_directory):
    """将 RBC 位置导出到 VTK 文件"""
    try:
        # 创建 vtkPoints 对象
        points = vtk.vtkPoints()

        # 创建存储 RBC ID 的数组
        rbc_id_array = vtk.vtkIntArray()
        rbc_id_array.SetName('RBC_ID')

        # 遍历 RBC 并添加点和属性
        for rbc in rbc_positions:
            x, y, z = rbc['position']
            points.InsertNextPoint(x, y, z)
            rbc_id_array.InsertNextValue(int(rbc['id']))

        # 创建 vtkPolyData 对象
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # 添加属性到点数据
        polydata.GetPointData().AddArray(rbc_id_array)

        # 设置时间步信息
        polydata.GetFieldData().AddArray(create_time_array(time_value))

        # 写入 VTK 文件
        output_filename = os.path.join(output_directory, f"rbc_positions_{time_step_index:06d}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_filename)
        writer.SetInputData(polydata)
        writer.Write()
    except Exception as e:
        print(f"导出时间步 {time_value} 的 RBC 位置时发生错误：{e}")

# ======================== 步骤 13: 运行模拟循环 ========================

for step_idx, t in enumerate(time_steps):
    for idx, e in enumerate(entry_vessels):
        cumulative_rbc[idx] += entry_vessel_rbc_generation_rates[idx] * time_step
        num_new_rbc = int(cumulative_rbc[idx])
        cumulative_rbc[idx] -= num_new_rbc
        print(f"时间步 {t}s, 累计计数 = {cumulative_rbc[idx]:.2f}, 新生成 = {num_new_rbc}")

        # 如果有新的 RBC 被生成
        if num_new_rbc > 0:
            for _ in range(num_new_rbc):
                if rbc_id_counter >= total_rbc_count:
                    break
                # 在入口血管的源节点生成 RBC
                source_node = e.source
                rbc_position = np.array([
                    g.vs[source_node]['x'],
                    g.vs[source_node]['y'],
                    g.vs[source_node]['z']
                ])
                rbc = {
                    'id': rbc_id_counter,
                    'edge_index': e.index,
                    'position': rbc_position,
                    'current_node': None,  # 正在移动的边
                    'edge_position': 0.0,
                    'path': [source_node]
                }
                rbc_positions.append(rbc)

                # 记录 RBC 的初始位置
                rbc_tracks.append({
                    'rbc_id': rbc_id_counter,
                    'time': t,
                    'x': rbc_position[0],
                    'y': rbc_position[1],
                    'z': rbc_position[2]
                })

                rbc_id_counter += 1

    # 更新 RBC 位置
    new_rbc_positions = []
    for rbc in rbc_positions:
        if rbc['current_node'] is not None:
            # 在节点处，决定下一个边
            current_node = rbc['current_node']
            if current_node in exit_nodes:
                # RBC 离开网络
                rbc_tracks.append({
                    'rbc_id': rbc['id'],
                    'time': t,
                    'x': rbc['position'][0],
                    'y': rbc['position'][1],
                    'z': rbc['position'][2]
                })
                continue
            out_edges = g.es.select(_source=current_node)
            if len(out_edges) == 0:
                # 没有出边
                rbc_tracks.append({
                    'rbc_id': rbc['id'],
                    'time': t,
                    'x': rbc['position'][0],
                    'y': rbc['position'][1],
                    'z': rbc['position'][2]
                })
                continue

            # 根据流量概率选择下一条边
            probabilities = np.array([abs(e['flow_rate']) for e in out_edges])
            total_prob = np.sum(probabilities)
            if total_prob == 0:
                probabilities = np.ones(len(probabilities)) / len(probabilities)
            else:
                probabilities /= total_prob

            selected_index = np.random.choice(len(out_edges), p=probabilities)
            next_edge = out_edges[selected_index]

            # 更新 RBC 的移动信息
            rbc['edge_index'] = next_edge.index
            rbc['edge_position'] = 0.0
            rbc['current_node'] = None
            rbc['position'] = np.array([
                g.vs[current_node]['x'],
                g.vs[current_node]['y'],
                g.vs[current_node]['z']
            ])
            rbc['path'].append(next_edge.target)
        else:
            # 在边上移动
            edge = g.es[rbc['edge_index']]
            velocity = abs(edge['rbc_velocity'])  # 确保速度为正
            distance = velocity * time_step
            edge_length = edge['length']

            # 更新 RBC 在边上的位置
            rbc['edge_position'] += distance

            if rbc['edge_position'] >= edge_length:
                # 到达边的末端
                rbc['current_node'] = edge.target
                rbc['position'] = np.array([
                    g.vs[edge.target]['x'],
                    g.vs[edge.target]['y'],
                    g.vs[edge.target]['z']
                ])
                rbc['path'].append(edge.target)
                rbc['edge_index'] = None
                rbc['edge_position'] = 0.0
            else:
                # 沿边移动
                coords = edge['coords']
                segment_lengths = [np.linalg.norm(np.array(coords[i+1]) - np.array(coords[i])) for i in range(len(coords)-1)]
                cumulative_lengths = np.cumsum([0] + segment_lengths)
                total_length = cumulative_lengths[-1]
                if total_length == 0:
                    rbc['position'] = np.array(coords[0])
                else:
                    position_along_edge = rbc['edge_position']
                    if position_along_edge >= total_length:
                        position_along_edge = total_length - 1e-6
                    idx_pos = np.searchsorted(cumulative_lengths, position_along_edge) - 1
                    segment_start = np.array(coords[idx_pos])
                    segment_end = np.array(coords[idx_pos + 1])
                    segment_fraction = (position_along_edge - cumulative_lengths[idx_pos]) / segment_lengths[idx_pos]
                    rbc['position'] = segment_start + segment_fraction * (segment_end - segment_start)

        # 记录 RBC 的当前位置
        rbc_tracks.append({
            'rbc_id': rbc['id'],
            'time': t,
            'x': rbc['position'][0],
            'y': rbc['position'][1],
            'z': rbc['position'][2]
        })
        new_rbc_positions.append(rbc)

    # 更新 RBC 位置列表
    rbc_positions = new_rbc_positions

    # 每隔 1000 步打印一次总体进度
    if step_idx % 1000 == 0 and step_idx > 0:
        print(f"模拟进度：{(step_idx / len(time_steps)) * 100:.1f}%，当前时间：{t:.1f}s，当前 RBC 数量：{len(rbc_positions)}")

    # 可选：导出 RBC 位置到 VTK（如果需要可视化）
    # if step_idx % 100 == 0:
    #     export_rbc_positions_to_vtk(rbc_positions, step_idx, t, rbc_output_directory)

# ======================== 步骤 14: 生成 .pvd 文件用于可视化（可选） ========================

# 如果您启用了 VTK 导出，可以生成 PVD 文件
# print("\n=== 生成 .pvd 文件用于可视化 ===")
# def generate_pvd_file(output_directory, time_steps):
#     """生成 .pvd 文件以便在 Paraview 中可视化"""
#     pvd_content = '''<?xml version="1.0"?>
# <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
#   <Collection>
# '''
#     for step_idx, t in enumerate(time_steps):
#         filename = f"rbc_positions_{step_idx:06d}.vtp"
#         pvd_content += f'    <DataSet timestep="{t}" group="" part="0" file="{filename}"/>\n'
#     pvd_content += '''  </Collection>
# </VTKFile>
# '''
#     pvd_filename = os.path.join(output_directory, "rbc_positions.pvd")
#     with open(pvd_filename, 'w') as f:
#         f.write(pvd_content)
#     print(f"成功生成 PVD 文件：{pvd_filename}")

# generate_pvd_file(rbc_output_directory, time_steps)

# ======================== 步骤 15: 保存 RBC 轨迹数据 ========================

print("\n=== 保存 RBC 轨迹数据 ===")
rbc_tracks_df = pd.DataFrame(rbc_tracks)
print("rbc_tracks_df 的列名：", rbc_tracks_df.columns.tolist())
if 'rbc_id' not in rbc_tracks_df.columns:
    raise ValueError("rbc_tracks_df 中没有 'rbc_id' 列，请检查数据。")
rbc_tracks_df.to_csv(output_csv_path, index=False)
print(f"成功保存 RBC 轨迹数据到 {output_csv_path}")

# ======================== 步骤 16: 导出血管网络到 VTK 文件（可选） ========================

# print("\n=== 导出血管网络到 VTK 文件 ===")
# def export_vessel_network_to_vtk(graph, output_filename):
#     """将血管网络导出到 VTK 文件"""
#     try:
#         print(f"开始导出血管网络到 {output_filename}")
#         points = vtk.vtkPoints()
#         point_id_counter = 0
#         point_id_map = {}

#         lines = vtk.vtkCellArray()

#         diameter_array = vtk.vtkFloatArray()
#         diameter_array.SetName('Diameter')

#         for e in tqdm(graph.es, desc='处理血管边', ascii=True, ncols=80):
#             coords = e['coords']
#             point_ids = vtk.vtkIdList()
#             for coord in coords:
#                 coord_tuple = tuple(coord)
#                 if coord_tuple not in point_id_map:
#                     point_id = point_id_counter
#                     points.InsertNextPoint(coord)
#                     point_id_map[coord_tuple] = point_id
#                     point_id_counter += 1
#                 else:
#                     point_id = point_id_map[coord_tuple]
#                 point_ids.InsertNextId(point_id)
#             lines.InsertNextCell(point_ids)
#             diameter_array.InsertNextValue(e['diameter'])

#         polydata = vtk.vtkPolyData()
#         polydata.SetPoints(points)
#         polydata.SetLines(lines)

#         polydata.GetCellData().AddArray(diameter_array)

#         writer = vtk.vtkXMLPolyDataWriter()
#         writer.SetFileName(output_filename)
#         writer.SetInputData(polydata)
#         writer.Write()
#         print(f"血管网络成功导出到 {output_filename}")
#     except Exception as e:
#         print(f"导出血管网络时发生错误：{e}")

# export_vessel_network_to_vtk(g, vessel_output_filename)

print("\n=== 模拟完成，所有数据已保存 ===")
