import pandas as pd
from collections import defaultdict

file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_simulated_edge_data.csv'

df = pd.read_csv(file_path, header=None, names=["vertex_1", "vertex_2", "diameter", "length", "flow_rate", "rbc_velocity", "ht"])

connections_vertex_1 = defaultdict(set)
connections_vertex_2 = defaultdict(set)

for _, row in df.iterrows():
    vertex_1 = row['vertex_1']
    vertex_2 = row['vertex_2']
    connections_vertex_1[vertex_1].add(vertex_2)
    connections_vertex_2[vertex_2].add(vertex_1)

ordered_vertex_1 = df['vertex_1'].unique()

result_data_vertex_1 = []
for vertex_1 in ordered_vertex_1:
    connected_vertices = sorted(list(connections_vertex_1[vertex_1]))
    count = len(connected_vertices)
    if count == 5:
        result_data_vertex_1.append([vertex_1, ', '.join(map(str, connected_vertices)), "WRONG_WRONG_WRONG"])
    elif count == 4:
        result_data_vertex_1.append([vertex_1, ', '.join(map(str, connected_vertices)), "WRONG_WRONG"])
    elif count == 3:
        result_data_vertex_1.append([vertex_1, ', '.join(map(str, connected_vertices)), "WRONG"])
    else:
        result_data_vertex_1.append([vertex_1, ', '.join(map(str, connected_vertices)), ""])

result_df_vertex_1 = pd.DataFrame(result_data_vertex_1, columns=["vertex_1", "connected_vertex_2", "T/F"])

result_df_vertex_1.to_csv('output_vertex_1.csv', index=False)

sorted_vertex_2 = sorted(connections_vertex_2.keys())

result_data_vertex_2 = []
for vertex_2 in sorted_vertex_2:
    connected_vertex_1_count = len(connections_vertex_2[vertex_2])
    if connected_vertex_1_count == 5:
        result_data_vertex_2.append([vertex_2, connected_vertex_1_count, "WRONG_WRONG_WRONG_WRONG"])
    elif connected_vertex_1_count == 4:
        result_data_vertex_2.append([vertex_2, connected_vertex_1_count, "WRONG_WRONG_WRONG"])
    elif connected_vertex_1_count == 3:
        result_data_vertex_2.append([vertex_2, connected_vertex_1_count, "WRONG_WRONG"])  
    elif connected_vertex_1_count == 2:
        result_data_vertex_2.append([vertex_2, connected_vertex_1_count, "WRONG"])   
    else:
        result_data_vertex_2.append([vertex_2, connected_vertex_1_count, ""])

result_df_vertex_2 = pd.DataFrame(result_data_vertex_2, columns=["vertex_2", "connected_vertex_1_count", "T/F"])

result_df_vertex_2.to_csv('output_vertex_2.csv', index=False)


import pandas as pd
import pickle
import pyvista as pv

# 文件路径
vertex_file_path = r'C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\data\network\network_simulated_vertex_data.csv'
vertex_1_df_path = 'output_vertex_1.csv'
vertex_2_df_path = 'output_vertex_2.csv'

# 读取3D坐标和压力数据
vertex_df = pd.read_csv(vertex_file_path, header=None, names=["x", "y", "z", "pressure"])

# 创建一个映射，将每个顶点代号与3D坐标文件中的行索引对应起来
index_mapping = {index: i for i, index in enumerate(vertex_df.index)}

# 读取之前生成的CSV文件
vertex_1_df = pd.read_csv(vertex_1_df_path)
vertex_2_df = pd.read_csv(vertex_2_df_path)

# 合并两个DataFrame
merged_df = pd.merge(vertex_1_df, vertex_2_df, left_on='vertex_1', right_on='vertex_2', how='outer', suffixes=('_left', '_right'))

# 检查合并后的DataFrame列名
print("Merged DataFrame Columns:", merged_df.columns)

# 提取连接信息
edges = []
for index, row in merged_df.iterrows():
    vertex_1 = int(row['vertex_1'])
    connected_vertices = row['connected_vertex_2_left'].split(', ') if pd.notna(row['connected_vertex_2_left']) else []
    for vertex_2 in connected_vertices:
        vertex_1_idx = index_mapping[vertex_1]
        vertex_2_idx = index_mapping[int(vertex_2)]
        edges.append((vertex_1_idx, vertex_2_idx))

# 保存为pkl文件
with open('complete_network.pkl', 'wb') as f:
    pickle.dump(merged_df, f)

# 创建点和边的数据结构
points = vertex_df[['x', 'y', 'z']].values

# 创建PolyData对象
polydata = pv.PolyData(points)
lines = []
for edge in edges:
    lines.extend([2, edge[0], edge[1]])

polydata.lines = lines

# 保存为VTK文件
polydata.save('complete_network.vtk')

