import csv
import pandas as pd
from collections import deque, defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, length, speed):
        self.graph[u].append((v, length, speed))

def load_graph_from_csv(file_path):
    graph = Graph()
    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        headers = [header.lstrip('\ufeff') for header in reader.fieldnames]
        for row in reader:
            row = {headers[i]: value for i, value in enumerate(row.values())}  # Rebuild row dictionary with cleaned headers
            graph.add_edge(row['start'], row['end'], float(row['length']), float(row['speed']))
    return graph

def bfs_paths(graph, start):
    queue = deque([(start, [start], 0)])  # (current node, path, cumulative time)
    paths = []
    
    while queue:
        node, path, cum_time = queue.popleft()
        
        if node != start:
            paths.append((path, cum_time))
        
        for (adjacent, length, speed) in graph[node]:
            if adjacent not in path:  # Avoid cycles
                new_cum_time = cum_time + length / speed
                queue.append((adjacent, path + [adjacent], new_cum_time))
    
    return paths

def generate_csv(graph, paths, timestep):
    for idx, (path, total_time) in enumerate(paths):
        records = []
        
        current_time = 0
        path_length = len(path)
        while current_time <= total_time:
            cum_time = 0
            for i in range(path_length - 1):
                start_node = path[i]
                end_node = path[i + 1]
                length = next(edge[1] for edge in graph[start_node] if edge[0] == end_node)
                speed = next(edge[2] for edge in graph[start_node] if edge[0] == end_node)
                travel_time = length / speed

                if cum_time <= current_time < cum_time + travel_time:
                    edge_traversed_time = current_time - cum_time
                    position_on_edge = edge_traversed_time * speed
                    records.append([len(records) + 1, current_time, start_node, end_node, position_on_edge, path])
                    break
                cum_time += travel_time
            
            current_time += timestep
        
        df = pd.DataFrame(records, columns=['id', 'timestep', 'start_node', 'end_node', 'position_on_edge', 'path'])
        df.to_csv(f'path_{idx + 1}.csv', index=False)

# Example usage
import pandas as pd

class Graph:
    def __init__(self):
        self.edges = []

    def add_edge(self, start, end, length, speed):
        self.edges.append((start, end, length, speed))

def load_graph_from_csv(filepath):
    df = pd.read_csv(filepath)
    graph = Graph()
    for index, row in df.iterrows():
        graph.add_edge(row['start'], row['end'], float(row['length']), float(row['speed']))
    return graph

# 文件路径
filepath = r"C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\simulation\graph_edges.csv"
graph = load_graph_from_csv(filepath)

# 打印输出以验证
for edge in graph.edges:
    print(edge)

graph = load_graph_from_csv(r"C:\Users\zhuye\Desktop\microBlooM-main\microBlooM-main\simulation\graph_edges.csv")
paths = bfs_paths(graph.graph, 'A')
generate_csv(graph.graph, paths, 0.1)  # timestep is set to 0.1 seconds
