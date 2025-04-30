import pandas as pd
import math
from collections import defaultdict, deque
import csv
from tqdm import tqdm

# Define the Graph class
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, length):
        self.graph[u].append((v, length))

# Function to read points from a CSV file and create a graph
def read_points_from_csv(file_path):
    df = pd.read_csv(file_path)
    points = df[['id', 'x', 'y', 'z']].values.tolist()
    edges = []
    for idx, row in df.iterrows():
        current_id = row['id']
        next_ids = str(row['next_ids']).split(',')
        for next_id in next_ids:
            next_id = next_id.strip()
            if next_id and next_id.lower() != 'nan':  # Check if next_id is not empty or 'NaN'
                next_id = int(float(next_id))  # Convert to integer
                current_point = df[df['id'] == current_id].iloc[0]
                next_point = df[df['id'] == next_id].iloc[0]
                distance = calculate_distance((current_point['x'], current_point['y'], current_point['z']),
                                              (next_point['x'], next_point['y'], next_point['z']))
                edges.append((current_id, next_id, distance))
    return points, edges

# Function to calculate the distance between two 3D points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# Function to create a graph from the list of points and edges
def create_graph(edges):
    g = Graph()
    for u, v, length in edges:
        g.add_edge(u, v, length)
    return g

# Function to simulate movement of points and record their positions
def move_points_and_record(graph, points, start, speed, timestep=1):
    points_queue = deque([(start, 0, None, 0)])  # (current node id, time elapsed, previous node id, distance traveled on current edge)
    record = []
    current_time = 0

    total_edges = sum(len(edges) for edges in graph.values())  # Total number of edges to process
    pbar = tqdm(total=total_edges, desc="Processing Edges")  # Initialize progress bar

    while points_queue:
        current_node, arrival_time, previous_node, distance_traveled = points_queue.popleft()

        for neighbor, length in graph[current_node]:
            if neighbor != previous_node:  # Prevent going back on the same edge immediately
                time_to_travel = length / speed
                points_queue.append((neighbor, arrival_time + time_to_travel, current_node, 0))

        while distance_traveled < length:
            if current_time >= arrival_time:
                break
            record.append({
                'time': current_time,
                'start_node': current_node,
                'end_node': neighbor,
                'distance_on_edge': distance_traveled,
                'position': (
                    next(p[1] for p in points if p[0] == current_node) + (next(p[1] for p in points if p[0] == neighbor) - next(p[1] for p in points if p[0] == current_node)) * (distance_traveled / length),
                    next(p[2] for p in points if p[0] == current_node) + (next(p[2] for p in points if p[0] == neighbor) - next(p[2] for p in points if p[0] == current_node)) * (distance_traveled / length),
                    next(p[3] for p in points if p[0] == current_node) + (next(p[3] for p in points if p[0] == neighbor) - next(p[3] for p in points if p[0] == current_node)) * (distance_traveled / length)
                )
            })
            distance_traveled += speed * timestep
            current_time += timestep

        pbar.update(1)  # Update progress bar for each processed edge

    pbar.close()  # Close progress bar
    return record

# Function to write the recorded positions to a CSV file
def write_record_to_csv(record, output_file_path):
    with open(output_file_path, 'w', newline='') as file:  # Corrected unmatched parenthesis
        fieldnames = ['time', 'start_node', 'end_node', 'distance_on_edge', 'position_x', 'position_y', 'position_z']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in record:
            writer.writerow({
                'time': entry['time'],
                'start_node': entry['start_node'],
                'end_node': entry['end_node'],
                'distance_on_edge': entry['distance_on_edge'],
                'position_x': entry['position'][0],
                'position_y': entry['position'][1],
                'position_z': entry['position'][2]
            })

# Main execution
input_file_path = 'input_points.csv'  # Path to your input CSV file
output_file_path = 'output_positions.csv'  # Path to the output CSV file
speed = 0.5  # Speed of movement in units per second
start_node = 1  # Starting node id
timestep = 0.5  # Timestep in seconds

points, edges = read_points_from_csv(input_file_path)
graph = create_graph(edges)
record = move_points_and_record(graph.graph, points, start_node, speed, timestep)
write_record_to_csv(record, output_file_path)
