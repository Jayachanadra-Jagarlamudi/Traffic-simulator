import numpy as np
from collections import deque
import heapq

np.random.seed(123)

class Vehicle:
    waiting_time = {}
    def __init__(self, source, destination, spawn_time):
        self.source = source
        self.destination = destination
        self.spawn_time = spawn_time

class Node:
    def __init__(self, node_addr, pos=(0,0), lambda_rate=0.0, is_sink=0):
        self.node_addr = node_addr
        self.pos = pos
        self.lambda_rate = lambda_rate  # For Poisson generation
        self.forwarding_table = {}      # {dest_node_addr: edge_object}
        self.receive_queue = deque()    # Vehicles arriving from edges
        self.internal_queue = deque()   # Vehicles waiting to be sent
        self.is_sink = is_sink          # Flag for if the node is a sink
        self.completed_vehicles = []    # Sink storage

    def assign_routing(self, forwarding_table):
        self.forwarding_table = forwarding_table
        
    def report_status(self):
        rec = [(vehicle.source, vehicle.destination) for vehicle in self.receive_queue]
        internal = [(vehicle.source, vehicle.destination) for vehicle in self.internal_queue]
        done = len(self.completed_vehicles)
        print(f"Node {self.node_addr}: {rec}, {internal}, {done}")

    def prime(self, current_time, all_nodes):
        # 1. Process Received Vehicles
        while self.receive_queue:
            v = self.receive_queue.popleft()
            if v.destination == self.node_addr:
                self.completed_vehicles.append(v)
                if (v.source, v.destination) not in Vehicle.waiting_time.keys():
                    Vehicle.waiting_time[(v.source,v.destination)] = [1, current_time - v.spawn_time]
                else:
                    Vehicle.waiting_time[(v.source,v.destination)][0] += 1
                    Vehicle.waiting_time[(v.source,v.destination)][1] += (current_time - v.spawn_time)
                    
            else:
                self.internal_queue.append(v)

        # 2. Generate New Vehicles (Poisson)
        if self.lambda_rate > 0:
            num_new = np.random.poisson(self.lambda_rate)
            for _ in range(num_new):
                dest = np.random.choice([n.node_addr for n in all_nodes if n.node_addr != self.node_addr and n.is_sink])
                #dest = 3
                self.internal_queue.append(Vehicle(self.node_addr, dest, current_time))

        # 3. Prepare for Outbound (Filling Edges)
        # We group by destination to respect link capacity
        to_remove = []
        for v in self.internal_queue:
            if v.destination not in self.forwarding_table.keys():
                edge = self.forwarding_table.get('def')
            else:
                edge = self.forwarding_table.get(v.destination)
                
            if edge and len(edge.transit_buffer) < edge.capacity:
                edge.transit_buffer.append(v)
                to_remove.append(v)
        
        for v in to_remove:
            self.internal_queue.remove(v)

class Edge:
    def __init__(self, start_node, end_node, capacity):
        self.start_node = start_node
        self.end_node = end_node
        self.capacity = capacity
        self.transit_buffer = []

    def transport(self):
        """Moves vehicles from transit to the destination node's queue."""
        for vehicle in self.transit_buffer:
            self.end_node.receive_queue.append(vehicle)
        self.transit_buffer = []

class TrafficGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_addr, pos=(0,0), lambda_rate=0.0, is_sink=0):
        node = Node(node_addr, pos, lambda_rate, is_sink)
        self.nodes[node_addr] = node
        return node

    def assign_routing(self, node_addr, forwarding_table):
        if node_addr not in self.nodes.keys():
            print(f"Address Error: {node_addr} is not in the network")
        self.nodes[node_addr].assign_routing(forwarding_table)

    def add_edge(self, u, v, capacity):
        edge = Edge(self.nodes[u], self.nodes[v], capacity)
        self.edges.append(edge)
        return edge

    def generate_routing_tables(self):
        """
        Computes the shortest path from every node to every potential sink node
        using Dijkstra's algorithm, where edge weight = 1/capacity.
        """
        # Identify all possible destinations (sinks)
        sinks = [node_id for node_id, node in self.nodes.items() if node.is_sink]
        
        # Pre-calculate adjacency list for Dijkstra: {start_node: [(end_node, weight, edge_obj), ...]}
        adj = {node_addr: [] for node_addr in self.nodes.keys()}
        for edge in self.edges:
            # Cost is defined as 1 / capacity
            weight = 1.0 / edge.capacity
            adj[edge.start_node.node_addr].append((edge.end_node.node_addr, weight, edge))

        for sink in sinks:
            # Run Dijkstra in "reverse" or simply run for each sink to find paths TO it
            # Here we find the best next-hop for every node to reach this specific sink
            distances = {node_addr: float('inf') for node_addr in self.nodes.keys()}
            distances[sink] = 0
            
            # We use a priority queue: (distance, current_node)
            pq = [(0, sink)]
            
            # To populate forwarding tables, it's actually easier to run Dijkstra 
            # from each node to see its best path to the sink.
            for start_node_addr in self.nodes.keys():
                if start_node_addr == sink:
                    continue
                    
                path_edge = self._dijkstra_next_hop(start_node_addr, sink, adj)
                if path_edge:
                    self.nodes[start_node_addr].forwarding_table[sink] = path_edge

    def _dijkstra_next_hop(self, start_addr, target_addr, adj):
        """Helper to find the specific edge used for the first hop of the best path."""
        distances = {node_addr: float('inf') for node_addr in self.nodes.keys()}
        distances[start_addr] = 0
        # Store: (total_cost, current_node, first_edge_taken)
        pq = [(0, start_addr, None)]
        
        while pq:
            d, u, first_edge = heapq.heappop(pq)

            if d > distances[u]:
                continue
            if u == target_addr:
                return first_edge

            for v, weight, edge_obj in adj[u]:
                new_dist = d + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    # Carry the first edge taken through the search
                    next_first_edge = first_edge if first_edge is not None else edge_obj
                    heapq.heappush(pq, (new_dist, v, next_first_edge))
        return None


    def step_prime(self, t):
        node_list = list(self.nodes.values())

        # Phase 1: Nodes Prime (Logic, Generation, Queueing)
        for node in node_list:
            node.prime(t, node_list)

    def step_transport(self, t, report=0):
        node_list = list(self.nodes.values())
        
        # Phase 2: Edges Transport (Physical Movement)
        for edge in self.edges:
            edge.transport()
        
        if report:
            print(f"After step {t}:")
            for node in node_list:
                node.report_status()

    def simulate(self, steps, report=0):
        for node in self.nodes.values():
            node.report_status()
    
        for t in range(steps):
            self.step_prime(t)
            self.step_transport(t,report)
            print(f"Step {t} complete.")
            
    def publish_results(self, steps):
        with open("time_results.txt",'w') as f:
            f.write("AVERAGE TRAVEL DURATION:\n")
            for key, vals in Vehicle.waiting_time.items():
                f.write(f"{key} : {round(vals[1]/vals[0],3)}\n")
                
            f.write("AVERAGE THROUGHPUT AT SINK:\n")
            for node in self.nodes.values():
                if node.is_sink:
                    f.write(f"{node.node_addr} : {round(len(node.completed_vehicles)/steps,3)}\n")
                
            f.write("AVERAGE THROUGHPUT AT SOURCES:\n")
            src_thrhput = {}
            for node in self.nodes.values():
                if node.is_sink:
                    for v in node.completed_vehicles:
                        if v.source in src_thrhput.keys():
                            src_thrhput[v.source] += 1
                        else:
                            src_thrhput[v.source] = 1
                            
            for key,val in src_thrhput.items():
                f.write(f"{key} : {val/steps}\n")