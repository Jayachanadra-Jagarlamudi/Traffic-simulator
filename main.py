import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec
from traffic_classes import TrafficGraph


def generate_dynamic_gif(sim, steps, filename="traffic_simulation.gif"):
    """
    Generates a GIF visualization of the discrete-time traffic flow.
    Includes directional arrows and a color legend for vehicle destinations.
    """
    
    # 1. Setup the figure and multi-panel layout (similar to image_0.png structure)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 10]) # Top for phase indicator, bottom for graph
    
    # Text Axis (Shows current phase/time)
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')
    phase_text = ax_text.text(0.5, 0.5, '', transform=ax_text.transAxes,
                              ha='center', va='center', fontsize=18, fontweight='bold')

    # Main Graph Axis
    ax = fig.add_subplot(gs[1])
    ax.set_aspect('equal')
    
    # Define distinct colors for destinations using a colormap
    destination_ids = [n_id for n_id, node in sim.nodes.items() if node.is_sink]
    cmap = plt.cm.get_cmap('gist_rainbow', len(destination_ids))
    # Map {dest_id: (R, G, B, A)}
    dest_colors = {d_id: cmap(i) for i, d_id in enumerate(destination_ids)}

    # Ensure nodes have set positions (x, y) for plotting
    node_positions = {n_id: node.pos for n_id, node in sim.nodes.items()}

    def update_frame(frame):
        print(f"Drawing frame {frame}")
        ax.clear()
        current_time = frame // 2
        is_prime_phase = (frame % 2 == 0)
        
        # Update text display
        phase_name = "PRIME Phase" if is_prime_phase else "TRANSPORT Phase"
        phase_text.set_text(f"Timestep T={current_time} | {phase_name}")
        if is_prime_phase:
            phase_text.set_color('blue')
        else:
            phase_text.set_color('darkred')

        # Run one logic half-step
        if is_prime_phase:
            sim.step_prime(current_time)
        else:
            sim.step_transport(current_time)

        # 1. Draw Directional Edges (Arrows)
        for edge in sim.edges:
            start_pos = np.array(node_positions[edge.start_node.node_addr])
            end_pos = np.array(node_positions[edge.end_node.node_addr])
            
            # Create the arrow
            arrow = patches.FancyArrowPatch(
                start_pos, 
                end_pos,
                connectionstyle="arc3,rad=0", # Straight line
                arrowstyle='-|>',             # Solid head arrow
                mutation_scale=20,            # Size of the arrow head
                linewidth=2,
                color='gray',
                alpha=0.3,
                zorder=1,                     # Below vehicles and nodes
                shrinkA=15,                   # Shrink start point away from node center
                shrinkB=15                    # Shrink end point so head is visible
            )
            ax.add_patch(arrow)

        # 2. Draw Nodes
        for n_id, pos in node_positions.items():
            node_color = 'skyblue' if not sim.nodes[n_id].is_sink else 'lightgreen'
            circle = plt.Circle(pos, 0.3, color=node_color, ec='black', lw=2, zorder=3)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], n_id, ha='center', va='center', fontweight='bold', fontsize=14)

        # 3. Draw Vehicles (and update arrows during Transport)
        # Vehicles in Nodes' queues (small squares)
        for n_id, node in sim.nodes.items():
            n_x, n_y = node_positions[n_id]
            for i, v in enumerate(node.internal_queue):
                ax.plot(n_x, n_y + 0.4 + (i * 0.1), 's', ms=6, 
                        color=dest_colors[v.destination], markeredgecolor='black')
        
        # Vehicles in Edges' transit buffers (circles moving along arrows)
        for edge in sim.edges:
            num_v = len(edge.transit_buffer)
            if num_v > 0:
                s_pos = node_positions[edge.start_node.node_addr]
                e_pos = node_positions[edge.end_node.node_addr]
                
                # Highlight active edge during Transport phase
                if not is_prime_phase:
                    # Optional: Color the arrow by the dominant destination color
                    dominant_dest = edge.transit_buffer[0].destination # Simple approximation
                    edge_color = dest_colors[dominant_dest]
                    arrow_highlight = patches.FancyArrowPatch(s_pos, e_pos,
                                                             arrowstyle='-|>', mutation_scale=20,
                                                             lw=4, color=edge_color, alpha=0.8,
                                                             connectionstyle="arc3,rad=0.0")
                    ax.add_patch(arrow_highlight)

                for i, v in enumerate(edge.transit_buffer):
                    # Linearly interpolate position between nodes (t in [0, 1])
                    # Note: vehicles move at t=1 during transport, visualize mid-way
                    # For visualization of *movement*, we slightly shift them
                    viz_t = 0.5 if is_prime_phase else 0.8
                    # Distribute vehicles slightly along the edge
                    v_offset = (i+1) / (num_v + 1) * 0.1 
                    
                    v_x = s_pos[0] + (e_pos[0] - s_pos[0]) * (viz_t + v_offset)
                    v_y = s_pos[1] + (e_pos[1] - s_pos[1]) * (viz_t + v_offset)
                    
                    ax.plot(v_x, v_y, 'o', ms=10, color=dest_colors[v.destination], 
                            markeredgecolor='black', zorder=4)

        # 4. Generate the Legend (Crucial requirement)
        # Map dest_id to a proxy object for the legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'To: {d_id}',
                                      markerfacecolor=dest_colors[d_id], markersize=12,
                                      markeredgecolor='black')
                           for d_id in destination_ids]
        
        ax.legend(handles=legend_elements, loc='upper right', title="Vehicle Destinations",
                  fontsize=12, title_fontsize=14, framealpha=1, edgecolor='black')

        # Clean axis
        ax.axis('off')
        ax.set_xlim(-1, 5) # Set fixed viewing area
        ax.set_ylim(-1, 5)

    # Note: Interval 1000ms = 1 sec per half-step
    ani = animation.FuncAnimation(fig, update_frame, frames=steps*2, interval=1000)
    ani.save(filename, writer='pillow')
    print(f"Generated {filename}")
    
    
# Initialize Graph
sim = TrafficGraph()
sim.add_node(1.1, pos=(2,0), lambda_rate=0.0, is_sink=1) 
sim.add_node(1.2, pos=(1,0), lambda_rate=0.7, is_sink=0) 
sim.add_node(1.3, pos=(0,0), lambda_rate=0.0, is_sink=1) 
sim.add_node(2.1, pos=(1,1), lambda_rate=0.0, is_sink=0) 
sim.add_node(2.2, pos=(2,1), lambda_rate=0.0, is_sink=0) 
sim.add_node(2.3, pos=(0,1), lambda_rate=0.0, is_sink=0) 
sim.add_node(3.1, pos=(1,2), lambda_rate=0.8, is_sink=0) 
sim.add_node(3.2, pos=(2,2), lambda_rate=0.0, is_sink=0) 
sim.add_node(3.3, pos=(0,2), lambda_rate=0.0, is_sink=1) 

# Add Edges
e1 = sim.add_edge(1.1, 2.1, capacity=1)
e2 = sim.add_edge(1.2, 2.1, capacity=1)
e3 = sim.add_edge(2.1, 1.3, capacity=1)
e4 = sim.add_edge(2.1, 3.1, capacity=2)
e5 = sim.add_edge(2.3, 2.1, capacity=2)
e6 = sim.add_edge(2.2, 1.1, capacity=1)
e7 = sim.add_edge(3.3, 2.3, capacity=1)
e8 = sim.add_edge(3.2, 2.2, capacity=1)
e9 = sim.add_edge(3.1, 3.2, capacity=1)
e10 = sim.add_edge(3.1, 3.3, capacity=1)

# Manual Forwarding Table (A -> C goes through B)
#sim.assign_routing(1.1, {'def': e1})
#sim.assign_routing(1.2, {'def': e2})
#sim.assign_routing(2.1, {1.3:e3, 'def': e4})
#sim.assign_routing(2.2, {'def': e6})
#sim.assign_routing(2.3, {'def': e5})
#sim.assign_routing(3.1, {1.1:e9, 1.3: e10, 3.3: e10})
#sim.assign_routing(3.2, {'def': e8})
#sim.assign_routing(3.3, {'def': e7})

sim.generate_routing_tables()

for node in sim.nodes.values():
    tab = {addr:ed.end_node.node_addr for addr, ed in node.forwarding_table.items()}
    print(f"{node.node_addr}:{tab}")

# Run for 10 steps
#generate_dynamic_gif(sim, steps=20)
sim.simulate(steps=20)
sim.publish_results(steps=20)

#print(f"Vehicles arrived at C: {len(sim.nodes[3.1].completed_vehicles)}")