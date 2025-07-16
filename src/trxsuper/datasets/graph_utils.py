import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline


def add_edge_lengths(graph):
    """
    Add the length of each edge to the graph.

    Parameters:
    - graph: networkx.DiGraph with 'position' attribute for each node.

    Returns:
    - networkx.DiGraph with 'length' attribute for each edge.
    """
    for u, v in graph.edges:
        pos_u = np.array(graph.nodes[u]['position'])
        pos_v = np.array(graph.nodes[v]['position'])
        graph.nodes[v]['edge_length'] = np.linalg.norm(pos_u - pos_v)

    if 'edge_length' not in graph.nodes['0-0']:
        graph.nodes['0-0']['edge_length'] = 0

    return graph


def sample_equidistant_points(positions, radii, n_points=None, points_dist=None,
                              smooth=True, smooth_override=True):
    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    avg_segment_distance = np.mean(segment_distances)
    if (segment_distances[0] > 2 * avg_segment_distance
            or segment_distances[-1] > 2 * avg_segment_distance) and smooth_override:
        smooth = False

    if len(positions) > 2 and smooth:
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_distances)])
        total_length = cumulative_distances[-1]
        if points_dist is not None:
            n_points = max(round(total_length / points_dist) + 1, 2)
        assert n_points is not None, "Either n_points or points_dist should be provided"
        target_distances = np.linspace(0, total_length, n_points)
        splines_positions = [CubicSpline(cumulative_distances, positions[:, i]) for i in range(3)]
        positions = np.column_stack([spline(target_distances) for spline in splines_positions])
        radii = np.interp(target_distances, cumulative_distances, radii)

    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_distances)])

    total_length = cumulative_distances[-1]
    if points_dist is not None:
        n_points = max(round(total_length / points_dist) + 1, 2)
    assert n_points is not None, "Either n_points or points_dist should be provided"
    target_distances = np.linspace(0, total_length, n_points)

    sampled_positions = np.empty((n_points, 3))
    for i in range(3):
        sampled_positions[:, i] = np.interp(target_distances, cumulative_distances, positions[:, i])
    sampled_radii = np.interp(target_distances, cumulative_distances, radii)

    return sampled_positions, sampled_radii


def rename_node_names(G, root):
    renamed_tree = nx.DiGraph()
    st_br_id = 0
    old_node_id_to_new = {}
    for level in nx.bfs_layers(G, root):
        for node in level:
            if node == root:
                node_name = str(st_br_id) + "-" + str(0)
                renamed_tree.add_node(node_name, **G.nodes[node])
                old_node_id_to_new[node] = node_name
            else:
                parent = list(G.predecessors(node))[0]
                parent_name = old_node_id_to_new[parent]
                if G.out_degree(parent) > 1:
                    st_br_id += 1
                    node_name = str(st_br_id) + "-" + str(0)
                    renamed_tree.add_node(node_name, **G.nodes[node])
                    old_node_id_to_new[node] = node_name
                else:
                    parent_br = int(parent_name.split("-")[0])
                    parent_pt = int(parent_name.split("-")[1])
                    node_name = str(parent_br) + "-" + str(parent_pt + 1)
                    renamed_tree.add_node(node_name, **G.nodes[node])
                    old_node_id_to_new[node] = node_name
                renamed_tree.add_edge(parent_name, node_name)

    return renamed_tree


def merge_remove_small_branches(G, min_length=0.5, root='0-0'):
    """
    Merge small intermediate branches to the parent branch
    """
    init_num_nodes = len(G.nodes)
    branches = get_branches(G)
    for branch in branches:
        start_node, end_node = branches[branch]
        if start_node not in G.nodes or end_node not in G.nodes:
            continue

        if start_node == end_node:
            assert start_node == end_node == root, "There should be no self loops in the graph"
            continue

        shortest_path = nx.shortest_path(G, start_node, end_node)
        branch_length = sum([G.nodes[node]["edge_length"] for node in shortest_path[1:]])
        if branch_length < min_length:
            if G.out_degree(end_node) == 0:
                print(f"Removing end branch: {shortest_path[1:]} with length {branch_length}")
                G.remove_nodes_from(shortest_path[1:])
                continue
            else:
                successors = list(G.successors(end_node))
                for succ in successors:
                    succ_pos = G.nodes[succ]["position"]
                    start_pos = G.nodes[start_node]["position"]
                    edge_length = np.linalg.norm(succ_pos - start_pos)
                    G.add_edge(succ, start_node)
                    G.nodes[succ]["edge_length"] = edge_length
                print(f"Removing intermediate branch: {shortest_path[1:]} with length {branch_length}")
                G.remove_nodes_from(shortest_path[1:])
                continue
    G = rename_node_names(G, root)
    final_num_nodes = len(G.nodes)
    if init_num_nodes != final_num_nodes:
        G = merge_remove_small_branches(G, min_length, root)

    return G


def merge_remove_small_edges(G, min_length=0.5):
    """
    Merge small edges in the graph G based on the specified minimum length.
    """
    init_num_nodes = len(G.nodes)
    edges = list(G.edges)

    for u, v in edges:
        if u not in G.nodes or v not in G.nodes:
            continue

        edge_length = G.nodes[v]["edge_length"]

        if edge_length >= min_length:
            continue

        grandparent_nodes = list(G.predecessors(u))
        children = list(G.successors(v))

        if len(grandparent_nodes):
            grandparent = grandparent_nodes[0]  # Assuming a single grandparent
            parent_edge_length = G.nodes[u]["edge_length"]

            if edge_length + parent_edge_length < 1.5:
                G.add_edge(grandparent, v)
                G.nodes[v]["edge_length"] = edge_length + parent_edge_length
                G.remove_node(u)
                print(f"Merged edge {u} -> {v} with length {edge_length}")
                continue

        if len(children):
            children_edge_lengths = np.array([G.nodes[child]["edge_length"] for child in children])
            merged_edge_lengths = children_edge_lengths + edge_length

            if (merged_edge_lengths < 1.5).all():
                for c_i, child in enumerate(children):
                    G.add_edge(u, child)
                    G.nodes[child]["edge_length"] = merged_edge_lengths[c_i]
                G.remove_node(v)
                print(f"Merged edge {u} -> {v} with length {edge_length}")
                continue

        print(f"Edge {u} -> {v} with length {edge_length} could not be merged")
    final_num_nodes = len(G.nodes)
    if init_num_nodes != final_num_nodes:
        G = merge_remove_small_edges(G, min_length)

    return G


def get_branches(G):
    nodes = list(G.nodes)
    branch_names = list(set([node.split("-")[0] for node in nodes]))
    branch_start_end_points = {}
    for branch in branch_names:
        start_node_name = branch + "-0"
        parent_node = list(G.predecessors(start_node_name))
        if len(parent_node) == 0:
            parent_node = branch + "-0"
        else:
            parent_node = parent_node[0]
        branch_nodes = [node for node in nodes if node.split('-')[0] == branch]
        max_point_id = max([int(node.split("-")[1]) for node in branch_nodes])
        end_node_name = branch + "-" + str(max_point_id)
        branch_start_end_points[branch] = (parent_node, end_node_name)
    return branch_start_end_points


def get_resampled_graph(G, n_points=None, points_dist=None, smooth=True, remove_small=True):
    """
    Resample the graph to have equidistant points
    """
    G_resampled = nx.DiGraph()
    if remove_small:
        G = merge_remove_small_branches(G, min_length=0.5, root='0-0')

    branches = get_branches(G)
    for branch in branches:
        start_node, end_node = branches[branch]
        if start_node == end_node:
            assert start_node == end_node == "0-0", "There should be no self loops in the graph"
            G_resampled.add_node("0-0", position=G.nodes["0-0"]["position"],
                                 radius=G.nodes["0-0"]["radius"])
            continue

        shortest_path = nx.shortest_path(G, start_node, end_node)
        positions = np.array([G.nodes[node]["position"] for node in shortest_path])
        radii = np.array([G.nodes[node]["radius"] for node in shortest_path])

        if len(positions) > 1:
            resampled_positions, resampled_radii = sample_equidistant_points(positions, radii,
                                                                             n_points=n_points, points_dist=points_dist, smooth=smooth)
        else:
            raise ValueError("Branch has less than 2 points!")

        if branch == '0':
            resampled_positions = np.concatenate([np.array([[0.0, 0.0, 0.0]]), resampled_positions])
            resampled_radii = np.concatenate([np.array([0]), resampled_radii])

        point_id = 0
        G_resampled.add_node(branch + '-' + str(point_id),
                             position=resampled_positions[1],
                             radius=resampled_radii[1])
        for (position, radius) in zip(resampled_positions[2:], resampled_radii[2:]):
            point_id += 1
            node_name = branch + "-" + str(point_id)
            parent_pos = G_resampled.nodes[branch + "-" + str(point_id - 1)]["position"]
            edge_length = np.linalg.norm(position - parent_pos)
            G_resampled.add_node(node_name, position=position, radius=radius,
                                 edge_length=edge_length)
            G_resampled.add_edge(branch + "-" + str(point_id - 1), node_name)

    # connect the branches
    new_branches = get_branches(G_resampled)
    for branch in new_branches:
        if branch == '0':
            continue
        parent_br = branches[branch][0].split("-")[0]
        parent_end = new_branches[parent_br][1]
        branch_start = branch + "-0"
        parent_pos = G_resampled.nodes[parent_end]["position"]
        branch_start_pos = G_resampled.nodes[branch_start]["position"]
        edge_length = np.linalg.norm(branch_start_pos - parent_pos)
        G_resampled.add_edge(parent_end, branch_start)
        G_resampled.nodes[branch_start]["edge_length"] = edge_length

    G_resampled.nodes['0-0']["edge_length"] = 0
    if remove_small:
        G_resampled = merge_remove_small_edges(G_resampled, min_length=0.5)

    return G_resampled
