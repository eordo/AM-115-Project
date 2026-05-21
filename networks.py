import networkx as nx
from params import ALPHA, BETA


def build_grid_network():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(1, 10))

    cordon = {5}

    # Outer square.
    peripheral_edges = [
        (1,2), (2,3),
        (1,4), (4,7),
        (3,6), (6,9),
        (7,8), (8,9)
    ]
    for (u, v) in peripheral_edges:
        _add_car_edge(G, u, v, t0=2, K=10)
        _add_car_edge(G, v, u, t0=2, K=10)
    
    # Inner cross.
    radial_edges = [
        (2,5), (5,8),
        (4,5), (5,6)
    ]
    for (u, v) in radial_edges:
        _add_car_edge(G, u, v, t0=2, K=15)
        _add_car_edge(G, v, u, t0=2, K=15)
    
    # Cross-town transit line and feeder lines.
    transit_edges = [
        (1,4), (4,5), (5,6), (6,9) 
    ]
    for (u, v) in transit_edges:
        _add_transit_edge(G, u, v, t_bar=1.5)
        _add_transit_edge(G, v, u, t_bar=1.5)
    
    return G, cordon


def build_hub_network():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(1, 10))

    hub = 5
    outer = [1, 2, 3, 4, 6, 7, 8, 9]

    # Spokes.
    for n in outer:
        _add_car_edge(G, n, hub, t0=3, K=10)
        _add_car_edge(G, hub, n, t0=3, K=10)
    
    # Weak peripheral connections.
    weak_edges = [(1,2), (2,3), (7,8), (8,9)]
    for u, v in weak_edges:
        _add_car_edge(G, u, v, t0=2, K=8)
        _add_car_edge(G, v, u, t0=2, K=8)
    
    # Radial transit lines.
    for n in outer:
        _add_transit_edge(G, n, hub, t_bar=3)
        _add_transit_edge(G, hub, n, t_bar=3)
    
    return G, {hub}


def build_core_network():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(1, 10))

    core = [4, 5, 6, 7]

    # Dense core.
    for u in core:
        for v in core:
            if u != v:
                _add_car_edge(G, u, v, t0=1, K=12)

    # Periphery-to-core connections.
    connectors = [
        (1,6), (2,6), (3,4),    # Periphery A
        (9,5), (8,7), (8,4)     # Periphery B
    ]
    for u, v in connectors:
        _add_car_edge(G, u, v, t0=2, K=10)
        _add_car_edge(G, v, u, t0=2, K=10)
    
    # Peripheral connections.
    periphery_edges = [(1,2), (2,3), (8,9)]
    for u, v in periphery_edges:
        _add_car_edge(G, u, v, t0=2, K=8)
        _add_car_edge(G, v, u, t0=2, K=8)
    
    # Outer ring transit.
    transit_edges = [(1,2), (2,3), (3,4), (4,8), (8,9)]
    for u, v in transit_edges:
        _add_transit_edge(G, u, v, t_bar=1.5)
        _add_transit_edge(G, v, u, t_bar=1.5)
    
    return G, set(core)


def build_paths(G, cordon, od_pairs, car_cutoff=4, transit_cutoff=7):
    def enum(mode, o, d, cutoff):
        mg = nx.DiGraph()
        mg.add_edges_from((u, v) for u, v, k in G.edges(keys=True) if k == mode)
        if o not in mg or d not in mg:
            return []
        return [
            [(p[i], p[i+1]) for i in range(len(p)-1)]
            for p in nx.all_simple_paths(mg, o, d, cutoff=cutoff)
        ]
    paths = []
    for (o, d) in od_pairs:
        for i, edges in enumerate(enum('car', o, d, car_cutoff)):
            enters = any(u not in cordon and v in cordon for u, v in edges)
            path = _create_path(f'car_{o}_{d}_{i}', (o, d), 'car', edges, enters)
            paths.append(path)
        for i, edges in enumerate(enum('transit', o, d, transit_cutoff)):
            path = _create_path(f'transit_{o}_{d}_{i}', (o, d), 'transit', edges)
            paths.append(path)
    return paths


def get_cordon_edges(paths, cordon):
    cordon_edges = set()
    cordon_paths = [path for path in paths if path['enters_cordon']]
    for path in cordon_paths:
        for u, v in path['edges']:
            if u not in cordon and v in cordon:
                cordon_edges.add((u, v))
    return list(cordon_edges)


def _create_path(name, od, mode, edges, enters_cordon=False):
    return {
        'name': name,
        'od': od,
        'mode': mode,
        'edges': edges,
        'enters_cordon': enters_cordon
    }


def _add_car_edge(G, u, v, t0=1.0, K=1.0):
    G.add_edge(u, v, key='car', t0=t0, K=K, alpha=ALPHA, beta=BETA)


def _add_transit_edge(G, u, v, t_bar=1.0):
    G.add_edge(u, v, key='transit', t_bar=t_bar)
