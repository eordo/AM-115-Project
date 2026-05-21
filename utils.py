import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_network(G, pos, node_colors,
                 ax=None,
                 transit_offset=0.01,
                 node_size=800,
                 title=None):
    def draw_offset_edge(ax, p1, p2, offset, **style):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0:
            return
        perp = np.array([-dy, dx]) / length
        shift = perp * offset
        ax.plot(
            [p1[0] + shift[0], p2[0] + shift[0]],
            [p1[1] + shift[1], p2[1] + shift[1]],
            **style
        )
    
    if (show_plot := ax is None):
        _, ax = plt.subplots()

    offset_map = {
        ('car', True): +transit_offset,
        ('transit', True): -transit_offset,
        ('car', False): 0.0,
        ('transit', False): 0.0
    }

    car_edges = [(u, v) for u, v, k in G.edges(keys=True) if k == 'car']
    transit_edges = [(u, v) for u, v, k in G.edges(keys=True) if k == 'transit']
    parallel_edges = set(car_edges).intersection(set(transit_edges))

    edge_styles = {
        'car': dict(color='black', linestyle='solid'),
        'transit': dict(color='black', linestyle='dashed')
    }

    seen = set()
    car_deduped = []
    for u, v, k in G.edges(keys=True):
        if k != 'car':
            continue
        key = frozenset((u, v))
        if key not in seen:
            seen.add(key)
            car_deduped.append((u, v, k))

    for key in ('car', 'transit'):
        edge_iter = (
            car_deduped
            if key == 'car'
            else [(u, v, k) for u, v, k in G.edges(keys=True) if k == 'transit']
        )
        for u, v, k in edge_iter:
            is_parallel = (u, v) in parallel_edges
            offset = offset_map[(key, is_parallel)]
            draw_offset_edge(ax, pos[u], pos[v], offset, **edge_styles[k])

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        ax=ax
    )
    nx.draw_networkx_labels(
        G, pos,
        font_color='white',
        font_weight='bold',
        ax=ax
    )

    if title is not None:
        ax.set_title(title)

    ax.axis('off')
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_flows(G, pos, x, cordon,
               ax=None, title=None, node_size=600,
               transit_offset=0.025, car_offset=0.025,
               cmap_name='YlOrRd', show_colorbar=True):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = matplotlib.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.5)

    car_edges     = {(u, v) for u, v, k in G.edges(keys=True) if k == 'car'}
    transit_edges = {(u, v) for u, v, k in G.edges(keys=True) if k == 'transit'}
    parallel      = car_edges & transit_edges

    def _shifted(p1, p2, offset):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0:
            return p1, p2
        perp = np.array([-dy, dx]) / length * offset
        return (p1[0]+perp[0], p1[1]+perp[1]), (p2[0]+perp[0], p2[1]+perp[1])

    # Transit edges.
    seen = set()
    for (u, v) in transit_edges:
        key = frozenset((u, v))
        if key in seen:
            continue
        seen.add(key)
        p1, p2 = np.array(pos[u]), np.array(pos[v])
        if (u, v) in parallel:
            p1, p2 = _shifted(p1, p2, -transit_offset)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#999999', linestyle='--', linewidth=1.2,
                zorder=1, solid_capstyle='round')

    # Car edges.
    seen = set()
    for (u, v) in car_edges:
        key = frozenset((u, v))
        if key in seen:
            continue
        seen.add(key)
        K          = G[u][v]['car']['K']
        xe_uv      = x['car'].get((u, v), 0.0)
        xe_vu      = x['car'].get((v, u), 0.0)
        saturation = (xe_uv + xe_vu) / (2 * K)
        color      = cmap(norm(saturation))
        linewidth  = 1.5 + 4.0 * min(saturation, 1.5) / 1.5
        p1, p2 = np.array(pos[u]), np.array(pos[v])
        if (u, v) in parallel:
            p1, p2 = _shifted(p1, p2, car_offset)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=linewidth,
                zorder=2, solid_capstyle='round')

    # Nodes.
    node_list  = list(G.nodes())
    node_color = ['firebrick' if n in cordon else 'steelblue' for n in node_list]
    for n, c in zip(node_list, node_color):
        ax.scatter(*pos[n], s=node_size, color=c, zorder=3)
    for n in node_list:
        ax.text(pos[n][0], pos[n][1], str(n),
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=9, zorder=4)

    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label('Road saturation  $x_e / K_e$', fontsize=8)
        cbar.set_ticks([0, 0.5, 1.0, 1.5])
        cbar.set_ticklabels(['0', '0.5', '1.0', '≥1.5'])

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if standalone:
        plt.tight_layout()
        plt.show()


def node_colors(G, cordon, base_color='steelblue', cordon_color='firebrick'):
    return [
        cordon_color if node in cordon else base_color
        for node in G.nodes()
    ]
