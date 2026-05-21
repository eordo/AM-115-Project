import numpy as np
from collections import defaultdict
from params import KAPPA, THETA


def solve_sue(G, paths, od_pairs, demands,
              theta=THETA, kappa=KAPPA, tau_z=0.0,
              max_iter=2000, tol=1e-6, damping=1.0):
    od_index = defaultdict(list)
    for i, p in enumerate(paths):
        od_index[p['od']].append(i)

    x = _initialize_edge_flows(G)
    f = np.zeros(len(paths))

    for it in range(max_iter):
        # Evaluate all path costs at current flows.
        costs = np.array([_path_cost(G, p, x, kappa, tau_z) for p in paths])
        # Assign logits independently per OD pair.
        f_new = np.zeros(len(paths))
        for od in od_pairs:
            idx = od_index[od]
            if not idx:
                continue
            c = costs[idx]
            log_w = -theta * c
            log_w -= log_w.max()
            w = np.exp(log_w)
            f_new[idx] = demands[od] * w / w.sum()
        # Aggregate edge flows from new path flows.
        x_new = _compute_edge_flows(G, paths, f_new)
        # Check for convergence.
        if damping < 1.0:
            for e in x['car']:
                x_new['car'][e] = (damping * x_new['car'][e] 
                                   + (1 - damping) * x['car'][e])
            for e in x['transit']:
                x_new['transit'][e] = (damping * x_new['transit'][e] 
                                       + (1 - damping) * x['transit'][e])
        delta = max(
            np.abs([[x_new['car'][e] - x['car'][e] for e in x['car']]]).max(),
            np.abs([[x_new['transit'][e] - x['transit'][e] for e in x['transit']]]).max()
        )
        f = f_new
        x = x_new
        if delta < tol:
            print(f"Converged after {it+1:,} iterations.")
            return x, f
    else:
        print(f"Warning: Did not converge after {max_iter:,} iterations.")
        return x, f


def solve_sue_first_best(G, paths, od_pairs, demands, x_unpriced,
                         theta=THETA, kappa=KAPPA,
                         max_iter=2000, tol=1e-6, damping=1.0):
    tolls = _compute_pigouvian_tolls(G, x_unpriced)
    for (u, v), tau in tolls.items():
        G[u][v]['car']['toll'] = tau
    
    x, f = solve_sue(G, paths, od_pairs, demands,
                     theta=theta, kappa=kappa,
                     max_iter=max_iter, tol=tol, damping=damping)

    for (u, v) in tolls:
        G[u][v]['car'].pop('toll', None)
    
    return x, f, tolls


def total_social_cost(G, x):
    car_cost = 0.0
    for (u, v), xe in x['car'].items():
        data = G[u][v]['car']
        car_cost += xe * _car_edge_cost(
            xe=xe,
            t0=data['t0'],
            K=data['K'],
            alpha=data['alpha'],
            beta=data['beta']
        )
    transit_cost = 0.0
    for (u, v), xe in x['transit'].items():
        data = G[u][v]['transit']
        transit_cost += xe * _transit_edge_cost(data['t_bar'])
    
    return float(car_cost + transit_cost)


def compute_mode_share(x):
    agg_car_flow = np.sum(list(x['car'].values()))
    agg_transit_flow = np.sum(list(x['transit'].values()))
    agg_flow = agg_car_flow + agg_transit_flow
    return {
        'car': float(agg_car_flow / agg_flow),
        'transit': float(agg_transit_flow / agg_flow)
    }


def _compute_pigouvian_tolls(G, x):
    tolls = {}
    for (u, v), xe in x['car'].items():
        data = G[u][v]['car']
        t0, K = data['t0'], data['K']
        alpha, beta = data['alpha'], data['beta']
        deriv = t0 * alpha * beta / K * (xe / K)**(beta - 1)
        tolls[(u,v)] = xe * deriv
    return tolls


def compute_cordon_toll(tolls, cordon_edges):
    cordon_tolls = [tolls[e] for e in cordon_edges if e in tolls]
    return float(np.mean(cordon_tolls)) if cordon_tolls else 0.0


def _path_cost(G, path, x, kappa=KAPPA, tau_z=0.0):
    if path['mode'] == 'car':
        cost = _car_path_cost(G, path, x['car'])
        cost += tau_z * path['enters_cordon']
    else:
        cost = _transit_path_cost(G, path, x['transit'])
    return cost


def _car_path_cost(G, path, x_car):
    assert path['mode'] == 'car', "Incorrect path type"
    cost = 0.0
    for (u, v) in path['edges']:
        data = G[u][v]['car']
        cost += _car_edge_cost(
            xe=x_car[(u, v)],
            t0=data['t0'],
            K=data['K'],
            alpha=data['alpha'],
            beta=data['beta'],
            toll=data.get('toll', 0.0)
        )
    return cost


def _transit_path_cost(G, path, x_transit, kappa=KAPPA):
    assert path['mode'] == 'transit', "Incorrect path type"
    cost = kappa
    for (u, v) in path['edges']:
        cost += _transit_edge_cost(G[u][v]['transit']['t_bar'])
    return cost


def _car_edge_cost(xe, t0, K, alpha, beta, toll=0.0):
    return t0 * (1 + alpha * (xe / K)**beta) + toll


def _transit_edge_cost(t_bar):
    return t_bar


def _compute_edge_flows(G, paths, f):
    x = _initialize_edge_flows(G)
    for path, flow in zip(paths, f):
        for e in path['edges']:
            x[path['mode']][e] += flow
    return x


def _initialize_edge_flows(G):
    return {
        'car': {
            (u, v): 0.0 for u, v, k in G.edges(keys=True) if k == 'car'
        },
        'transit': {
            (u, v): 0.0 for u, v, k in G.edges(keys=True) if k == 'transit'
        }
    }
