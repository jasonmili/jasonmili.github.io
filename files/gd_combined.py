import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.stats import entropy
from scipy.linalg import sinm, cosm, logm
from scipy.linalg import expm
from copy import deepcopy
import networkx as nx 
import pydtmc 
import functools as ft
import itertools
from collections import deque, Counter
from time import time, sleep
from tqdm import tqdm
import cProfile
def computePayoffs(i: int, mixed_profile: np.ndarray, i_utility_tensor: np.ndarray) -> np.ndarray: 
    after_i = ft.reduce(np.dot, [i_utility_tensor, *mixed_profile[i:][::-1]]) 
    return ft.reduce(np.dot, [after_i.transpose(), *mixed_profile[:i-1]]) 
def second_max(arr):
    return np.partition(arr, -2)[-2]
def getRandTrajInit(num_players, game_size, _range=[0.2, 0.8]):
    upper_limit = _range[1]
    lower_limit = _range[0]
    v = np.random.rand(num_players, game_size) * (upper_limit - lower_limit) + lower_limit
    v_hat = v / np.sum(v, axis=1, keepdims=True)
    return np.array([v_hat, v_hat]) 
def random_point_in_ball(dim, x0, radius):
    direction = np.random.randn(dim)
    direction /= np.linalg.norm(direction)
    radius = np.random.uniform(0, radius)
    return x0 + radius * direction
def sample_simplex(k):
    return tuple(np.random.dirichlet((1,)*k))
def sample_uniform(players, strategies, samples=1):
    points = -np.ones((samples, players, strategies))
    for i in range(samples):
        points[i] = np.array([sample_simplex(strategies) for _ in range(players)])
    return points
def sample_around_corner(pure_profile, epsilon, players, strategies, samples=1):
    points = np.array([epsilon/2 * np.reshape( sample_simplex((strategies-1)*players) , (players,strategies-1) ) for _ in range(samples)])
    points = np.pad(points, tuple([(0, 0)] * (points.ndim - 1)) + ((0, 1),))
    for i in range(samples):
        for k in range(players):
            points[i, k] = np.insert(points[i, k, :-1], pure_profile[k]-1, 1.0-np.sum(points[i, k]), axis=0)
    return points
def ReplicatorVectorField(points, u): 
    num_players = points.shape[1]
    new_points = np.zeros_like(points)
    for k, point_k in enumerate(points):
        for i in range(num_players):
            all_is_payoffs = computePayoffs(i+1, point_k, u[i])
            new_points[k][i] = np.multiply(point_k[i], all_is_payoffs - np.dot( point_k[i] , all_is_payoffs ))
    return new_points
def MWUUpdateGeneral(traj, eta, u):
    new_point = np.zeros(traj[0].shape)
    num_players = traj.shape[1]
    for i in range(num_players): 
        all_is_payoffs = computePayoffs(i+1, traj[-1], u[i]) 
        new_point[i] = np.multiply(traj[-1][i], np.exp(eta*all_is_payoffs) / np.dot( traj[-1][i] , np.exp(eta*all_is_payoffs) ))
    return new_point
def MWUGeneral(T, init, u, decrease = 'E', eta=0.1, exponent=0.5):
    traj = init
    for t in np.arange(1, T+1):
        if decrease == 'C':
            eta = eta
        elif decrease == 'E':
            eta = 1/t**exponent
        elif decrease == 'S':
            eta = ((init.shape[1]-1)**(-1/2))*(t**(-1/4))
        elif decrease == 'D':
            if t > 10:
                eta = 1/((np.log(t))**4)
            else:
                eta = 1
        new_point = MWUUpdateGeneral(traj, eta, u)
        traj = np.append(traj, [new_point], axis=0)
    return traj
def noisyMWUGeneral(T, init, u, eps=0.1, close_to_pure=1e-12, decrease = 'E', eta=0.1, exponent=0.5):
    traj = init
    for t in np.arange(1, T+1):
        if decrease == 'C':
            eta = eta
        elif decrease == 'E':
            eta = 1/t**exponent
        elif decrease == 'S':
            eta = ((init.shape[1]-1)**(-1/2))*(t**(-1/4))
        elif decrease == 'D':
            if t > 10:
                eta = 1/((np.log(t))**4)
            else:
                eta = 1
        new_point = MWUUpdateGeneral(traj, eta, u)
        mask = (new_point != 0) | (new_point != 1) 
        new_point[mask] += np.random.normal(0, eps, new_point[mask].shape)
        new_point = np.minimum( np.maximum(new_point, 0), 1)
        new_point /= np.sum(new_point, axis=1, keepdims=True) 
        traj = np.append(traj, [new_point], axis=0)
        if np.max(np.apply_along_axis(second_max, axis=1, arr=new_point)) < close_to_pure: 
            break
    return traj
def runToPureNoisyMWUGeneral(T, init, u, eps, close_to_pure, decrease, eta, exponent):
    new_point, t = init[-1], 0.0 
    while np.max(np.partition(new_point, -2, axis=1)[:, -2]) > close_to_pure:
        t += 1.0
        if decrease == 'C':
            eta = eta
        elif decrease == 'E':
            eta = 1/t**exponent
        elif decrease == 'S':
            eta = ((init.shape[1]-1)**(-1/2))*(t**(-1/4))
        elif decrease == 'D':
            if t > 10:
                eta = 1/((np.log(t))**4)
            else:
                eta = 1
        new_point = MWUUpdateGeneral(np.array([new_point]), eta, u)
        mask = (new_point != 0) | (new_point != 1) 
        test_noise = True 
        if test_noise:
            flag = True
            while flag:
                test_point = new_point.copy()
                test_point[mask] += np.random.normal(0, eps, test_point[mask].shape)
                test_point = np.minimum( np.maximum(test_point, 0), 1)
                test_sums = np.sum(test_point, axis=1, keepdims=True)
                if (test_sums != 0.0).all(): 
                    flag = False
                    new_point = test_point / test_sums
        else:
            new_point[mask] += np.random.normal(0, eps, new_point[mask].shape)
            new_point = np.minimum( np.maximum(new_point, 0), 1)
            new_point /= np.sum(new_point, axis=1, keepdims=True) 
    return new_point
def findPureProfile(T, init, u, eps, close_to_pure, decrease, eta, exponent):
    last_mixed_point = runToPureNoisyMWUGeneral(T, init, u, eps, close_to_pure, decrease, eta, exponent)
    translated_pure_profile = tuple( np.argmax(last_mixed_point, axis=1) + 1 )
    return translated_pure_profile
def extractAfterSecondStage(cur_first_stage_occ_dict, tries, hitting_probs, pure_profs):
    long_run_after_two_stages = np.zeros(len(hitting_probs[0]))
    for pure_profile, prob in cur_first_stage_occ_dict.items():
        long_run_after_two_stages += (prob/tries) * hitting_probs[pure_profs.index(pure_profile)]
    return long_run_after_two_stages
def findPureProfiles(tries, hitting_probs, pure_profs, T, inits, u, eps=0.1, close_to_pure=10e-6, decrease = 'E', eta=0.1, exponent=0.5, multi_core=False):
    tries *= len(inits) 
    occ_list = dict()
    convergence_dists = np.zeros((tries, len(hitting_probs[0])))
    if multi_core:
        import multiprocess as mp
        with mp.Pool(8) as pool:
            translated_pure_profiles = pool.map(lambda i_try: (np.random.seed((int(time() * 1e8) + i_try) % 2**32), findPureProfile(T, inits[i_try%len(inits)], u, eps, close_to_pure, decrease, eta, exponent))[1], range(tries))
            occ_lists = [Counter(translated_pure_profiles[:i+1]) for i in range(tries)]
            occ_list = occ_lists[-1]
            convergence_dists = np.array([extractAfterSecondStage(occ_lists[i_try], i_try+1, hitting_probs, pure_profs) for i_try in range(tries)])
    else:
        for i_try in tqdm(range(tries)):
            translated_pure_profile = findPureProfile(T, inits[i_try%len(inits)], u, eps, close_to_pure, decrease, eta, exponent)
            if translated_pure_profile not in occ_list:
                occ_list[translated_pure_profile] = 1.0
            else: occ_list[translated_pure_profile] += 1.0
            convergence_dists[i_try] = extractAfterSecondStage(occ_list, i_try+1, hitting_probs, pure_profs)
    return occ_list, np.array([np.sum(np.abs(convergence_dists[i_try] - convergence_dists[-1])) for i_try in range(tries)])
def findConvDist(hitting_probs, pure_profs, conv_l1_max, T, u, eps=0.1, close_to_pure=4e-3, decrease = 'E', eta=0.1, exponent=0.5, multi_core=True):
    n, s = len(u), u[0].shape[0]
    close_to_pure *= n*s 
    nsamples, nexperiments_per_sample, cur_conv = 0, 40, 1.0
    tocompare_convergence_dist = np.zeros(len(hitting_probs[0]))
    last_occ_list = Counter()
    while cur_conv > conv_l1_max:
        add_samples = 1
        nsamples += add_samples
        new_samples = sample_uniform(players=n, strategies=s, samples=add_samples)
        inits = [np.array([sample, sample]) for sample in new_samples]
        if multi_core:
            import multiprocess as mp
            with mp.Pool(8) as pool:
                new_translated_pure_profiles = pool.map(lambda i_try: (np.random.seed((int(time() * 1e8) + i_try + nsamples - add_samples+2) % 2**32), findPureProfile(T, inits[i_try%add_samples], u, eps, close_to_pure, decrease, eta, exponent))[1], range(add_samples*nexperiments_per_sample))
                last_occ_list.update(new_translated_pure_profiles)
                new_convergence_dist = extractAfterSecondStage(last_occ_list, nsamples*nexperiments_per_sample, hitting_probs, pure_profs)
                cur_conv = 0.5 * np.sum(np.abs(new_convergence_dist - tocompare_convergence_dist))
                tocompare_convergence_dist = new_convergence_dist 
    return last_occ_list, nsamples*nexperiments_per_sample 
def find_better_response_G(u: list[np.ndarray]) -> nx.DiGraph:
    n, s = len(u), u[0].shape[0]
    G = nx.DiGraph()
    for s in itertools.product(*map(range, u[0].shape)):
        new_normalization_coeff = 0.0
        for i in range(n):
            slicing_indexing_tuple = tuple(entry if idx != i else slice(None) for idx, entry in enumerate(s))
            i_utilities = u[i][slicing_indexing_tuple]
            i_better_responses = np.argwhere(i_utilities >= i_utilities[s[i]]).flatten().tolist()
            new_normalization_coeff += np.sum(i_utilities[i_better_responses] - i_utilities[s[i]])
        for i in range(n):
            slicing_indexing_tuple = tuple(entry if idx != i else slice(None) for idx, entry in enumerate(s))
            i_utilities = u[i][slicing_indexing_tuple] 
            i_better_responses = np.argwhere(i_utilities >= i_utilities[s[i]]).flatten().tolist() 
            normalization_coeff = np.sum(i_utilities[i_better_responses] - i_utilities[s[i]])
            G.add_edges_from([(
                tuple(map(lambda entry: entry+1, s)) 
                ,
                tuple(entry+1 if idx != i else better_s_i+1 for idx, entry in enumerate(s)) 
                ,
                {'w': (i_utilities[better_s_i] - i_utilities[s[i]])/new_normalization_coeff if new_normalization_coeff != 0.0 else 0.0}
                ) for better_s_i in i_better_responses if better_s_i != s[i]]) 
    return G
def find_tie_edges(G: nx.DiGraph, silent=False) -> dict[tuple[int], set[tuple[int]]]:
    tie_edges = {node: set() for node in G.nodes}
    for e in G.edges:
        if G.has_edge(e[1], e[0]): 
            tie_edges[e[0]].add(e[1])
    if not silent:
        print('Tie edges:')
        print('\n'.join([f"From {e0} to {e1}" for e0, set_of_e1 in tie_edges.items() for e1 in set_of_e1]))
        print('')
    return tie_edges
def calc_sccs(G: nx.DiGraph, tie_edges: dict[tuple[int], set[tuple[int]]], silent=False):
    G_condensation = nx.condensation(G)
    if not silent:
        print('SCCs of original graph:\n' + '\n'.join([
            f"SCC {scc_i[0]+1}: {scc_i[1]['members']}"
            for scc_i in G_condensation.nodes.data()
            ]) + '\n')
        print('sink SCCs of original graph:\n' + '\n'.join([
            f"SCC {scc_i+1}: {G_condensation.nodes.data()[scc_i]['members']}"
            for scc_i, out_degree in G_condensation.out_degree() if out_degree == 0
            ]) + '\n')
    G_notie = G.copy()
    G_notie.remove_edges_from([(e0, e1) for e0, set_of_e1 in tie_edges.items() for e1 in set_of_e1])
    G_notie_condensation = nx.condensation(G_notie)
    if not silent:
        print('SCCs of graph without tie edges:\n' + '\n'.join([
            f"SCC {scc_i[0]+1}: {scc_i[1]['members']}"
            for scc_i in G_notie_condensation.nodes.data()
            ]) + '\n')
        print('sink SCCs of graph without tie edges:\n' + '\n'.join([
            f"SCC {scc_i+1}: {G_notie_condensation.nodes.data()[scc_i]['members']}"
            for scc_i, out_degree in G_notie_condensation.out_degree() if out_degree == 0
            ]) + '\n')
    return (G_condensation, G_notie, G_notie_condensation)
def calc_orders(G_condensation, G_notie, tie_edges):
    sink_SCCs = [scc_i for scc_i, out_degree in G_condensation.out_degree() if out_degree == 0]
    all_sink_SCC_nodes = set(n for scc_i in sink_SCCs for n in G_condensation.nodes.data()[scc_i]['members'])
    sink_SCC_nodes = [G_condensation.nodes.data()[scc_i]['members'] for scc_i in sink_SCCs]
    nodes_by_order = [set()] 
    visited = set() 
    queue = deque(all_sink_SCC_nodes)
    current_order, next_order_nodes_to_visit = 0, deque() 
    while queue: 
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                if current_order != 0 or node not in all_sink_SCC_nodes: 
                    nodes_by_order[current_order].add(node)
                queue.extend(
                    new_node for new_node in G_notie.predecessors(node)
                    if new_node not in visited
                )
                next_order_nodes_to_visit.extend([n for n in tie_edges[node] if n not in visited])
        current_order += 1
        nodes_by_order.append(set())
        queue = next_order_nodes_to_visit
        next_order_nodes_to_visit = deque()
    while not nodes_by_order[-1] and len(nodes_by_order) != 1: nodes_by_order = nodes_by_order[:-1]
    max_order = len(nodes_by_order) - 1 
    return (sink_SCCs, all_sink_SCC_nodes, sink_SCC_nodes, nodes_by_order, max_order)
def make_coloring(G, all_sink_SCC_nodes, nodes_by_order, max_order):
    color_codes = [mcolors.to_hex(plt.get_cmap('tab20')(i)) for i in np.linspace(0.2, 1, max_order + 1)]
    color_dict = {}
    for node in all_sink_SCC_nodes:
        color_dict[node] = 'red'
    for order, nodes in enumerate(nodes_by_order):
        for node in nodes:
            color_dict[node] = color_codes[order]
    coloring = [color_dict[node] for node in G.nodes()] 
    return (coloring, color_codes)
def plot_orders_coloring(max_order, color_codes):
    plt.figure(figsize=(max_order+2,1))
    colors_g = nx.DiGraph()
    colors_g.add_nodes_from(range(max_order+1))
    pos = {node: (node, 0) for node in colors_g.nodes()}
    nx.draw(colors_g, pos=pos, with_labels=True, node_color=color_codes, node_size=1500)
    plt.xlim(-0.5, max_order + 0.5)
    plt.show()
def pretty_print_graph(G, coloring = None, with_weights = True, custom_pos = None, labels_without_frozenset = True,
                       node_size=3000, figsize=(6,6), prog="circo"):
    plt.figure(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(G, prog=prog) if custom_pos is None else custom_pos 
    nx.draw(G, pos=pos, with_labels=not labels_without_frozenset, node_color=coloring, node_size=node_size)
    if labels_without_frozenset:
        custom_labels = {n: ','.join(map(str, n)) if isinstance(n, frozenset) else str(n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=custom_labels)
    edge_labels = nx.get_edge_attributes(G, 'w')
    if with_weights:
        edge_labels = {k: '{:.2f}'.format(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
def add_edge(G, u, v, w):
    if G.has_edge(u, v):
        G[u][v]['w'] += w
    else:
        G.add_edge(u, v, w=w)
def add_edges_from(G, edges):
    for u, v, w in edges:
        add_edge(G, u, v, w)
def find_hitting_probabilities(sink_SCC_nodes, G_notie, tie_edges, nodes_by_order, max_order, node_size=3000, figsize=(6,6), prog="circo", silent=False):
    processed_G_notie = G_notie.copy()
    processed_nodes_by_order = nodes_by_order.copy()
    processed_tie_edges = tie_edges.copy() 
    if not silent:
        print('Initial graph with no ties:')
        pretty_print_graph(processed_G_notie, with_weights=True, node_size=node_size, figsize=figsize, prog=prog)
    for order in range(max_order, 0, -1):
        if not silent: print('TO PROCESS NOW:', processed_nodes_by_order[order])
        current_condensation = nx.condensation(processed_G_notie) 
        pseudosink_nodes, stationaries = [], []
        for scc_i, out_degree in current_condensation.out_degree():
            candidate_pseudosink_nodes = current_condensation.nodes.data()[scc_i]['members']
            if out_degree == 0 and candidate_pseudosink_nodes.intersection( processed_nodes_by_order[order] ): 
                if not silent: print('Nodes in current pseudo-sink SCC:', candidate_pseudosink_nodes)
                pseudosink_nodes.append(candidate_pseudosink_nodes)
                if len(candidate_pseudosink_nodes) == 1: 
                    stationaries.append([1.0])
                    continue
                pseudosink_subgraph = processed_G_notie.subgraph(candidate_pseudosink_nodes)
                transition_matrix = nx.adjacency_matrix(pseudosink_subgraph, weight='w').toarray() 
                transition_matrix = np.clip(transition_matrix, a_min=0, a_max=None) 
                remaining_row_sums = 1 - transition_matrix.sum(axis=1)
                np.fill_diagonal(transition_matrix, transition_matrix.diagonal() + remaining_row_sums.flatten())
                transition_matrix = np.clip(transition_matrix, a_min=0, a_max=None) 
                if not silent:
                    print( 'Transition matrix for stationary probabilities of pseudo-sink SCC:' )
                    print( transition_matrix )
                mc = pydtmc.MarkovChain(transition_matrix) 
                stationary_probabilities = mc.stationary_distributions[0] 
                if not silent:
                    print( 'Stationary distribution:', stationary_probabilities )
                stationaries.append(stationary_probabilities)
        new_names_of_collapsed_pseudosinks = [ frozenset().union(*(n if isinstance(n, frozenset) else {n} for n in this_pseudosink_nodes))
                                                for this_pseudosink_nodes in pseudosink_nodes ]
        for i, this_pseudosink_nodes in enumerate(pseudosink_nodes):
            processed_tie_edges[new_names_of_collapsed_pseudosinks[i]] = set() 
            for j, n in enumerate(this_pseudosink_nodes): 
                if not silent: print('here\'s the back-edges I\'m about to add:', [(u, new_names_of_collapsed_pseudosinks[i], data['w']) for u, _n, data in processed_G_notie.in_edges(n, data=True) if u not in this_pseudosink_nodes])
                add_edges_from(processed_G_notie, [(u, new_names_of_collapsed_pseudosinks[i], data['w']) for u, _n, data in processed_G_notie.in_edges(n, data=True) if u not in this_pseudosink_nodes]) 
                processed_G_notie.remove_node(n)
                for n2 in processed_tie_edges[n]: 
                    if n2 in this_pseudosink_nodes: continue
                    elif n2 in processed_nodes_by_order[order-1]:
                        add_edge(processed_G_notie, new_names_of_collapsed_pseudosinks[i], n2, w = stationaries[i][j])
                        if not silent: print(f'also adding the reverse tie edge: from {n2} to {new_names_of_collapsed_pseudosinks[i]}')
                        processed_tie_edges[n2].add(new_names_of_collapsed_pseudosinks[i]) 
                        processed_tie_edges[n2].remove(n) 
                    elif n2 in processed_nodes_by_order[order]:
                        which_pseudosink = next((i for i, set_ in enumerate(pseudosink_nodes) if n2 in set_), None)
                        assert(which_pseudosink != i) 
                        if which_pseudosink is not None: 
                            add_edge(processed_G_notie, new_names_of_collapsed_pseudosinks[i], new_names_of_collapsed_pseudosinks[which_pseudosink], w = stationaries[i][j])
                        else: 
                            assert(n2 in processed_G_notie.nodes) 
                            add_edge(processed_G_notie, new_names_of_collapsed_pseudosinks[i], n2, w = stationaries[i][j])
                            if n in processed_tie_edges[n2]: 
                                processed_tie_edges[n2].remove(n)
                                processed_tie_edges[n2].add(new_names_of_collapsed_pseudosinks[i])
                            processed_nodes_by_order[order-1].add(n2) 
                    else: raise Exception(f'Should never reach this stage! Unknown tie edge encountered: {n} -> {n2}')
            outgoing_prob_sum = sum(data['w'] for _, _, data in processed_G_notie.out_edges(new_names_of_collapsed_pseudosinks[i], data=True))
            for _, _, data in processed_G_notie.out_edges(new_names_of_collapsed_pseudosinks[i], data=True): data['w'] /= outgoing_prob_sum
        processed_nodes_by_order[order-1].update(new_names_of_collapsed_pseudosinks)
        processed_nodes_by_order[order-1].update(processed_nodes_by_order[order]) 
        if not silent:
            print('here\'s the updated tie edges adjacency list:')
            for k, v in processed_tie_edges.items():
                print(f'{k}: {v}')
            pretty_print_graph(processed_G_notie, with_weights=True, node_size=node_size, figsize=figsize, prog=prog)
    final_processed_G_notie_with_collapsed_true_sinks = processed_G_notie.copy()
    for this_sink_nodes in sink_SCC_nodes:
        new_node_name = frozenset(this_sink_nodes)
        final_processed_G_notie_with_collapsed_true_sinks.add_node(new_node_name) 
        for n in this_sink_nodes:
            for u, _n, data in final_processed_G_notie_with_collapsed_true_sinks.in_edges(n, data=True):
                if u not in this_sink_nodes:
                    add_edge(final_processed_G_notie_with_collapsed_true_sinks, u, new_node_name, w = data['w'])
        final_processed_G_notie_with_collapsed_true_sinks.remove_nodes_from(this_sink_nodes)
    if not silent:
        print('Final graph with collapsed true sinks:')
        pretty_print_graph(final_processed_G_notie_with_collapsed_true_sinks, with_weights=True, node_size=node_size, figsize=figsize, prog=prog)
    hitting_probs = []
    for scc in sink_SCC_nodes:
        target = list(final_processed_G_notie_with_collapsed_true_sinks.nodes).index( frozenset(scc) )
        transition_matrix = nx.adjacency_matrix(final_processed_G_notie_with_collapsed_true_sinks, weight='w').toarray()
        transition_matrix = np.clip(transition_matrix, a_min=0, a_max=None) 
        remaining_row_sums = 1 - transition_matrix.sum(axis=1)
        np.fill_diagonal(transition_matrix, transition_matrix.diagonal() + remaining_row_sums.flatten())
        transition_matrix = np.clip(transition_matrix, a_min=0, a_max=None) 
        if transition_matrix.shape[0] == 1: 
            hitting_probs.append( np.array([1.0]) )
        else:
            mc = pydtmc.MarkovChain(transition_matrix)
            hitting_probs.append( mc.hitting_probabilities([target]) )
    return hitting_probs, final_processed_G_notie_with_collapsed_true_sinks
def run_algo(u: list[np.ndarray], node_size=3000, figsize=(6,6), prog="circo", silent=False):
    G = find_better_response_G(u)
    tie_edges = find_tie_edges(G, silent=silent)
    G_condensation, G_notie, G_notie_condensation = calc_sccs(G, tie_edges, silent=silent)
    sink_SCCs, all_sink_SCC_nodes, sink_SCC_nodes, nodes_by_order, max_order = calc_orders(G_condensation, G_notie, tie_edges)
    if not silent:
        coloring, color_codes = make_coloring(G, all_sink_SCC_nodes, nodes_by_order, max_order)
        custom_pos = None
        pretty_print_graph(G, coloring, with_weights=True, custom_pos=custom_pos, labels_without_frozenset=True, node_size=node_size, figsize=figsize, prog=prog)
        plot_orders_coloring(max_order, color_codes)
    pre_hitting_probs, final_processed_G_notie_with_collapsed_true_sinks = find_hitting_probabilities(sink_SCC_nodes, G_notie, tie_edges, nodes_by_order, max_order, node_size=node_size, figsize=figsize, prog=prog, silent=silent)
    packed_hitting_probs = np.dstack(pre_hitting_probs).reshape(-1, len(sink_SCC_nodes))
    if not silent:
        print('Sink SCCs (in order):', sink_SCC_nodes)
        print('Nodes whose probabilities will be computed (FROM):', final_processed_G_notie_with_collapsed_true_sinks.nodes)
        print('Hitting probabilities:', packed_hitting_probs)
    sinks = sink_SCC_nodes
    pure_profs, hitting_probs = [], []
    for i, nodes in enumerate(final_processed_G_notie_with_collapsed_true_sinks.nodes):
        if not isinstance(nodes, frozenset):
            nodes = [nodes] 
        for node in nodes:
            pure_profs.append(node)
            hitting_probs.append(packed_hitting_probs[i])
    hitting_probs = np.array(hitting_probs)
    if not silent: print(np.mean(hitting_probs, axis=0)) 
    return sinks, pure_profs, hitting_probs
def run_game(n, s, i_try):
    silent = True
    np.random.seed((int(time() * 1e8) + i_try) % 2**32)
    u = [np.random.uniform(-1, 1, size=(s,) * n) for _ in range(n)]
    G = find_better_response_G(u)
    tie_edges = find_tie_edges(G, silent=silent)
    G_condensation, G_notie, _ = calc_sccs(G, tie_edges, silent=silent)
    _, _, sink_SCC_nodes, _, max_order = calc_orders(G_condensation, G_notie, tie_edges)
    this_num_sinks = len(sink_SCC_nodes)
    this_avg_size_sink = sum(map(len, sink_SCC_nodes))/len(sink_SCC_nodes)
    this_max_size_sink = max(map(len, sink_SCC_nodes))
    interesting = max_order >= 4 and this_max_size_sink >= 2 and this_num_sinks >= 2
    return this_num_sinks, this_max_size_sink, this_avg_size_sink, interesting, u
def search_for_game_graph(n: int, s:int, max_tries:int = 10**10, multi_core=True) -> tuple[int, int, list[np.ndarray]]:
    if multi_core:
        import multiprocess as mp
        with mp.Pool(8) as pool:
            batch = 8*600 
            for i_try in range(int(max_tries/batch)):
                results = pool.map(lambda i: run_game(n, s, i), range(batch*i_try, batch*(i_try+1)))
                interesting_games = [(this_num_sinks, this_max_size_sink, u) for this_num_sinks, this_max_size_sink, _, interesting, u in results if interesting]
                if len(interesting_games) > 0: return interesting_games[0]
    else:
        for i_try in tqdm(range(max_tries)):
            this_num_sinks, this_max_size_sink, _, interesting, u = run_game(n, s, i_try)
            if interesting: return (this_num_sinks, this_max_size_sink, u)
    return (0, 0, [])
def avg_characteristics_of_games(n: int, s:int, tries:int = 1000, multi_core=False):
    interesting_games, num_sinks, avg_size_sink = [], 0, 0.0
    if multi_core:
        import multiprocess as mp
        with mp.Pool(8) as pool:
            results = pool.map(lambda i: run_game(n, s, i), range(tries))
            interesting_games = [(this_num_sinks, this_max_size_sink, u) for this_num_sinks, this_max_size_sink, _, interesting, u in results if interesting]
            num_sinks = sum(num_sinks for num_sinks, _, _, _, _ in results)
            avg_size_sink = sum(avg_size_sink for _, _, avg_size_sink, _, _ in results)
    else:
        for i_try in tqdm(range(tries)):
            this_num_sinks, this_max_size_sink, this_avg_size_sink, interesting, u = run_game(n, s, i_try)
            num_sinks += this_num_sinks
            avg_size_sink += this_avg_size_sink
            if interesting: interesting_games.append((this_num_sinks, this_max_size_sink, u))
    return num_sinks/tries, avg_size_sink/tries, interesting_games
p1 = np.array([[[-1, -1,  1], [ 0, -1,  1], [-1, -2,  1]],
               [[-2,  1,  2], [-1,  0,  2], [-2,  1, -2]],
               [[-1, -1,  1], [ 0, -2, -2], [-2, -1, -2]]])
p2 = np.array([[[-1,  2, -1], [ 0, -2,  0], [-1,  1,  1]],
               [[ 2, -1,  1], [ 2, -2,  1], [-2,  0, -2]],
               [[ 2, -2,  2], [ 2,  1,  2], [ 1,  0,  2]]])
p3 = np.array([[[ 2,  1,  1], [ 1, -2, -1], [ 2, -1,  2]],
               [[-2,  0,  2], [-2, -1,  0], [ 0,  2, -2]],
               [[ 1,  1,  1], [ 1, -1, -2], [ 1,  2, -2]]])
u = [p1, p2, p3]
n, s = len(u), u[0].shape[0]
print('Game:', u[0].shape, '\n')
for i in range(n):
    print(f'utility of p{i+1}:')
    print(u[i], '\n')
silent = True
G = find_better_response_G(u)
tie_edges = find_tie_edges(G, silent=silent)
G_condensation, G_notie, G_notie_condensation = calc_sccs(G, tie_edges, silent=silent)
sink_SCCs, all_sink_SCC_nodes, sink_SCC_nodes, nodes_by_order, max_order = calc_orders(G_condensation, G_notie, tie_edges)
coloring, color_codes = make_coloring(G, all_sink_SCC_nodes, nodes_by_order, max_order)
pretty_print_graph(G, coloring, with_weights=True, labels_without_frozenset=True, node_size=2000, figsize=(10,10), prog="circo")
plot_orders_coloring(max_order, color_codes)
pre_hitting_probs, final_processed_G_notie_with_collapsed_true_sinks = find_hitting_probabilities(sink_SCC_nodes, G_notie, tie_edges, nodes_by_order, max_order, node_size=2000, figsize=(6,6), prog="circo", silent=True)
packed_hitting_probs = np.dstack(pre_hitting_probs).reshape(-1, len(sink_SCC_nodes))
print('Sink SCCs (in order):', sink_SCC_nodes)
print('Nodes whose probabilities will be computed (FROM):', final_processed_G_notie_with_collapsed_true_sinks.nodes)
print('Hitting probabilities:', packed_hitting_probs)
restriction = [(2,3,2),(2,2,3),(2,1,3), (3,2,1), (3,1,1), (1,1,1), (1,2,1), (1,3,1), (1,3,3)]
subG = G.subgraph(restriction)
color_codes = [mcolors.to_hex(plt.get_cmap('tab20')(i)) for i in np.linspace(0.2, 1, max_order + 1)]
color_dict = {}
for node in all_sink_SCC_nodes:
    color_dict[node] = 'red'
for order, nodes in enumerate(nodes_by_order):
    for node in nodes:
        color_dict[node] = color_codes[order]
new_coloring = [color_dict[node] for node in subG.nodes()]
pretty_print_graph(subG, new_coloring, with_weights=True, labels_without_frozenset=True, node_size=2000, figsize=(6,6), prog="circo")
sinks, pure_profs, hitting_probs = run_algo(u, node_size=3000, figsize=(6,6), prog="circo", silent=True)
first_stage, ntries = findConvDist(hitting_probs, pure_profs, conv_l1_max=0.2/100, T=1e10, u=u, eps=0.3, decrease='C', eta=0.5, multi_core=True)
samples_needed = ntries 
long_run_after_two_stages = extractAfterSecondStage(first_stage, ntries, hitting_probs, pure_profs)
print(ntries/40, 'needed')
print('First stage:', first_stage)
print('Aggregated:', long_run_after_two_stages)
def time_lots_of_times(n, s, tries):
    times = []
    for t in range(tries):
        np.random.seed((int(time() * 1e8) + n*s+t) % 2**32)
        flag_success = False
        sinks, pure_profs, hitting_probs = [], [], []
        while not flag_success:
            u = [np.random.randint(-6, 11, size=(s,) * n) for _ in range(n)]
            try:
                sinks, pure_profs, hitting_probs = run_algo(u, node_size=3000, figsize=(6,6), prog="circo", silent=True)
                flag_success = True
            except:
                flag_success = False
        start = time()
        first_stage, nsamples = findConvDist(hitting_probs, pure_profs, conv_l1_max=0.2/100, T=1e10, u=u, eps=0.3, decrease='C', eta=0.5, multi_core=True)
        times.append(time() - start)
    return sum(times)/tries
exp2s_time_needed = []
for s in tqdm(range(2, 13)):
    n = 2
    exp2s_time_needed.append(time_lots_of_times(n, s, tries=10)) 
exp3s_time_needed = []
for s in tqdm(range(2, 10)):
    n = 3
    exp3s_time_needed.append(time_lots_of_times(n, s, tries=10)) 
expn2_time_needed = []
for n in tqdm(range(2, 13)):
    s = 2
    expn2_time_needed.append(time_lots_of_times(n, s, tries=1)) 
expn3_time_needed = []
for n in tqdm(range(2, 8)):
    s = 3
    expn3_time_needed.append(time_lots_of_times(n, s, tries=5)) 
plt.figure()
plt.plot(range(2, 13), exp2s_time_needed, marker='.')
plt.title('Seconds needed for convergence of 2-player s-strategy games')
plt.xlabel('s')
plt.show()
plt.figure()
plt.plot(range(2, 10), exp3s_time_needed, marker='.')
plt.title('Seconds needed for convergence of 3-player s-strategy games')
plt.xlabel('s')
plt.show()
plt.figure()
plt.plot(range(2, 13), expn2_time_needed, marker='.')
plt.title('Seconds needed for convergence of n-player 2-strategy games')
plt.xlabel('n')
plt.show()
plt.figure()
plt.plot(range(2, 8), expn3_time_needed, marker='.')
plt.title('Seconds needed for convergence of n-player 3-strategy games')
plt.xlabel('n')
plt.show()
plt.figure()
plt.scatter([s*2 for s in range(2, 13)], exp2s_time_needed[:], marker='.', label='2s')
plt.scatter([s*3 for s in range(2, 10-1)], exp3s_time_needed[:-1], marker='.', label='3s')
plt.scatter([2*n for n in range(2, 13)], expn2_time_needed[:], marker='.', label='n2')
plt.scatter([3*n for n in range(2, 8)], expn3_time_needed[:], marker='.', label='n3')
plt.title('Seconds needed for convergence with total size of game')
plt.xlabel('$n\\times s$')
plt.xticks([2*n for n in range(2, 13)])
plt.show()
