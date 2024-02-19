from gurobipy import GRB
import gurobipy as gp
import numpy as np
from fmmd import parallel_utils
from tqdm.auto import tqdm, trange
from typing import List, Tuple, Optional, Set, Dict
import networkx as nx
from scipy.spatial.distance import pdist, squareform, cdist
from itertools import combinations
import matplotlib.pyplot as plt
import logging
from time import perf_counter as time
import math
logger = logging.getLogger(__name__)

def gonzales_algorithm(
        initial_solution: set,
        features: np.ndarray,
        ids: np.ndarray,
        k: int,
        diversity: float = -np.inf,
        lower_constraint: int = 1,
        initial_sol_distances: Optional[np.ndarray] = None,
        initial_sol_diversity: Optional[float] = None,
        _tqdm: bool = False,
        parallel_dist_update: bool = False
) -> Tuple[set, float, np.ndarray, bool]:
    """Greedy algorithm for k-centers clustering.

    Args:
        initial_solution (set): The initial items in the solution. Can be empty.
        features (np.ndarray): The feature vector of the items
        ids (np.ndarray): The ids of the items 
        k (int): The number of samples required
        diversity (float, optional): The lower threshold for diversity when being greedy. Defaults to -np.inf.
        lower_constraint (int, optional): The lower bound for how many samples are needed. Defaults to 1.
        initial_sol_distances (Optional[np.ndarray],optional): The smallest distance between each items and all solution items. Defaults to None.
        initial_sol_diversity (Optional[float],optional): The diversity of the initial solution set
        _tqdm (bool, optional): Whether to display a tqdm loading bar. Defaults to False.
        parallel_dist_update (bool, optional): Whether to update distances in parallel. Defaults to False.

    Returns:
        Tuple[set,float,np.ndarray,bool]: A solution set of item ids, its diversity,. 
                                        the minimum distances of items to solution and if it was successful.
    """
    # if initial solution is empty add first element
    if len(initial_solution) == 0:
        initial_solution.add(ids[0])
        solution_idxs = set([0])
    else:
        # get the index for the elements in array
        solution_idxs = set(
            np.where(np.isin(ids, list(initial_solution), assume_unique=True))[0])
    assert len(solution_idxs) == len(initial_solution)
    # if distance to initial solution doesn't exist
    if initial_sol_distances is None:
        # item distances to solution set
        element_distances = np.repeat(np.inf, features.shape[0])
        sol_div = np.inf
        # update diversity for each solution
        for solution_idx in solution_idxs:
            sol_div = min(sol_div, element_distances[solution_idx])
            if parallel_dist_update:
                parallel_utils.update_dists(element_distances, features, features[solution_idx])
            else:
                parallel_utils.update_dists_sequential(element_distances, features, features[solution_idx])
    elif initial_sol_distances is not None and initial_sol_diversity is not None:
        element_distances = initial_sol_distances
        sol_div = initial_sol_diversity
    else:
        raise ValueError(
            f"Initial solution diversity must be provided with initial solution distances")
    if _tqdm:
        pbar = tqdm(total=k-len(solution_idxs))
    while len(solution_idxs) < k:
        max_idx = np.argmax(element_distances)
        max_dist = element_distances[max_idx]
        max_item = features[max_idx]
        sol_div = min(sol_div, max_dist)
        logger.debug(f"\t\t{max_dist=}")
        logger.debug(f"\t\t{sol_div=}")
        if sol_div < diversity:
            break
        logger.debug(f"\t\tAdded item {ids[max_idx]}")
        solution_idxs.add(max_idx)
        # update the closest distance to current candidate set
        if parallel_dist_update:
            parallel_utils.update_dists(element_distances, features, max_item)
        else:
            parallel_utils.update_dists_sequential(element_distances, features, max_item)
        if _tqdm:
            pbar.update()
    if _tqdm:
        pbar.close()
    solution = set(ids[list(solution_idxs)])
    # check if diversity and lower bound constraints are met
    if len(solution) < lower_constraint and diversity != -np.inf:
        return solution, sol_div, element_distances, False
    else:
        return solution, sol_div, element_distances, True


def group_gonzales_algorithm(
        initial_solution: set,
        features: np.ndarray,
        ids: np.ndarray,
        groups: np.ndarray,
        k: int,
        eps: float,
        constraints: Dict[int,Tuple[int, int]],
        diversity_threshold: float = 0,
        working_data: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
        parallel_dist_update: bool = False
) -> Tuple[set, float, dict]:
    """A version of greedy k-centres algorithm when data has groups.

    Args:
        initial_solution (set): The initial items in the solution
        features (np.ndarray): The feature vectors of the items
        ids (np.ndarray): The ids of the items
        groups (np.ndarray): The group (int) of items
        k (int): The minimum number of total samples needed
        eps (int): The fraction to relax the diversity threshold after each failure
        constraints (Dict[int,Tuple[int, int]]): The set of lower and upper limits of samples for each group.
        diversity_threshold (float, optional): The threshold for each group. Defaults to 0.
        working_data (Optional[Dict[int,Tuple[np.ndarray,float]]],optional]: The distances and diversity for each group. If available skips certain computations. Defaults to None.
        parallel_dist_update (bool, optional): Whether to update distances in parallel. Defaults to False.
    Returns:
        Tuple[set, float, dict]: The solution set of item ids, the final diversity threshold and the working data.
    """
    final_solution = initial_solution.copy()
    _groups = np.unique(groups)
    if working_data is None:
        working_data = {}
    sorter = np.argsort(ids)
    while True:
        under_capped = False
        solution_idxs = sorter[np.searchsorted(ids, list(final_solution), sorter=sorter)]
        assert len(solution_idxs) == len(final_solution)
        for group in _groups:
            group_idxs = np.where(groups == group)[0]
            lower_constraint = constraints[group][0]
            # extract current group from initial solution
            solution_group_idxs = list(set(solution_idxs)&set(group_idxs))
            # if all items from group are already in solution then skip
            if len(solution_group_idxs) == len(group_idxs):
                continue
            # create a copy of the features and ids
            _features = features[group_idxs]
            _ids = ids[group_idxs]
            # if the group is to be included completely
            if len(group_idxs) <= lower_constraint:
                # add all group items
                final_solution.update(_ids)
                _diversity = compute_diversity(_ids,_features,_ids)
                # Check if overall diversity threshold reduces
                if _diversity < diversity_threshold:
                    # update the diversity threshold
                    diversity_threshold = _diversity
                    logger.info(f"Added full group {group} and diversity threshold changed to: {diversity_threshold:e}. Restarting")
                    # set the flag to restart
                    under_capped = True
                    break
                else:
                    # just continue with other groups
                    logger.info(f"Added full group {group} and diversity threshold didn't change. Continuing")
                    continue
            
            _initial_solution = set(ids[solution_group_idxs])
            logger.debug(f"{group=}")
            logger.debug(f"\t{_initial_solution=}")
            logger.debug(f"\t{diversity_threshold=}")
            
            # if working data for group exists load them
            if group not in working_data:
                _initial_solution_distances = None
                _initial_solution_diversity = None
            else:
                _initial_solution_distances = working_data[group][0]
                _initial_solution_diversity = working_data[group][1]
            # Runs the Gonzales algorithm for the group
            _group_solution, _group_solution_diversity, _group_solution_distances, success = gonzales_algorithm(
                _initial_solution, _features, _ids, k, diversity_threshold, lower_constraint, _initial_solution_distances, _initial_solution_diversity, _tqdm=False, parallel_dist_update=parallel_dist_update)
            # update working data
            working_data[group] = (
                _group_solution_distances, _group_solution_diversity)
            final_solution.update(_group_solution)
            # if the algorithm was unsuccessful
            if not success:
                # set flag to restart
                under_capped = True
                diversity_threshold = (1-eps)*diversity_threshold
                logger.info(
                    f"{group=} is undercapped. Reducing Diversity Threshold to {diversity_threshold:E}")
                break
        
        # if all groups are successful return solution
        if not under_capped:
            return final_solution, diversity_threshold, working_data


def get_coreset_graph(
    solution: set,
    diversity: float,
    features: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    constraints: Dict[int,Tuple[int, int]],
    G:Optional[nx.Graph]=None,
    parallel_edge_creation: bool = False
) -> nx.Graph:
    """Creates a coreset graph given a greedy solution for data with groups

    Args:
        solution (set): The set of item ids in coreset
        diversity (float): The diversity threshold for edges
        features (np.ndarray): The feature vectors of the items
        ids (np.ndarray): The item ids
        groups (np.ndarray): The groups (int) of the ids
        constraints (Dict[int,Tuple[int, int]]): The list of lower and upper limits on number of samples for each group

    Returns:
        nx.Graph: The coreset graph
    """
    if G is None:
        _solution = sorted(list(solution))
        # ensure that order of features is same as that in solutions
        sorter = np.argsort(ids)
        solution_idxs = sorter[np.searchsorted(ids, _solution, sorter=sorter)]
        # find edges between solution items where 
        # distance is below the threshold
        start = time()
        if parallel_edge_creation:
            us,vs,dists = parallel_utils.edges(features[solution_idxs],diversity)
            parallel_edges_time = time() - start
            logger.info(f"{parallel_edges_time=}")
        else:
            us,vs,dists = parallel_utils.edges_sequential(features[solution_idxs],diversity)
            sequential_edges_time = time() - start
            logger.info(f"{sequential_edges_time=}")
        
        start = time()
        G = nx.Graph()
        G.add_nodes_from(_solution)
        node_attr = dict()
        for idx, item in zip(solution_idxs, _solution):
            _group = groups[idx]
            node_attr[item] = {
                "lower_bound": constraints[_group][0],
                "upper_bound": constraints[_group][1],
                "group": _group,
                "id":item
                }

        nx.set_node_attributes(G, node_attr)
        N = len(_solution)
        M = len(dists)
        for i,j,dist in zip(us,vs,dists):
            # get the exact solution node
            u = _solution[i]
            v = _solution[j]
            G.add_edge(u,v,dist=dist)
        
        graph_creation_time = time() - start
        logger.info(f"{graph_creation_time=}")
    else:
        start = time()
        # Remove edges that don't satisfy current threshold
        deleted_edges = [(u,v) for u,v,d in G.edges(data="dist") if d >= diversity]
        logger.debug(f"Deleting {len(deleted_edges)} edges")
        G.remove_edges_from(deleted_edges)
        # find the new nodes
        new_nodes = solution - G.nodes
        old_nodes = solution - new_nodes
        logger.debug(f"Adding {len(new_nodes)} nodes")
        new_nodes = sorted(list(new_nodes))
        old_nodes = sorted(list(old_nodes))
        # ensure that order of features is same as that in solutions
        sorter = np.argsort(ids)
        new_node_idxs = sorter[np.searchsorted(ids, new_nodes, sorter=sorter)]
        old_node_idxs = sorter[np.searchsorted(ids, old_nodes, sorter=sorter)]
        G.add_nodes_from(new_nodes)
        node_attr = dict()
        for idx, item in zip(new_node_idxs, new_nodes):
            _group = groups[idx]
            node_attr[item] = {
                "lower_bound": constraints[_group][0],
                "upper_bound": constraints[_group][1],
                "group": _group,
                "id": item
                }
        nx.set_node_attributes(G, node_attr)
        logger.debug(f"Adding {len(new_nodes)} nodes")
        # compute distances between new and old nodes
        new_old_dists = cdist(features[new_node_idxs],features[old_node_idxs]).flatten()
        new_old_dists_idxs = np.where(new_old_dists<diversity)[0]
        logger.debug(f"Adding {len(new_old_dists_idxs)} edges")
        N = len(new_nodes)
        M = len(old_nodes)
        for idx,d in zip(new_old_dists_idxs,new_old_dists[new_old_dists_idxs]):
            # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
            i = idx // M
            j = idx % M
            u = new_nodes[i]
            v = old_nodes[j]
            G.add_edge(u,v,dist=d)
        # compute distances between new nodes
        new_new_dists = pdist(features[new_node_idxs])
        new_new_dists_idxs = np.where(new_new_dists<diversity)[0]
        logger.debug(f"Adding {len(new_new_dists_idxs)} edges")
        N = len(new_nodes)
        M = len(new_new_dists)
        for idx,d in zip(new_new_dists_idxs,new_new_dists[new_new_dists_idxs]):
            # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
            i = N - 2 - int(math.sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5)
            j = idx + i + 1 - M + (N-i)*((N-i)-1)//2
            u = new_nodes[i]
            v = new_nodes[j]
            G.add_edge(u,v,dist=d)
        graph_update_time = time() - start
        logger.info(f"{graph_update_time=}")

    # fig,ax=plt.subplots(figsize=(10,10))
    # pos = nx.circular_layout(G)
    # node_colors = list(nx.get_node_attributes(G,"group").values())
    # node_labels = {n:f"{n}\n{v}" for n,v in nx.get_node_attributes(G,"old_id").items()}
    # nx.draw_networkx_nodes(G,pos=pos,alpha=0.5,node_color=node_colors)
    # nx.draw_networkx_edges(G,pos=pos)
    # nx.draw_networkx_labels(G,pos=pos,labels=node_labels)
    # plt.show()
    logger.info(f"{G.number_of_edges()=} {G.number_of_nodes()=}")
    return G


def get_ILP_solution(G: nx.Graph, k: int, time_limit: int = 300, verbose: bool = False):
    """Attempts to solve a maximum independents set (MIS) problem given a coreset graph.

    Args:
        G (nx.Graph): A coreset graph. Nodes must have lowe and upper bounds and group
        k (int): The minimum number of total samples
        time_limit (int, optional): The time limit for gurobi optimizer. Defaults to 300.
        verbose (bool): Whether to have verbose gurobi output. Defaults to False.

    Returns:
        Optional[set]: A set if a feasible solution is found. None otherwise.
    """
    try:
        model: gp.Model = gp.Model("mis")
        model.setParam(GRB.Param.OutputFlag, verbose)
        model.setParam(GRB.Param.TimeLimit, time_limit)
        n = G.number_of_nodes()
        size = n
        vars_x = model.addVars(G.nodes, vtype=GRB.BINARY, obj=size, name="x")
        model.modelSense = GRB.MAXIMIZE
        for eid, (u, v) in enumerate(G.edges):
            model.addConstr(vars_x[u] + vars_x[v] <= 1, "edge_" + str(eid))
        expr = gp.LinExpr()
        for n in G.nodes:
            expr.addTerms(1, vars_x[n])
        model.addConstr(expr <= k, "size")

        groups = np.unique(
            np.array(list(nx.get_node_attributes(G, "group").values())))
        grp_expr = {i: gp.LinExpr() for i in groups}
        grp_lb = dict()
        grp_up = dict()
        node_attr = G.nodes(data=True)
        for node in G.nodes:
            node_group = node_attr[node]["group"]
            grp_expr[node_group].addTerms(1, vars_x[node])
            grp_lb[node_group] = node_attr[node]["lower_bound"]
            grp_up[node_group] = node_attr[node]["upper_bound"]
        for grp, _expr in grp_expr.items():
            model.addConstr(_expr >= grp_lb[grp], "lb_color_" + str(grp))
            model.addConstr(_expr <= grp_up[grp], "ub_color_" + str(grp))
        model.optimize()
        # for constrs in model.getConstrs():
        #     print(constrs)
        #     lhs= model.getRow(constrs)
        #     rhs =model.getAttr("RHS",[constrs])
        #     sense = model.getAttr("Sense",[constrs])
        #     print(lhs,sense,rhs)
        #     print()
        S = set()
        for x, var in vars_x.items():
            if var.X > 0.5:
                S.add(x)
        if len(S) >= k:
            return S
        else:
            logger.debug("Solution not full enough")
            return None

    except gp.GurobiError as error:
        print("Error code " + str(error))
        exit(0)

    except AttributeError:
        return None


def fmmd(
    features: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    k: int,
    constraints: Dict[int,Tuple[int, int]],
    eps: float,
    time_limit:int = 300,
    verbose:bool = False,
    parallel_dist_update: bool = False,
    parallel_edge_creation: bool = False
) -> Tuple[set, float]:
    """The implementation of the Fair Max-Min Diversification algorithm.
    First obtains a greedy solution ignoring fairness constrains. Then obtains a coreset by 
    getting greedy solutions in each group. Solves the MIS problem on the coreset.

    Args:
        features (np.ndarray): The feature vectors of the items 
        ids (np.ndarray): The ids of the items  
        groups (np.ndarray): The group (int) of the items
        k (int): The minimum number of total samples required
        num_groups (int): The number of unique groups in the data
        constraints (List[Tuple[int,int]]): The list of lower and upper limits on number of samples for each group 
        eps (float): The fraction to relax the diversity to get a solution.
        parallel_dist_update (bool, optional): Whether to update distances in parallel. Defaults to False.
        parallel_edge_creation (bool, optional): Whether to create coreset graph edges in parallel. Defaults to False.
    Returns:
        Tuple[set,float]: _description_
    """
    alg_start = time()
    initial_solution = set()
    initial_solution, diversity, _, _ = gonzales_algorithm(
        initial_solution, features, ids, k,parallel_dist_update=parallel_dist_update)
    init_sol_time = time() - alg_start
    logger.info(f"{init_sol_time=}")
    _initial_solution = initial_solution.copy()
    diversity_threshold = diversity
    working_data = {}
    G = None
    while True:
        start = time()
        _initial_solution, diversity_threshold, working_data = group_gonzales_algorithm(
            _initial_solution, features, ids, groups, k, eps, constraints, diversity_threshold, working_data,parallel_dist_update=parallel_dist_update)
        coreset_time = time()-start
        logger.info(f"{coreset_time=}")
        G = get_coreset_graph(_initial_solution, diversity_threshold,
                              features, ids, groups, constraints,G=G,parallel_edge_creation=parallel_edge_creation)
        start = time()
        final_solution = get_ILP_solution(G, k,time_limit=time_limit,verbose=verbose)
        ilp_time = time() - start
        logger.info(f"{ilp_time=}")
        if final_solution is None:
            diversity_threshold = diversity_threshold * (1.0 - eps)
            logger.info(
                f"ILP not feasible, decreasing diversity threshold to {diversity_threshold:e}")

        else:
            fmmd_time = time() - alg_start
            logger.info(f"{fmmd_time=}")
            final_diversity = compute_diversity_parallel(final_solution, features, ids)
            return final_solution, final_diversity


def compute_diversity(solution: set, features: np.ndarray, ids: np.ndarray) -> float:
    """Computes the diversity as the smallest pairwise distance between solution items

    Args:
        solution (set): The set of item ids in the solution
        features (np.ndarray): The feature vectors of the
        ids (np.ndarray): The ids of the items

    Returns:
        float: The minimum pairwise distance between solution items
    """
    _solution = sorted(list(solution))
    sorter = np.argsort(ids)
    solution_idxs = sorter[np.searchsorted(ids, _solution, sorter=sorter)]
    diversity = np.inf
    for i, j in combinations(solution_idxs, 2):
        diversity = min(diversity, parallel_utils.l2_dist(features[i], features[j]))
    return diversity

def compute_diversity_parallel(solution: set, features:np.ndarray, ids:np.ndarray) -> float:
    _solution = sorted(list(solution))
    sorter = np.argsort(ids)
    solution_idxs = sorter[np.searchsorted(ids, _solution, sorter=sorter)]
    dists = parallel_utils.pdist(features[solution_idxs])
    diversity = np.min(dists)
    return diversity
