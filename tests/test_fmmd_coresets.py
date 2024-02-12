from pathlib import Path
import pandas as pd
import numpy as np
from datasets.census import load_census_small,census_small_constraints
from fmmd.algorithms import fmmd,gonzales_algorithm,group_gonzales_algorithm,get_coreset_graph,get_ILP_solution
from fmmd_old.modular_alg import fmmd_coresets
import fmmd_old.utils as utils
import pytest
import logging 
import networkx as nx
from fmmd.definitions import DATA_DIR
logger = logging.getLogger(__name__)
EPS = 0.05


def _new_fmmd_coresets(features,ids,groups,k,constraints,eps):
    graphs = []
    initial_solution = set()
    initial_solution, diversity, _, _ = gonzales_algorithm(
        initial_solution, features, ids, k)
    _initial_solution = initial_solution.copy()
    diversity_threshold = diversity
    working_data = {}
    G = None
    while True:
        _initial_solution, diversity_threshold, working_data = group_gonzales_algorithm(
            _initial_solution, features, ids, groups, k, eps, constraints, diversity_threshold, working_data)
        logger.debug("Group Solution Found")
        G = get_coreset_graph(_initial_solution, diversity_threshold,
                              features, ids, groups, constraints,G=G)
        graphs.append(G.copy())
        logger.debug("Graph Created")
        final_solution = get_ILP_solution(G, k)
        if final_solution is None:
            diversity_threshold = diversity_threshold * (1.0 - eps)
            logger.info(
                f"ILP not feasible, decreasing diversity threshold to {diversity_threshold:e}")
        else:
            return graphs


def _test_fmmd_coresets(elements,features,ids,groups,k,dist,check_sets=True):
    C = len(np.unique(groups))
    constraints = census_small_constraints[C][k]
    # make a dict of constraints
    _constraints = {i:constraints[i] for i in range(C)}
    old_coresets = fmmd_coresets(V=elements, k=k, EPS=EPS, C=C,constr=constraints,dist=dist)
    new_coresets = _new_fmmd_coresets(features,ids,groups,k,_constraints,eps=EPS)
    assert len(old_coresets) == len(new_coresets)
    for oldG,newG in zip(old_coresets,new_coresets):
        assert oldG.number_of_nodes() == newG.number_of_nodes()
        assert oldG.number_of_edges() == newG.number_of_edges()
        assert nx.is_isomorphic(oldG,newG,node_match=_check_nodes)

def _check_nodes(node1:dict,node2:dict):
    # ensure that the nodes belong to the same group
    # and that nodes have same id
    return (node1["group"]==node2["group"])and(node1["id"]==node2["id"])

def _test_fmmd_coresets_census(k,check_sets,grouping_type:int):
    ids,features,groups = load_census_small(grouping=grouping_type,data_dir=DATA_DIR)
    elements =[]
    for (i,f,g) in zip(ids,features,groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    elements = np.array(elements)
    _test_fmmd_coresets(elements,features,ids,groups,k,utils.euclidean_dist,check_sets)


@pytest.mark.parametrize("k", [5,10,15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [False])
def test_fmmd_coresets_census_2(k,check_sets):
    _test_fmmd_coresets_census(k,check_sets,grouping_type=1)

@pytest.mark.parametrize("k", [10,15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [False])
def test_fmmd_coresets_census_7(k,check_sets):
    _test_fmmd_coresets_census(k,check_sets,grouping_type=2)

@pytest.mark.parametrize("k", [15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [False])
def test_fmmd_coresets_census_14(k,check_sets):
    _test_fmmd_coresets_census(k,check_sets,grouping_type=3)

