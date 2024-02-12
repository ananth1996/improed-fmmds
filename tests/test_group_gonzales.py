from pathlib import Path
import pandas as pd
import numpy as np
from fmmd.algorithms import gonzales_algorithm,group_gonzales_algorithm
from fmmd_old.modular_alg import get_group_greedy_solution
from datasets.census import load_census_small,census_small_constraints
import fmmd_old.utils as utils
from fmmd.definitions import DATA_DIR
import pytest

EPS = 0.05
def _test_group_gonzales(elements,features,ids,groups,k,dist,check_sets=True):
    C = len(np.unique(groups))
    constraints = census_small_constraints[C][k]
    old_solution_idxs, old_diversity = get_group_greedy_solution(elements,k,C,constraints,dist,EPS)
    initial_solution,diversity,_,_ = gonzales_algorithm(set(),features,ids,k)
    new_soluton,new_diversity,_ = group_gonzales_algorithm(
        initial_solution=initial_solution,
        features=features,
        ids=ids,
        groups=groups,
        k=k,
        eps=EPS,
        constraints=constraints,
        diversity_threshold=diversity
    )
    # to get solution need to get idx from element
    old_solution = {elem.idx for elem in elements[list(old_solution_idxs)]}

    assert np.allclose(old_diversity,new_diversity)
    if check_sets:
        assert old_solution==new_soluton

def _test_group_gonzales_census(k,check_sets,grouping_type:int):
    ids,features,groups = load_census_small(grouping=grouping_type,data_dir=DATA_DIR)
    elements =[]
    for (i,f,g) in zip(ids,features,groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    elements = np.array(elements)
    _test_group_gonzales(elements,features,ids,groups,k,utils.euclidean_dist,check_sets)


@pytest.mark.parametrize("k", [5,10,15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [True])
def test_group_gonzales_census_2(k,check_sets):
    _test_group_gonzales_census(k,check_sets,grouping_type=1)

@pytest.mark.parametrize("k", [10,15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [True])
def test_group_gonzales_census_7(k,check_sets):
    _test_group_gonzales_census(k,check_sets,grouping_type=2)

@pytest.mark.parametrize("k", [15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [True])
def test_group_gonzales_census_14(k,check_sets):
    _test_group_gonzales_census(k,check_sets,grouping_type=3)

