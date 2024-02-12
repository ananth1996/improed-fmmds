from pathlib import Path
import pandas as pd
import numpy as np
from datasets.census import load_census_small
from fmmd.algorithms import gonzales_algorithm
from fmmd_old.modular_alg import get_initial_greedy_solution
import fmmd_old.utils as utils
from fmmd.definitions import DATA_DIR
import pytest

def _test_gonzales(elements,features,ids,k,dist,check_sets=True):
    old_solution_idxs, old_diversity = get_initial_greedy_solution(elements,k=k,dist=dist)
    # to get solution need to get idx from element
    old_solution = {elem.idx for elem in elements[list(old_solution_idxs)]}

    new_soluton,new_diversity,_,_ = gonzales_algorithm(
        initial_solution=set(),
        features=features,
        ids=ids,
        k=k
    )

    assert np.allclose(old_diversity,new_diversity)
    if check_sets:
        assert old_solution==new_soluton

@pytest.mark.parametrize("k", [5,10,15,20,25,30,35,40,45,50])
@pytest.mark.parametrize("check_sets", [True])
def test_gonzales_census_2(k,check_sets):
    ids,features,groups = load_census_small(grouping=1,data_dir=DATA_DIR)
    elements =[]
    for (i,f,g) in zip(ids,features,groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    elements = np.array(elements)
    _test_gonzales(elements,features,ids,k,utils.euclidean_dist,check_sets)



def test_gonzales_shuffle():
    rng = np.random.default_rng(1)
    ids,features,groups = load_census_small(grouping=1,data_dir=DATA_DIR)
    n = features.shape[0]
    elements =[]
    for (i,f,g) in zip(ids,features,groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    
    # shuffle all arrays in the same order
    perm = rng.permutation(n)
    features = features[perm]
    ids = ids[perm]
    groups = groups[perm]
    # need to shuffle the elements list as well
    elements = np.array(elements)[perm]
    
    k=15
    _test_gonzales(elements,features,ids,k,utils.euclidean_dist)
