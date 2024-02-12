from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.resolve()
import sys
sys.path.append(str(project_root/"src"/"mahadeva"))
import pandas as pd
import numpy as np
from fmmd.algorithms import fmmd
from fmmd_old.modular_alg import fmmd as old_fmmd
import fmmd_old.utils as utils
from datasets.naive_balanced_sampling import find_num_samples_per_group
from fmmd.definitions import DATA_DIR
import pytest

EPS = 0.05


def load_ecco_df(data_dir:Path=DATA_DIR):
    if  not (data_loc:=Path(data_dir/"section_embeddings_pca.npz")).exists():
        print("PCA dimensions do not exist. Computing now")
        data = np.load(data_dir/"section_embeddings.npz") 
        X = data["X"]
        section_ids = data["section_ids"]
        from sklearn.decomposition import IncrementalPCA
        pca = IncrementalPCA(n_components=2)
        X_pca = pca.fit_transform(X)
        np.savez_compressed(data_dir/"section_embeddings_pca.npz",section_ids=section_ids,X_pca=X_pca)
    else:
        data = np.load(data_loc)
        X_pca = data["X_pca"]
        section_ids = data["section_ids"]

    genre_data_loc = data_dir
    metadata = pd.read_csv(genre_data_loc/"ecco_metadata.csv",dtype={"ecco_id":str})
    section_ids_df = pd.DataFrame(section_ids,columns=["id"])
    section_ids_df["ecco_id"] = section_ids_df.id.apply(lambda x: x.split("_")[0]).astype(str)
    section_ids_df["section_id"] = section_ids_df.id.apply(lambda x: x.split("_")[1]).astype(int)
    section_ids_df = section_ids_df.merge(metadata,on="ecco_id",how="inner")
    pca_embeds_df = pd.DataFrame(X_pca,columns=["PCA_dim_1","PCA_dim_2"])
    pca_df = pd.concat((section_ids_df,pca_embeds_df),axis=1)
    pca_df["ecco_module"] = pca_df["ecco_module"].astype("category")
    pca_df["publication_decade"] = pca_df["publication_decade"].astype("category")
    # Get the group number 
    pca_df["group_number"] = pca_df.groupby(["ecco_module","publication_decade"],dropna=False).ngroup()
    pca_df = pca_df.reset_index()
    return pca_df


def _load_ecco(group_subset=[1,2,3],min_number_of_samples=50):
    df = load_ecco_df()
    # Restrict to first 3 groups
    df = df[df.group_number.isin(group_subset)]
    # df = df.sample(1000,random_state=1)
    group_counts = df.groupby("group_number").id.count()
    num_samples_per_group,total_number_of_samples = find_num_samples_per_group(group_counts,num_samples=min_number_of_samples)
    constraints = dict()
    for group,count in zip(group_counts.index.values,group_counts.values):
        if count<num_samples_per_group:
            lb = count
            ub = count
        else:
            lb=num_samples_per_group
            ub = num_samples_per_group
        constraints[group]=(lb,ub)
    ids = df.id.values
    features = df[["PCA_dim_1","PCA_dim_2"]].values
    features = features.copy(order='C')
    groups = df["group_number"].values
    C = len(constraints)
    k=total_number_of_samples

    return ids,features,groups,constraints,k

def _test_fmmd(elements,_constraints,features,ids,groups,constraints,k,dist,check_sets=False):
    C = len(np.unique(groups))
    old_solution_idxs, old_diversity = old_fmmd(V=elements, k=k, EPS=EPS, C=C,constr=_constraints,dist=dist)
    new_soluton,new_diversity = fmmd(features,ids,groups,k,constraints,eps=EPS)
    # to get solution need to get idx from element
    old_solution = {elem.idx for elem in elements[list(old_solution_idxs)]}

    assert np.allclose(old_diversity,new_diversity)
    if check_sets:
        old_solution==new_soluton

def test_ecco_undercapped(check_sets=True):
    ids,features,groups,constraints,k = _load_ecco()
    uniq_groups = np.unique(groups)
    C = len(uniq_groups)
    # old solution needs groups to start from 0 and be continuos
    _group_mapping = {grp:i for i,grp in enumerate(uniq_groups)}
    _group_mapping_r = {v:k for k,v in _group_mapping.items()}
    _group_mapping_v = np.vectorize(_group_mapping.get)
    _groups = _group_mapping_v(groups)
    # also ensure constraints in the new group order
    _constraints = [constraints[_group_mapping_r[c]] for c in range(C)]
    
    elements =[]
    for (i,f,g) in zip(ids,features,_groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    elements = np.array(elements)
    _test_fmmd(elements,_constraints,features,ids,groups,constraints,k,utils.euclidean_dist,check_sets)


# def _test_add_small_groups(features,ids,groups,constraints,k,rounding):
#     eps = 10**(-rounding)
#     initial_solution1 = set()
#     for group,(lower_bound,_) in constraints.items():
#         group_mask = groups == group
#         group_size = sum(group_mask)
#         # take all group items if smaller than lower limit
#         if group_size <= lower_bound:
#             initial_solution1.update(ids[group_mask])
#     initial_solution1_updated,diversity,_,_ = gonzales_algorithm(initial_solution1,features,ids,k,_tqdm=False)
#     group_solution1,diversity_threshold1,_ = group_gonzales_algorithm(
#         initial_solution1_updated,
#         features,
#         ids,
#         groups,
#         k,
#         eps,
#         constraints,
#         diversity)
    
#     initial_solution2,diversity,_,_ = gonzales_algorithm(set(),features,ids,k,_tqdm=False)
#     group_solution2,diversity_threshold2,_ = group_gonzales_algorithm(
#         initial_solution2,
#         features,
#         ids,
#         groups,
#         k,
#         eps,
#         constraints,
#         diversity)
    
#     # The latter relaxing diversity threshold in epsilon 
#     # therefore, the method will produce different results
#     # However, the diversity threshold should be within the error bound of epsilon
#     # hence, round them to correct number of places and 
#     assert np.allclose(np.round(diversity_threshold1,rounding),np.round(diversity_threshold2,rounding))
#     # assert group_solution1 == group_solution2


# def test_ecco_undercapped():
#     ids,features,groups,constraints,k = _load_ecco()
#     _test_add_small_groups(features,ids,groups,constraints,k,rounding=3)