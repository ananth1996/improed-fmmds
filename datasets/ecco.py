import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .naive_balanced_sampling import find_num_samples_per_group
from fmmd.definitions import PROJECT_ROOT,DATA_DIR
from typing import Optional,List
logger = logging.getLogger(__name__)

def load_ecco_metadata(data_dir:Path=DATA_DIR):
    # read data files
    metadata = pd.read_csv(data_dir/"ecco_metadata.csv",dtype={"ecco_id":str})
    ids = pd.read_csv(data_dir/"sections_genre_with_seq.txt",header=None,sep="\t")
    ids.columns=["id","genre"]
    # split the ecco_section to ecco and section columns
    ids["ecco_id"] =ids.id.apply(lambda x: x.split("_")[0]).astype(str) 
    ids["section_id"] =ids.id.apply(lambda x: x.split("_")[1]).astype(int) 
    # merge the metadata
    ids = ids.merge(metadata,on="ecco_id",how="inner")
    # create group numbers based on ecco_module and publication_decade
    # dropna=False ensures NaN publication dates are also grouped
    ids["group_number"] = ids.groupby(["ecco_module","publication_decade"],dropna=False).ngroup()

    return ids

def load_ecco_features(data_dir:Path=DATA_DIR,PCA=True):
    if PCA: 
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
            section_ids = data["section_ids"]
    else:
        data = np.load(data_dir/"section_embeddings.npz")
        section_ids = data["section_ids"]

    if PCA:
        return section_ids,data["X_pca"]
    else:
        return section_ids,data["X"]


def get_ecco(data_dir:Path=DATA_DIR,min_num_samples:int=500,PCA:bool=False,group_subset:Optional[List[int]]=None):
    metadata = load_ecco_metadata(data_dir=data_dir)
    section_ids,features = load_ecco_features(data_dir,PCA=PCA)
    # ensure dataframe is in the same order as features
    df = metadata.set_index("id").loc[section_ids].reset_index()
    if group_subset:
        df = df[df.group_number.isin(group_subset)]
    group_counts = df.groupby("group_number").id.count()
    num_samples_per_group,total_number_of_samples = find_num_samples_per_group(group_counts,num_samples=min_num_samples)
    logger.info(f"Total number of samples: {total_number_of_samples} and maximum {num_samples_per_group} from each group")
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
    groups = df.group_number.values
    C = len(constraints)
    return ids,features,groups,constraints,total_number_of_samples,num_samples_per_group