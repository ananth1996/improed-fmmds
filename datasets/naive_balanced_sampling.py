#%%
from IPython import get_ipython
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    get_ipython().run_line_magic("load_ext", "Cython")
else:
    notebook = False
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import argparse
from fmmd.definitions import PROJECT_ROOT, DATA_DIR
#%%

def find_num_samples_per_group(group_counts:pd.Series,num_samples:int) -> Tuple[int,int]:
    """Finds the number of samples from each group for 
    uniform sampling. Accounts for group size variance. 

    Args:
        group_counts (pd.Series): Index is group number and value is number of elements in group
        num_samples (int): Minimum number of samples required   

    Returns:
        Tuple[int,int]: max number of samples from each group, total number of samples
    """
    def get_num_samples(gc:pd.Series,num_samples_per_group:int)->int:
        """Finds number of samples given a max from each group

        Args:
            gc (pd.Series): Index is group number and value is number of elements in group
            num_samples_per_group (int): Maximum number of samples from each group

        Returns:
            int: total number of samples 
        """
        # number of samples from groups with fewer than required
        val1 = gc[gc<=num_samples_per_group].sum()
        # number from rest of groups 
        val2 = num_samples_per_group*(gc>num_samples_per_group).sum()
        total_num_samples = val1+val2
        return total_num_samples
    num_samples_per_group = 0
    total_number_of_samples = get_num_samples(group_counts,num_samples_per_group)
    while True:
        total_number_of_samples = get_num_samples(group_counts,num_samples_per_group)
        if total_number_of_samples >=num_samples:
            break
        else:
            num_samples_per_group+=1
    return num_samples_per_group,total_number_of_samples
#%%
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('num_samples', type=int,help="Minimum number of samples required")
    parser.add_argument("--data_dir", type=Path,default=DATA_DIR,help="Data directory location")
    parser.add_argument("--output_dir", type=Path,default=DATA_DIR,help="Output directory location")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()     
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
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
    group_counts = ids.groupby("group_number").id.count()
    num_samples_per_group, total_num_samples = find_num_samples_per_group(group_counts,num_samples=num_samples)
    print(f"{num_samples_per_group=} and {total_num_samples=}")
    def sample_group(group:pd.DataFrame,num:int,random_state:int=1):
        if len(group) <= num:
            return group
        else:
            return group.sample(num,random_state=random_state,replace=False)

    # sample from each group based on the number 
    samples = ids.groupby("group_number",group_keys=False).apply(lambda group: sample_group(group,num_samples_per_group))
    output_fname = f"naive_balanced_samples_{total_num_samples}.csv"
    samples.to_csv(output_dir/output_fname,index=False)
    print(f"File saved to {output_dir/output_fname}")

#%%