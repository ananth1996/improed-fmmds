#%%
from IPython import get_ipython
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import fmmd_old.utils as utils
from tqdm.autonotebook import tqdm,trange
from fmmd.algorithms import fmmd
from fmmd_old.alg import scalable_fmmd_ILP
from datasets.census import load_census,census_constraints
from fmmd.definitions import PROJECT_ROOT,DATA_DIR
import argparse
import logging 
#%%
def get_parser():
    parser = argparse.ArgumentParser(description="Run FMMD-S on Census Dataset")
    parser.add_argument("-k",type=int,help="The number of samples required",default=10)
    parser.add_argument("-C",type=int,choices=[2,7,14],help="Number of groups in Census Dataset",default=2)
    parser.add_argument("--data_dir",type=Path,default=DATA_DIR,help="Location of data")
    parser.add_argument("--old",action="store_true",help="To use old version of FMMD-S algorithm")
    parser.add_argument("--eps",type=float,default=0.05,help="The factor to relax threshold by")
    parser.add_argument("--log-level",choices=[logging.WARNING,logging.INFO,logging.DEBUG],type=int,default=logging.INFO,help="The logging level")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=args.log_level)
    group_types = {2:1,7:2,14:3}
    constraints = census_constraints[args.C][args.k]
    ids,features,groups = load_census(grouping=group_types[args.C],data_dir=args.data_dir)
    elements = []
    for (i,f,g) in zip(ids,features,groups):
        elem = utils.Elem(int(i), int(g), f)
        elements.append(elem)
    elements = np.array(elements)
    old = True
    if args.old:
        solution,diversity,_ = scalable_fmmd_ILP(elements,args.eps,args.k,args.C,constraints,utils.euclidean_dist)
    else:
        solution,diversity = fmmd(features,ids,groups,args.k,constraints,args.eps)

    print(f"{diversity=}")
# %%
