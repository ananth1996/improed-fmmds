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
from fmmd.algorithms import fmmd, group_gonzales_algorithm, compute_diversity_parallel
import argparse
import logging
import datasets.ecco as ecco
from time import perf_counter as time
from fmmd.definitions import DATA_DIR

# %%


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["greedy", "fmmd"],
        default="fmmd",
        help="Which algorithm to use. Greedy will maximize diversity in each group greedily. FMMD will maximize global diversity.",
        type=str,
    )
    parser.add_argument("-k", type=int, help="Minimum number of samples needed")
    parser.add_argument(
        "--eps", type=float, help="The factor to relax threshold by", default=0.05
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Location of ecco data", default=DATA_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Location to store solution", default=DATA_DIR
    )
    parser.add_argument(
        "--pca", action="store_true", help="Whether to use PCA embeddigs"
    )
    parser.add_argument(
        "--parallel-dist-update",
        action="store_true",
            help="Update distances in parallel for Gonzales algorithm",
    )
    parser.add_argument(
        "--parallel-edge-creation",
        action="store_true",
        help="Create coreset graph edges in parallel",
    )

    parser.add_argument(
        "--time-limit", type=int, help="Time limit for ILP", default=300
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print verbose timing statements",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser


def write_solution_file(output_file: Path, solution: set, diversity: float):
    with open(output_file, "w") as fp:
        fp.write(f"#diversity={diversity}\n")
        for item in solution:
            fp.write(f"{item}\n")


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=args.loglevel)
    logger = logging.getLogger(__name__)
    ids, features, groups, constraints, k, num_samples_per_group = ecco.get_ecco(
        data_dir=args.data_dir, min_num_samples=args.k, PCA=args.pca
    )
    if args.algorithm == "greedy":
        start = time()
        # Run the group gonzales algorithm with an empty set
        # This selects elements in each group satisfying constraints
        #  greedily maximizing group diversity
        greedy_solution, _, _ = group_gonzales_algorithm(
            set(),
            features,
            ids,
            groups,
            num_samples_per_group,
            args.eps,
            constraints,
            parallel_dist_update=args.parallel_dist_update,
        )
        greedy_time = time() - start
        greedy_diversity = compute_diversity_parallel(greedy_solution, features, ids)
        logger.info(f"{greedy_time=}")
        print(f"Greedy Solution Diversity: {greedy_diversity}")
        if args.pca:
            output_file = (
                args.output_dir / f"greedy_diverse_balanced_samples_pca_{k}.txt"
            )
        else:
            output_file = args.output_dir / f"greedy_diverse_balanced_samples_{k}.txt"
        write_solution_file(output_file, greedy_solution, greedy_diversity)
    elif args.algorithm == "fmmd":
        solution, diversity = fmmd(
            features,
            ids,
            groups,
            k,
            constraints,
            args.eps,
            args.time_limit,
            parallel_dist_update=args.parallel_dist_update,
            parallel_edge_creation=args.parallel_edge_creation
        )
        if args.pca:
            output_file = args.output_dir / f"diverse_balanced_samples_pca_{k}.txt"
        else:
            output_file = args.output_dir / f"diverse_balanced_samples_{k}.txt"
        print(f"FMMD Solution Diversity: {diversity}")
        write_solution_file(output_file, solution, diversity)
