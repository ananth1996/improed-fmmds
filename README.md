# Improved FMMD-S

An improved implementation of the FMMD-S algorithm from the paper:
>Wang, Y., Mathioudakis, M., Li, J., & Fabbri, F. (2023). Max-Min Diversification with Fairness Constraints: Exact and Approximation Algorithms. In Proceedings of the 2023 SIAM International Conference on Data Mining (SDM) (pp. 91–99). Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611977653.ch11 

## Table of Contents

- [Improved FMMD-S](#improved-fmmd-s)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Python](#python)
    - [Gurobi](#gurobi)
    - [Cython](#cython)
  - [Balanced Sampling](#balanced-sampling)
    - [Old FMMD-S implementation Bottlenecks](#old-fmmd-s-implementation-bottlenecks)
  - [Benchmarks](#benchmarks)
    - [Relative Speedups](#relative-speedups)
    - [Multicore Performance](#multicore-performance)
    - [Balanced Sampling Performance](#balanced-sampling-performance)
  - [Limitations and Future Updates](#limitations-and-future-updates)

## Setup 

### Python 

```bash
conda env create -n fmmd -f env.yaml
```

### Gurobi

The FMMD-S algorithm uses [Gurobi](https://www.gurobi.com) to solve the MIS problem as explained in the paper. 

The gurobi optimizer can be used without a license for small datasets, but for larger datasets it is recommended to have an [academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/).

### Cython
This library uses Cython to parallelize several utility functions. 

Run the following command to build the module:
```bash
python setup.py build_ext --inplace
```

After building, the module is used like this:

```python
from fmmd.parallel_utils pdist
import numpy as np

X = np.random.random(1000,512)
d = pdist(X)
```

The `.pyx` is found in [ctython/parallel_utils.pyx](./cython/parallel_utils.pyx) and can be modified to add additional metrics and functionalities.

## Balanced Sampling

This library was developed specifically to balanced sampling.

Balanced sampling is when there is a need to uniformly sample from groups instead of the typical proportional 
sampling. Specifically, we wanted to perform balanced sampling when the distribution of groups was very skewed. This means that there are some groups with very few items and some groups with a lot of items. Furthermore, if `min_group_size` is the size of the smallest group and `k>min_group_size` then we might be forced to take all items from smaller groups. 

For example, assume we have the following group sizes:
|group|count |
|---:|----:|
|  1 |   2 |
|  2 |   5 |
|  3 |  10 |
|  4 | 200 |

Now, if we want to sample at least ten items uniformly `k>=10` then, we would need to select all items from group 1 and 3 items from the remaining groups. This would result in `k=11` items where we take a maximum of `3` from each group. This is what the `find_num_samples_per_group` method in the [datasets/naive_balanced_sampling.py](./datasets/naive_balanced_sampling.py) does. It returns `num_samples_per_group,total_number_of_samples` for a given group distribution and minimum number of samples. 


### Old FMMD-S implementation Bottlenecks

This sort of balanced sampling constraints causes several bottlenecks in the original FMMD-S implementation:

1. When entire groups need to be selected, the diversity threshold is relaxed several times until the constraints is not `under_capped`
2. The coreset graph typically is much larger as several groups are added directly. This results in a long time to compute the edges which satisfy the diversity constraints
3. When the ILP is infeasible steps 1 and 2 need to performed again

We fix these issues in the new implementations. Along with these fixes, a parallelization of several portions of the algorithm offer a general speedup compared the original implementation.


## Benchmarks


Running the benchmark on the Census dataset from the original paper.

New and improved algorithm implementation
```bash
python census_single_solution.py -k=10 -C=2
```

Original algorithm implementation

```bash
python census_single_solution.py -k=10 -C=2 --old
```

### Relative Speedups 

For the full census dataset. `k` is the number of samples and `C` is the number of groups. 

Experiments on an Apple MacBook Pro with a M2 Pro chip (12 cores)

| k| C | Original (in seconds) | New (in seconds) | Relative Speedup |
|:---|---:|---:|---:|---:|
10 | 2 | 112.67 | 10.49 |  10.73x |
10 | 7 | 106.24 | 10.71 | 9.91x |
50 | 14 | 474.152 | 19.06  | 24.88x |
100 | 14 | 3670.781 | 39.722  | 92.41x |

### Multicore Performance 

The parallel options are only useful when the number of dimensions increase.
Therefore, we use the full `768` dimensional embeddings for the ECCO dataset and select `k=500` balanced samples as follows:
```bash
python ecco_balanced_samples.py -k=500  --eps=0.5
```


The `fmmd` method found in [fmmd/algorithms.py](./fmmd/algorithms.py) has two options to utilize multiple cores:
1. `--parallel-dist-update`: Update the solution distances in parallel for the Gonzales algorithm
2. `--parallel-edge-creation`: Find the coreset edges in parallel which are below the diversity threshold

Both these options speed up the algorithm in different ways. For the ECCO dataset and balanced sampling we observe the following trends:


| `--parallel-dist-update`| `--parallel-edge-creation` | Running Time | Relative |
|:---:|:---:|---:|---:|
| ❌|❌ | 4444.224 | 4.55 |
| ✅| ❌| 1781.913 | 1.82 |
|❌ |✅ | 3625.860 | 3.71 |
| ✅| ✅ | 976.961 | 1.00 |

The above ablation results indicate the majority of the speedup is gained from the parallel edge creation due to the large number of items in the coreset.

All experiments were conducted on a Linux machine with 32 cores and 30GB RAM.

### Balanced Sampling Performance

The benefit of many cores comes when there is balanced sampling performed.

For the ECCO dataset, there are 4.2 million items and 137 groups with a very skewed distribution. The requirement was at least 3000 balanced samples. With the group distribution resulted in the following:
1. Maximum of `26`  samples from each group
2. `3063` total samples


The items are `768` dimensional embedding which take 3min to load the 22GB vectors.


The FFMD-S solution takes 9250.9 seconds (2.5hrs).


The FMMD-S algorithm has the following stages:
1. Finding an initial greedy solution without thinking of buckets: **1615.6 sec**
2. Finding a core-set of the data from buckets starting from the initial samples : **1832.83 sec**
3. Computing pair-wise distances and selecting ones which are below a diversity threshold: **5779.5  sec**
4. Creating a coreset graph with nodes from 2 and edges from 3: **0.69 sec**
5. Solving an Integer Linear Program of a coreset from 4 to get final solution: **17.27 sec**

All these times are on a system with 40 cores and 80GB RAM (of which only 25GB was actually used)
The main bottleneck as seen is step 3. This is because we end up with 274,398 items in the coreset leading to the computation of nearly 37B pair-wise distances. After step 3 we only have 253 edges below the required threshold, so the rest of the algorithm is very quick.


## Limitations and Future Updates

There are some limitations compared to the original implementations. These will be addressed as the need arises.

1. Only supports L2 distance metric
2. No ability to generate multiple solutions in parallel to obtain best overall diversity
   

