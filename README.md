# Improved FMMD-S

An improved implementation of the FMMD-S algorithm from the paper:
>Wang, Y., Mathioudakis, M., Li, J., & Fabbri, F. (2023). Max-Min Diversification with Fairness Constraints: Exact and Approximation Algorithms. In Proceedings of the 2023 SIAM International Conference on Data Mining (SDM) (pp. 91–99). Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611977653.ch11 


## Setup 

### Python 

```bash
conda env create -n fmmd -f env.yaml
```


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


## Benchmarking


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

| k| C | Original (in seconds) | New (in seconds) | Relative Speedup |
|:---|---:|---:|---:|---:|
10 | 2 | 112.67 | 10.49 |  10.73x |
10 | 7 | 106.24 | 10.71 | 9.91 |
50 | 14 | 474.152 | 19.06  | 24.88 |
100 | 14 | 3670.781 | 39.722  | 92.41 |


### Multicore performance

The benefit of many cores comes when there is balanced sampling performed.



## Limitations and Future Updates

There are some limitations compared to the original implementations. These will be addressed as the need arises.

1. Only supports L2 distance metric
2. No ability to generate multiple solutions in parallel to obtain best overall diversity
   

   