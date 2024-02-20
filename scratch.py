#%%
from IPython import get_ipython

if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fmmd.definitions import PROJECT_ROOT, DATA_DIR
from time import perf_counter as time
import timeit
#%%



def time_statement(stmt,variables):
    timer = timeit.Timer(stmt=stmt,globals=variables)
    # find ideal number of iterations
    num,_ = timer.autorange()
    repeat = 3
    timings = timer.repeat(repeat,number=num)
    running_time = min(timings)/num
    return running_time


def benchmark_parallel_edges():
    from fmmd import parallel_utils 
    ns = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ds = [2,4,16,32,64,128,256,512]
    rows = []
    for n in ns:
        for d in ds:
            d = int(d)
            rng = np.random.default_rng(seed=42)
            features = rng.random(size=(n,d))
            # print(f"{n=:,} {d=:,}")
            parallel_edge_stmt = "parallel_utils.edges(features,0.01)"
            parallel_edge_time = time_statement(stmt=parallel_edge_stmt,variables=locals())
            # print(f"{parallel_edge_time=}")
            sequential_edge_stmt = "parallel_utils.edges_sequential(features,0.01)"
            sequential_edge_time = time_statement(sequential_edge_stmt,variables=locals())
            # print(f"{sequential_edge_time=}")
            speedup = sequential_edge_time/parallel_edge_time
            # print(f"Speedup = {speedup:.2f}x")
            rows.append({
                "n":n,
                "d":d,
                "parallel":parallel_edge_time,
                "sequential":sequential_edge_time,
                "speedup":speedup
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR/"parallel_edges_benchmarking.csv",index=False)
    return df


def benchmark_parallel_dist_update():
    from fmmd.algorithms import gonzales_algorithm
    ns = [2**i for i in range(15,22)]
    ds = [2,4,16,32,64,128,256,512]
    rows = []
    for n in ns:
        for d in ds:
            d = int(d)
            rng = np.random.default_rng(seed=42)
            features = rng.random(size=(n,d))
            ids = np.arange(n)
            # print(f"{n=:,} {d=:,}")
            # start = time()
            # gonzales_algorithm(set(),features,ids,k=10,parallel_dist_update=True)
            # print(locals())
            parallel_stmt = "gonzales_algorithm(set(),features,ids,k=10,parallel_dist_update=True)"
            parallel_time = time_statement(stmt=parallel_stmt,variables=locals())
            # parallel_time = time() - start
            print(f"{parallel_time=}")
            # start = time()
            # gonzales_algorithm(set(),features,ids,k=10,parallel_dist_update=False)
            sequential_stmt = "gonzales_algorithm(set(),features,ids,k=10,parallel_dist_update=False)"
            sequential_time = time_statement(sequential_stmt,variables=locals())
            # print(f"{sequential_time=}")
            # sequential_time = time()-start
            speedup = sequential_time/parallel_time
            # print(f"Speedup = {speedup:.2f}x")
            rows.append({
                "n":n,
                "d":d,
                "parallel":parallel_time,
                "sequential":sequential_time,
                "speedup":speedup
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR/"parallel_dist_benchmarking.csv",index=False)
    return df
# %%
def plot_speedup(df):
    from matplotlib import cm, ticker,colors
    x_data = df.n.values
    y_data = df.d.values
    z_data = df.speedup.values
    # z_data = np.array([ -1/z if z<1 else z for z in z_data ])
    x_items = df.n.unique()
    y_items = df.d.unique()
    idx = np.lexsort((y_data, x_data)).reshape(len(x_items), len(y_items))
    # cs = plt.contourf(x_data[idx], y_data[idx], z_data[idx],locator=ticker.LogLocator(subs='all'))
    cmap = cm.bwr
    norm=colors.TwoSlopeNorm(vcenter=1)
    # norm=colors.LogNorm()
    cs = plt.contourf(x_data[idx], y_data[idx], z_data[idx],norm=norm,cmap=cmap)
    plt.colorbar(cs,label="speedup")
    plt.xscale("log",base=2)
    plt.yscale("log",base=2)
    plt.xlabel("Number of data points (n)")
    plt.ylabel("Dimension of data points (d)")
# %%
    
if __name__ == "__main__":
    benchmark_parallel_dist_update()
    benchmark_parallel_edges()
