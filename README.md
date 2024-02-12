# Improved FMMDS

## Python 

```bash
conda env create -n fmmd -f env.yaml
```


## Cython

```bash
python setup.py build_ext --inplace
```


## Benchmark 

New and improved algorithm implementation
```bash
python census_single_solution.py -k=10 -C=2
```

Old algorithm implementation

```bash
python census_single_solution.py -k=10 -C=2
```

