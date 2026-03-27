# PageANN

PageANN is a page-based approximate nearest neighbor system with online insert and delete support.

## Layout

- `include/`, `src/`: core library code and support infrastructure
- `include/pageann/`, `src/pageann/`: PageANN update and search path
- `tests/benchmark_adapter.cpp`: normalized benchmark entrypoint
- `tests/test_update_experiment.cpp`: native PageANN update demo
- `benchmarks/`: experiment runner, configs, and adapter files

## Build

```bash
sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev libboost-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pageann_benchmark pageann_update_demo build_disk_index search_disk_index -j
```


## Run

SSD index build and search:

```bash
./build/tests/build_disk_index ...
./build/tests/search_disk_index ...
```

PageANN update demo and benchmark adapter:

```bash
./build/tests/pageann_update_demo ...
./build/tests/pageann_benchmark ...
```

## Benchmarking

The benchmark pack is vendored under [`benchmarks/`](benchmarks/).

- runner: [`benchmarks/runner/run_comparison.py`](benchmarks/runner/run_comparison.py)
- configs: [`benchmarks/config/`](benchmarks/config/)
- quick check: `./benchmarks/runner/run_small_verify.sh`

For benchmark setup, supported systems, datasets, and external adapter installation, see [`benchmarks/README.md`](benchmarks/README.md).
