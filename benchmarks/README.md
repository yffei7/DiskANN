# PageANN Benchmarks

This directory contains the benchmark runner, benchmark configs, and adapter source files used to compare PageANN with other systems.

## Supported Systems

The benchmark pack currently supports:

| System | Paper | Codebase |
| --- | --- | --- |
| PageANN | This repository | This repository |
| DiskANN | [DiskANN paper](https://arxiv.org/abs/2105.09613) | [microsoft/DiskANN `diskv2`](https://github.com/microsoft/DiskANN/tree/diskv2) |
| Greator | [Greator paper](https://www.vldb.org/pvldb/vol19/p495-yu.pdf) | [iDC-NEU/Greator](https://github.com/iDC-NEU/Greator) |
| OdinANN | [OdinANN paper](https://www.usenix.org/system/files/fast26-guo.pdf) | [thustorage/PipeANN](https://github.com/thustorage/PipeANN) |

## Datasets

| Dataset | Points | Type | Dimension |
| --- | ---: | --- | ---: |
| `arxiv` | 122,112 | `float32` | 768 |
| `sift` | 1,000,000 | `float32` | 128 |
| `ms_turing` | 100,000,000 | `float32` | 100 |
| `spacev` | 100,000,000 | `int8` | 100 |
| `sift-100M` | 100,000,000 | `uint8` | 128 |

## Layout

- `runner/`: benchmark entrypoints
- `config/`: benchmark profiles and dataset map
- `adapters/`: benchmark adapter source

## Build

Build PageANN first:

```bash
cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release
cmake --build ../build --target pageann_benchmark -j
```

## Install External Adapters

For each external system:

1. Copy the adapter source into that repo as `tests/benchmark_adapter.cpp`.
2. Add a `benchmark_adapter` target to that repo's `tests/CMakeLists.txt`.
3. Configure and build the `benchmark_adapter` target in `build/`.

Example copy commands:

```bash
cp adapters/diskann/benchmark_adapter.cpp ../DiskANN/tests/benchmark_adapter.cpp
cp adapters/greator/benchmark_adapter.cpp ../Greator/tests/benchmark_adapter.cpp
cp adapters/odinann/benchmark_adapter.cpp ../PipeANN/tests/benchmark_adapter.cpp
```

Add this target to `../DiskANN/tests/CMakeLists.txt`:

```cmake
add_executable(benchmark_adapter benchmark_adapter.cpp)
if(MSVC)
    target_link_options(benchmark_adapter PRIVATE /MACHINE:x64 /DEBUG:FULL)
    target_link_libraries(benchmark_adapter debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
    target_link_libraries(benchmark_adapter optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else()
    target_link_libraries(benchmark_adapter ${PROJECT_NAME} -ltcmalloc aio)
endif()
```

Add this target to `../Greator/tests/CMakeLists.txt`:

```cmake
add_executable(benchmark_adapter benchmark_adapter.cpp)
target_link_libraries(benchmark_adapter ${PROJECT_NAME} aio -ltbb)
```

Add this target to `../PipeANN/tests/CMakeLists.txt`:

```cmake
add_executable(benchmark_adapter benchmark_adapter.cpp)
target_link_libraries(benchmark_adapter ${PROJECT_NAME})
```

Example build commands:

```bash
cmake -S ../DiskANN -B ../DiskANN/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../DiskANN/build --target benchmark_adapter -j

cmake -S ../Greator -B ../Greator/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../Greator/build --target benchmark_adapter -j

cmake -S ../PipeANN -B ../PipeANN/build -DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON
cmake --build ../PipeANN/build --target benchmark_adapter -j
```

## Run

Quick verification:

```bash
./runner/run_small_verify.sh
```

Single command:

```bash
python3 runner/run_comparison.py --profile small_verify
```

Real datasets:

```bash
python3 runner/run_comparison.py --profile real_standard --datasets arxiv,sift
python3 runner/run_comparison.py --profile real_100m --datasets ms_turing,spacev,sift-100M
```

Outputs are written under:

- `benchmarks/runs/<run_id>/...`
- `benchmarks/latest/...` when latest publishing is enabled

## Config

- [`config/comparison_config.json`](config/comparison_config.json): benchmark profiles and per-system tuning
- [`config/benchmark_datasets_config.json`](config/benchmark_datasets_config.json): benchmark-runner dataset map
