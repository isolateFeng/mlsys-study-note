# [MLIR] matmul cpu性能测试

## 0. 前言

本文以带有 `transform.sequence` 的 `matmul.mlir` 为输入，以 `cpu` 为目标后端，经过一些列变换，最终成为 二进制代码，并测试得到代码的性能。（函数执行时间 + google benchmark）

完整流程如下：

- matmul.mlir  —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll
- llvm.ll(有 `_mlir_ciface_matmul_tensors_1`接口) + test.cpp —> llvm.o —> a.out
- run a.out

第一节中，实现了对 matmul.mlir 的完整变换和调用流程，一直到 cpu 上的二进制。

第二节中，更改了 matmul.mlir 的transform.sequence 中 tile 变换使用的 tile_size，观察 tile_size 对变换性能（函数执行时间）的影响。

这里的 tile_size 是指 transform.structured.tile 后面的三维数组参数 。

第三节中，将变换性能的评价由 函数执行时间 改为了 google benchmark，依旧是改变 tile_size，测试性能影响。

第四节中，一些还没实现的想法（挺重要的吧

我们首先需要 **将编译好mlir后的lib目录添加到全局的库目录中**，方便后续链接使用

```bash
export LD_LIBRARY_PATH="/path/to/llvm-project/build/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/path/to/llvm-project/build/lib:$LIBRARY_PATH"
```

以及我们使用的输入 `matmul.mlir` 如下：(主要进行了 tile 和 vectorize 操作)

```llvm
module {
    func.func @matmul_tensors_1(
        %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
        %arg2: tensor<128x128xf32>)
            -> tensor<128x128xf32> {
        %0 = linalg.matmul { test.attrA, test.attrC }
                            ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%arg2: tensor<128x128xf32>)
            -> tensor<128x128xf32>
        func.return %0 : tensor<128x128xf32>
    }

    transform.with_pdl_patterns {
        ^bb0(%arg0: !pdl.operation):
        pdl.pattern @pdl_target_attrA : benefit(1) {
            %args = operands
            %results = types
            %attr = attribute
            %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
            rewrite %0 with "transform.dialect"
        }

        pdl.pattern @pdl_target_attrC : benefit(1) {
            %args = operands
            %results = types
            %attr = attribute
            %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrC" = %attr}-> (%results : !pdl.range<type>)
            // TODO: we don't want this, but it is the required terminator for pdl.pattern
            rewrite %0 with "transform.dialect"
        }

        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %0 = pdl_match @pdl_target_attrA in %arg1 : (!pdl.operation) -> !pdl.operation
            transform.structured.tile %0 [8, 8, 8] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
            %1 = pdl_match @pdl_target_attrC in %arg1 : (!pdl.operation) -> !pdl.operation
            %2 = transform.get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
            transform.structured.vectorize %2 : (!pdl.operation) -> !pdl.operation
        }
    }
}
```

输入也可以写成

```llvm
// RUN: mlir-opt -llvm-request-c-wrappers --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule matmul.mlir \
// RUN: | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1" -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-translate -mlir-to-llvmir \
// RUN: | llc -filetype=obj -o llvm.o

module {
    func.func @matmul_tensors(
        %A: tensor<128x128xf32>, %B: tensor<128x128xf32>,
        %D: tensor<128x128xf32>)
            -> tensor<128x128xf32> {
        %ret = linalg.matmul ins(%A, %B: tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%D: tensor<128x128xf32>)
            -> tensor<128x128xf32>
        func.return %ret : tensor<128x128xf32>
    }

    transform.sequence failures(propagate) {
    ^bb1(%module: !pdl.operation):
        %matmul = transform.structured.match ops{["linalg.matmul"]}
            in %module : (!pdl.operation) -> !pdl.operation
        %func = transform.structured.match ops{["func.func"]}
            in %module : (!pdl.operation) -> !pdl.operation
        transform.structured.tile %matmul [16, 2, 1]
         : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
        transform.structured.vectorize %func : (!pdl.operation) -> !pdl.operation
    }
}
```

## 1. 完整流程

- matmul.mlir  —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll

在对于matmul.mlir进行变换时，需要添加llvm-request-c-wrappers选项，通过如下指令生成transformed.mlir

```bash
mlir-opt -llvm-request-c-wrappers --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule matmul.mlir -o transformed.mlir
```

将transformed.mlir下降到llvm dialect，并且通过mlir-translate生成llvm.ll

```bash
mlir-opt --pass-pipeline="builtin.module(sparse-compiler{ parallelization-strategy=dense-outer-loop vl=8 reassociate-fp-reductions=1 enable-index-optimizations=1 })" transformed.mlir -o llvm.mlir
mlir-translate llvm.mlir --mlir-to-llvmir -o llvm.ll
```

将llvm.ll生成目标文件llvm.o

```bash
llc -filetype=obj llvm.ll -o llvm.o
```

- llvm.ll(有 `_mlir_ciface_matmul_tensors_1`接口) + test.cpp —> llvm.o —> a.out

在 llvm.ll 中有用于调用的函数接口`_mlir_ciface_matmul_tensors_1(&arg3, &arg0, &arg1, &arg2)` ，其中，arg0，arg1和arg2与`matmul_tensors_1`的参数相对应，而arg3则会保存`matmul_tensors_1`返回的memref的结果

通过c++编写test.cpp，其中需要自行定义结构体MemRefDescriptor和mlir中中的memref数据结构进行对应，具体数据结构布局详见[https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types](https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types)

```c++
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define s1 128
#define s2 128

using namespace std;

template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
void _mlir_ciface_matmul_tensors_1(MemRefDescriptor<float, 2>* arg3, MemRefDescriptor<float, 2>* arg0,MemRefDescriptor<float, 2>* arg1,MemRefDescriptor<float, 2>* arg2);
}

int main(){
  float *a = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *b = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *c = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *d = (float*)malloc(sizeof(float) * (long long)s1 * s2);

  for(int i=0;i<s1*s2;i++){
      a[i] = 1;
      b[i] = 1;
      c[i] = 0;
      d[i] = 0;
  }

  MemRefDescriptor<float, 2> arg0 = {
      a,    // allocated
      a,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg1 = {
      b,    // allocated
      b,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg2 = {
      c,    // allocated
      c,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg3 = {
      d,    // allocated
      d,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  _mlir_ciface_matmul_tensors_1(&arg3, &arg0, &arg1, &arg2);

  free(a);
  free(b);
  free(c);

  return 0;
}
```

使用g++链接时，需要链接自己编译好的mlir库中的libmlir_runner_utils.so和libmlir_c_runner_utils.so

```bash
g++ llvm.o test.cpp -lmlir_runner_utils -lmlir_c_runner_utils
```

- run a.out

可以修改 `test.cpp` 来实现 测试 `_mlir_ciface_matmul_tensors_1(&arg3, &arg0, &arg1, &arg2);` 函数的运行时间

## 2. 变换tile_size测试

观察 输入 `matmul.mlir` 中的 `transform.sequence` 可以发现，基本只进行了 tile 和 vectorize 操作，所以不妨修改下 tile_size 来看看函数执行性能

- 首先在 `test.cpp` 中 输出增加函数运行时间

`test.cpp` （用于调用llvm.o文件生成的接口来测试）如下：

```cpp
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define s1 128
#define s2 128

using namespace std;

template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
void _mlir_ciface_matmul_tensors_1(MemRefDescriptor<float, 2>* arg3, MemRefDescriptor<float, 2>* arg0,MemRefDescriptor<float, 2>* arg1,MemRefDescriptor<float, 2>* arg2);
}

int main(){
  // tensor<128x128xf32>
  // %arg0, %arg1, %arg2, %arg3: memref<128x128xf32>
  float *a = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *b = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *c = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *d = (float*)malloc(sizeof(float) * (long long)s1 * s2);

  for(int i=0;i<s1*s2;i++){
      a[i] = 1;
      b[i] = 1;
      c[i] = 0;
      d[i] = 0;
  }

  MemRefDescriptor<float, 2> arg0 = {
      a,    // allocated
      a,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg1 = {
      b,    // allocated
      b,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg2 = {
      c,    // allocated
      c,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg3 = {
      d,    // allocated
      d,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };
  auto start = chrono::high_resolution_clock::now();

  _mlir_ciface_matmul_tensors_1(&arg3, &arg0, &arg1, &arg2);

  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start; 
  std::cout<< "matmul.mlir cpu_pipeline_test success!" << std::endl;
  std::cout << "Time taken: " << elapsed.count()*1000 << " ms." << std::endl;

  free(a);
  free(b);
  free(c);

  return 0;
}
```

- 用python实现自动 **变换tile_size + 测试**

（1）该代码和测试输入 matmul.mlir 以及 test.cpp放在同一个文件夹

（2）`the_path_to_mlir` 替换为自己的的mlir-opt所在路径

（3）`llc` 也在 mlir-opt 的同路径下

```python
# auto-scheduler/examples/mlir-transform-cpu 

# Step1: replace the tile_size
# Step2: matmul.mlir —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll —> llvm.o --> a.out
# Step3: run ./a.out

import os
import subprocess

# go into where the matmul.mlir is
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# auto_test begins:
# Step1: replace the tile_size
def replace_tile_size(new_tile_size=[8, 8, 8]):
    with open('matmul.mlir') as f:
        lines = f.readlines()

    with open('matmul.mlir', 'w') as f:
        for line in lines:
            if line.startswith('            transform.structured.tile %0 '):
                before = line[line.index('['):line.index(']')+1]
                line = line.replace(before, str(new_tile_size))  
            f.write(line)
    # print(f"before: {before}, new_tile_size: {new_tile_size} ")

# the_path_to_mlir_opt = "../../build/bin/" # auto-scheduler/build/bin/
the_path_to_mlir_opt = "/lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/bin/"

def run_transform(now_tile_size=[8, 8, 8]):
    print("now the tile_size is: {}".format(now_tile_size))
    # Step2: matmul.mlir —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll —> llvm.o --> a.out
    command1 = [f"{the_path_to_mlir_opt}/mlir-opt", 
            "-llvm-request-c-wrappers",
            "--test-transform-dialect-interpreter", 
            "--test-transform-dialect-erase-schedule",
            "matmul.mlir", "-o", "transformed.mlir"]
    # matmul.mlir --> transformed.mlir(dropped sequence)
    subprocess.run(command1)

    command2 = [f"{the_path_to_mlir_opt}/mlir-opt", 
            "--pass-pipeline=builtin.module(sparse-compiler{ parallelization-strategy=dense-outer-loop vl=8 reassociate-fp-reductions=1 enable-index-optimizations=1 })",
            "transformed.mlir", "-o", "llvm.mlir"]
    # transformed.mlir --> llvm.mlir
    subprocess.run(command2)

    command3 = [f"{the_path_to_mlir_opt}/mlir-translate", 
                "llvm.mlir", 
                "--mlir-to-llvmir", 
                "-o", "llvm.ll"]
    # llvm.mlir --> llvm.ll
    subprocess.run(command3)

    command4 = [f"{the_path_to_mlir_opt}/llc", 
                "-filetype=obj", 
                "llvm.ll", 
                "-o", "llvm.o"]
    # llvm.ll --> llvm.o
    subprocess.run(command4)

    command5 = ["g++", 
                "llvm.o", 
                "test.cpp", 
                "-lmlir_runner_utils", "-lmlir_c_runner_utils"]
    # llvm.o + test.cpp --> a.out
    subprocess.run(command5)

    # Step3: run ./a.out
    command6 = ["./a.out"]
    # run a.out
    subprocess.run(command6)

if __name__ == "__main__":
    for i in range(2, 7):
        # Step1
        tile = 2**i
        tile_size = [tile] * 3
        replace_tile_size(tile_size)
        # Step2 & Step3
        run_transform(tile_size)
        print("\n")
```

测试结果

![Untitled](./img_matmul性能测试/Untitled.png)

## 3. 使用 google benchmark 测试

- 编译google_benchmark：

```bash
# env
module delete gcc
module load gcc/9.3.0
module delete llvm
module load llvm/9.0.1

# download and build
# iprc-project/benchmark
set http_proxy=http://127.0.0.1:17890 & set https_proxy=http://127.0.0.1:17890
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
make -j 4 --build "build" --config Release

# check
cmake -E chdir "build" ctest --build-config Release
```

- 在 `test.cpp` 中加入benchmark测试

Google benchmark用法：[https://blog.csdn.net/uestcyms/article/details/121705514](https://blog.csdn.net/uestcyms/article/details/121705514)

性能测试结果：

```bash
now the tile_size is: [8, 8, 8]
2023-06-09T11:33:03+08:00
Run on (64 X 3400 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x32)
  L1 Instruction 32 KiB (x32)
  L2 Unified 1280 KiB (x32)
  L3 Unified 49152 KiB (x1)
Load Average: 1.62, 1.93, 2.58
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
------------------------------------------------------
Benchmark            Time             CPU   Iterations
------------------------------------------------------
BM_mlir_cpu      0.622 ms        0.621 ms         1089

now the tile_size is: [16, 16, 16]
2023-06-09T11:33:05+08:00
Run on (64 X 3400 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x32
  L1 Instruction 32 KiB (x32)
  L2 Unified 1280 KiB (x32)
  L3 Unified 49152 KiB (x1)
Load Average: 1.62, 1.93, 2.58
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
------------------------------------------------------
Benchmark            Time             CPU   Iterations
------------------------------------------------------
BM_mlir_cpu      0.480 ms        0.479 ms         1348

now the tile_size is: [32, 32, 32]
2023-06-09T11:33:09+08:00
Run on (64 X 3400 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x32)
  L1 Instruction 32 KiB (x32)
  L2 Unified 1280 KiB (x32)
  L3 Unified 49152 KiB (x1)
Load Average: 1.65, 1.93, 2.58
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
------------------------------------------------------
Benchmark            Time             CPU   Iterations
------------------------------------------------------
BM_mlir_cpu      0.363 ms        0.362 ms         1949

now the tile_size is: [64, 64, 64]
2023-06-09T11:35:21+08:00
Run on (64 X 3400 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x32)
  L1 Instruction 32 KiB (x32)
  L2 Unified 1280 KiB (x32)
  L3 Unified 49152 KiB (x1)
Load Average: 2.72, 2.33, 2.65
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
------------------------------------------------------
Benchmark            Time             CPU   Iterations
------------------------------------------------------
BM_mlir_cpu      0.845 ms        0.843 ms          804
```

所使用的 代码如下：

`test_bm.cpp`

```cpp
#include <benchmark/benchmark.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#define s1 128
#define s2 128

using namespace std;

template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
void _mlir_ciface_matmul_tensors_1(MemRefDescriptor<float, 2>* arg3, MemRefDescriptor<float, 2>* arg0,MemRefDescriptor<float, 2>* arg1,MemRefDescriptor<float, 2>* arg2);
}

static void BM_mlir_cpu(benchmark::State& state, 
                    MemRefDescriptor<float, 2> arg3, 
                    MemRefDescriptor<float, 2> arg0, 
                    MemRefDescriptor<float, 2> arg1, 
                    MemRefDescriptor<float, 2> arg2) {
    for (auto _ : state)
        _mlir_ciface_matmul_tensors_1(&arg3, &arg0, &arg1, &arg2);
}

int main(){
  // tensor<128x128xf32>
  // %arg0, %arg1, %arg2, %arg3: memref<128x128xf32>
  float *a = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *b = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *c = (float*)malloc(sizeof(float) * (long long)s1 * s2);
  float *d = (float*)malloc(sizeof(float) * (long long)s1 * s2);

  for(int i=0;i<s1*s2;i++){
      a[i] = 1;
      b[i] = 1;
      c[i] = 0;
      d[i] = 0;
  }

  MemRefDescriptor<float, 2> arg0 = {
      a,    // allocated
      a,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg1 = {
      b,    // allocated
      b,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg2 = {
      c,    // allocated
      c,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  MemRefDescriptor<float, 2> arg3 = {
      d,    // allocated
      d,    // aligned
      0,    // offset
      {s1,s2}, // sizes[N]
      {s2,1},  // strides[N]
  };

  ::benchmark::RegisterBenchmark("BM_mlir_cpu", &BM_mlir_cpu, 
                                  arg3, arg0, arg1, arg2)
                                  ->Unit(benchmark::kMillisecond);
  
  ::benchmark::RunSpecifiedBenchmarks(); // run
  ::benchmark::Shutdown(); 

  free(a);
  free(b);
  free(c);

  return 0;
}
```

`test_e2e_cpu.py`

```python
# auto-scheduler/examples/mlir-transform-cpu 

# Step1: replace the tile_size
# Step2: matmul.mlir —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll —> llvm.o --> a.out
# Step3: run ./a.out

import os
import subprocess

# go into where the matmul.mlir is
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# auto_test begins:
# Step1: replace the tile_size
def replace_tile_size(new_tile_size=[8, 8, 8]):
    with open('matmul.mlir') as f:
        lines = f.readlines()

    with open('matmul.mlir', 'w') as f:
        for line in lines:
            if line.startswith('            transform.structured.tile %0 '):
                before = line[line.index('['):line.index(']')+1]
                line = line.replace(before, str(new_tile_size))  
            f.write(line)
    # print(f"before: {before}, new_tile_size: {new_tile_size} ")

# the_path_to_mlir_opt = "../../build/bin/" # auto-scheduler/build/bin/
the_path_to_mlir_opt = "/lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/bin/"

def run_transform(now_tile_size=[8, 8, 8]):
    print("now the tile_size is: {}".format(now_tile_size))
    # Step2: matmul.mlir —> transformed.mlir(dropped sequence) —> llvm.mlir —> llvm.ll —> llvm.o --> a.out
    command1 = [f"{the_path_to_mlir_opt}/mlir-opt", 
            "-llvm-request-c-wrappers",
            "--test-transform-dialect-interpreter", 
            "--test-transform-dialect-erase-schedule",
            "matmul.mlir", "-o", "transformed.mlir"]
    # matmul.mlir --> transformed.mlir(dropped sequence)
    subprocess.run(command1)

    command2 = [f"{the_path_to_mlir_opt}/mlir-opt", 
            "--pass-pipeline=builtin.module(sparse-compiler{ parallelization-strategy=dense-outer-loop vl=8 reassociate-fp-reductions=1 enable-index-optimizations=1 })",
            "transformed.mlir", "-o", "llvm.mlir"]
    # transformed.mlir --> llvm.mlir
    subprocess.run(command2)

    command3 = [f"{the_path_to_mlir_opt}/mlir-translate", 
                "llvm.mlir", 
                "--mlir-to-llvmir", 
                "-o", "llvm.ll"]
    # llvm.mlir --> llvm.ll
    subprocess.run(command3)

    command4 = [f"{the_path_to_mlir_opt}/llc", 
                "-filetype=obj", 
                "llvm.ll", 
                "-o", "llvm.o"]
    # llvm.ll --> llvm.o
    subprocess.run(command4)

    # command5 = ["g++", 
    #             "llvm.o", 
    #             "test.cpp", 
    #             "-lmlir_runner_utils", "-lmlir_c_runner_utils"]
    command5 = ["g++", 
                "llvm.o", 
                "test_bm.cpp", 
                "-lmlir_runner_utils", "-lmlir_c_runner_utils",
                "-I", "../../benchmark/include/", 
                 "-L", "../../benchmark/build/src/",
                "-lbenchmark", "-pthread"]
    # llvm.o + test.cpp --> a.out
    subprocess.run(command5)

    # Step3: run ./a.out
    command6 = ["./a.out"]
    # run a.out
    subprocess.run(command6)

if __name__ == "__main__":
    for i in range(3, 7):
        # Step1
        tile = 2**i
        tile_size = [tile] * 3
        replace_tile_size(tile_size)
        # Step2 & Step3
        run_transform(tile_size)
        print("\n")
```

## 4. 思考

（1）对于每个测试的 `.mlir` 代码都写一个 `test.cpp` 时间成本开销太大了，需要完成类似于 `iree-run-module` 的类似工作：自动识别 `.mlir` 代码中的测试函数，给定 shape，自动生成测试用的数据

（2）将多个动态链接库 链接成一个动态库 `libout.so` 

在变换步骤：llvm.ll(有 `_mlir_ciface_matmul_tensors_1`接口) + test.cpp —> llvm.o —> a.out 时，我们使用的运行命令如下

```bash
g++ llvm.o test_bm.cpp -lmlir_runner_utils -lmlir_c_runner_utils  -I ../../benchmark/include/ -L ../../benchmark/build/src/ -lbenchmark -pthread 
```

为了更好得迁移，期望将这些动态库合成一个 libout.so 

- `/lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_runner_utils.so`
- `/lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_c_runner_utils.so`
- `-I ../../../benchmark/include/`
- `-L ../../../benchmark/build/src/ -lbenchmark -pthread`

```bash
# produce libout.so
g++ -shared /lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_runner_utils.so /lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_c_runner_utils.so -I ../../../benchmark/include/ -L ../../../benchmark/build/src/ -lbenchmark -pthread -o libout.so 
# produce a.out
g++ llvm.o test_bm.cpp -L ./libout.so -o a.out
# run a.out
./a.out

# error
$ g++ llvm.o test_bm.cpp -L ./ -lout -o a.out
test_bm.cpp:1:10: fatal error: benchmark/benchmark.h: No such file or directory
    1 | #include <benchmark/benchmark.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

如果只链接`libmlir_c_runner_utils.so` 和 `libmlir_c_runner_utils.so` 并修改 test.cpp(去掉benchmark)

```bash
# produce libout.so
g++ -shared -o libout.so /lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_runner_utils.so /lustre/S/ruantingfeng/mlir-project/tools/llvm-project/build/lib/libmlir_c_runner_utils.so
# produce a.out
g++ llvm.o test_bm.cpp -L ./libout.so -o a.out
# run a.out
./a.out

# 输出
matmul.mlir cpu_pipeline_test success!
Time taken: 1.93078 ms.
```