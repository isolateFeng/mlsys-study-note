# Triton

why?

cuda太底层，pytorch太高层；效率和性能的tradeoff

可见：algorithm、distribute to block、grid size（distribute to warps/threads/cores都是不可见的）

## triton language

```python
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
```

1. `@triton.jit` 装饰器decorator，表示下面这段代码是一个triton kernel（后续会被划分成多个mlu.func）
2. `a_ptr, b_ptr, c_ptr` 指针，为其代表的tensor的第一个元素的地址。用来将数据load到memory
3. 输入中一般也有stride，对于n维的tensor a，a.stride()会输出一个n维数组。stride用来找个每个数字的指针

```python
a = torch.rand([3,6])
a.stride() # (6, 1)
# 这里的第一个维度的 stride 是 6, 因为从 a[m, k] 的地址 到 a[m+1, k] 的地址,
```

1. 超参数 `tl.constexptr` ，对于不同的硬件使用时，最佳性能的参数可能是不同的，后续由 Triton compiler 来进行搜索不同的值
2. 虚拟循环 `pid = tl.program_id(axis=0)` ，每个kernel可能被执行多次
    1. program_id是这个虚拟的 for "循环" 里面的 index (第几次循环)
    2. `axis` , 是说明 "循环"有几层
    3. 调用kernel时，需要说明该kernel执行循环有几层，每层有几次，这就是 `grid` 的概念

## 特点

SIMD编程范式

显式地load和store（对标gather和scatter）

grid：task_id（mlu），每个triton kernel跑在一个grid内

offset为传入的block_size，auto_tuning时，将block_size以tl.constexptr（超参数）传入

load(x_ptr, mask = offset < N) ，例如offset=1024，mask为一个1024维的数组，每个数为0/1，当某位为1时，则load该数据，当某位为0时，舍弃。

使用mask来规范访存行为，防止越界

离散优化：尽量保证数据加载连续性—>分析每一步操作并总结出stride和strideVal，最终用于静态信息将tl.load优化成tensor.extract_slice（会下降会copy），比d2d的离散访存速度快