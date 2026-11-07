---
title: 显存解剖：一块 GPU 内存在 Spark 里的一生
date: 2026-11-07 15:40:42
categories:
  - GPU
tags:
  - cuDF-Spark
  - RMM
  - GPU
  - Spark
  - 内存管理
---
GPU 没有虚拟内存，没有 swap，容量比主机内存小一到两个数量级，却要被一个 executor 里几十个 Spark task 同时瓜分。 cuDF-Spark 为此搭了四层机制：准入、池化、溢出、重试。这篇文章把它们拆开讲，落到 join / agg / sort / window / shuffle 五个算子上，并用一节细讲 shuffle 的内存路径与 Kudo 序列化。

<!-- more -->

![图 0](fig-00.png)

**图 0** — 四层机制按"从预防到补救"的顺序串成一条链路，任何一层解决了问题，下面的层就不会被触发。下文十节分别拆解这条链路上的各个环节。

## 01. 显存难在哪：五个和 CPU 内存不一样的地方

写 CPU 程序的人很少认真处理 `malloc` 返回 `NULL` 的分支 —— 在有 overcommit 和 swap 的 Linux 上，这一行几乎不会执行。GPU 上完全相反：OOM 不是异常路径，是**日常路径**。原因有五个。

### 1. 没有虚拟内存，也就没人替你兜底

主机内存不够时，内核会把冷页换到 swap，进程只是变慢。显存没有这一层 —— 驱动不会替你把 column 换到主机内存上。**所以 cuDF-Spark 必须自己实现一套"换页"**，这就是后面要讲的 spill 框架存在的全部理由。

### 2. 容量差一到两个数量级，但并发度不变

一台机器可以有 1 TB 主机内存配一块 80 GB 的 A100。Spark 并不会因为你换了 GPU 就少起几个 task —— 一个 executor 开 16 个 core 就有 16 个 task 线程，它们共享的是那 80 GB 里扣掉保留量之后的一部分。单 task 的"合理内存预算"因此被压缩得非常小。

### 3. 分配本身很贵，而且会隐式同步

`cudaMalloc` 要陷入驱动、修改设备页表，并且在很多情况下会同步整个设备 —— 把所有流上正在跑的 kernel 等完。一次查询里有几十万次分配，逐次走驱动是不可接受的。**池化不是优化，是必需品。**

### 4. 释放是异步的，引用计数到零不等于可以 free

CPU 线程调用 `free` 时，GPU 上可能还有 kernel 在读这块显存 —— kernel 是异步排在 stream 上的。这直接决定了 spill 必须拆成两阶段：先把数据拷走并标记，等一次 CUDA 同步之后，才真正归还显存。

> 这条约束渗透到整个框架：`SpillFramework` 的 `spill()` 和 `releaseSpilled()` 分开、`GpuPartitioning` 在 `contiguousSplit` 之后显式 `Cuda.DEFAULT_STREAM.sync()`，都是同一个原因。

### 5. 失败的代价不对称

CPU 侧内存压力的后果是"变慢"；GPU OOM 的后果是 task 直接抛异常，Spark 重试整个 task，重算的代价可能是几分钟。而且因为所有 task 共享一块显存，**一个 task 的峰值会撑爆所有人** —— 没有内核级的隔离。

这五点合起来解释了一个设计取向：cuDF-Spark 宁可牺牲一点吞吐（排队、切批、spill），也要保证查询能跑完。下面所有机制都服务于这个目标。

## 02. RMM 与 malloc：像在哪，不像在哪

RMM（RAPIDS Memory Manager）是 cuDF 的分配器。从接口形态看，它和 glibc 的 `malloc` 是同一类东西：都是用户态分配器，都在一块向系统预留的大区域上做子分配，都要和碎片作斗争。差别在于它能做什么、不能做什么。

![图 1](fig-01.png)

**图 1** — 两条路径的前半段几乎同构，真正的分叉在最后一跳：CPU 上"内存不够"被内核吸收掉了，GPU 上这件事被**退回到应用层**。cuDF-Spark 的全部复杂度都来自于承接这个退回。

### 相似的部分

| 能力 | malloc | RMM |
|---|---|---|
| 池化子分配 | arena + bins + tcache | `pool_memory_resource` / `arena_memory_resource` |
| 线程本地缓存 | tcache，减少锁争用 | arena 模式按线程分 arena，PTDS 下推荐 |
| 分配器可替换 | `LD_PRELOAD` 换 jemalloc / tcmalloc | `spark.rapids.memory.gpu.pool` 选 pool / arena / async / none |
| 可嵌套组合 | 较弱 | `memory_resource` 可层层包装，见下文的 spill 专用池 |

### 不同的部分

- **池大小是启动时的一次性决策。**`GpuDeviceManager.computeRmmPoolSize` 按 `(gpu.free − reserve) × allocFraction` 算出来，再被 min/max fraction 夹住，之后**不再增长**。`reserve` 是留给 cuDF/cuBLAS 这些库自己 `cudaMalloc` 的，以及 UCX shuffle 的 bounce buffer。
- **有"分配失败回调"这种东西。**`malloc` 失败只能返回 `NULL`；RMM 允许注册 `RmmEventHandler`，在失败时同步回调到 JVM，由你决定"腾完空间后要不要重试"。这是整个架构的支点。
- **释放要考虑流。**见上一节第 4 点。
- **可以开两个池。**spill 自己也要显存（`chunked_pack` 需要工作空间），如果和主池抢就会死锁式地失败。所以 `GpuDeviceManager` 额外开了一个 `chunkedPackMemoryResource` 小池专供 spill。

### 编程接口：三层，你真正调的是最上面那层

"RMM 的接口是什么"这个问题有三个答案，取决于你站在哪一层。

#### ① C++ · `cuda::mr::resource` 概念

早期 RMM 用的是一个叫 `device_memory_resource` 的抽象基类加虚函数分发。**现在这个基类已经没有了** —— RMM 改成了 CCCL 的编译期概念（concept）模型：任何类型只要提供下面这组成员函数，就自动满足 `cuda::mr::resource`，不需要继承任何东西。

```cpp
// 异步版：stream 是第一个参数，不是可选项
void* allocate(cuda::stream_ref stream, size_t bytes, size_t alignment);
void  deallocate(cuda::stream_ref stream, void* ptr, size_t bytes, size_t alignment);

// 同步版
void* allocate_sync(size_t bytes, size_t alignment);
void  deallocate_sync(void* ptr, size_t bytes, size_t alignment);

// 能力声明：靠 ADL 友元函数在编译期打标签
friend void get_property(cuda_memory_resource const&, cuda::mr::device_accessible) noexcept {}

// 编译期验证，不满足概念就编译不过
static_assert(cuda::mr::resource_with<pinned_host_memory_resource,
                                       cuda::mr::device_accessible>);
```

和 `malloc` 最本质的签名差异就在第一行：**`stream` 是必需参数**。分配和释放都发生在某条 CUDA 流的时间线上，而不是一个全局的、即时生效的动作 —— 这正是第 01 节说的"释放是异步的"在接口层面的体现。另一个差异是 `deallocate` 要求你**把 size 传回来**（`malloc`/`free` 不用），这让分配器可以省掉每块内存的头部元数据。

需要在运行时切换分配器时，用 `resource_ref` 做类型擦除，而不是虚函数：

```cpp
using device_async_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;
using host_async_resource_ref   = cuda::mr::resource_ref<cuda::mr::host_accessible>;
```

RAII 容器是 `rmm::device_buffer` 和 `rmm::device_uvector`，构造时就要同时交出 `stream` 和一个资源引用。cuDF 的每一列显存最终都来自这里。这一层 cuDF-Spark 不直接碰，但它决定了上面两层的形状 —— 比如 Java 侧的 `Rmm.alloc(size, stream)` 为什么必须带 stream。

> `get_property` 这个机制在 pinned 内存上给出了一个很干净的答案：`pinned_host_memory_resource`**同时声明了 `host_accessible` 和 `device_accessible` 两个属性**，而普通的 pageable 主机内存只有前者。"GPU 能不能直接寻址这块主机内存"这件事，在类型系统里就表达清楚了 —— 这也正是 pinned 内存能做真正异步 DMA 的根本原因。

#### ② Java · `ai.rapids.cudf.Rmm`

cuDF 的 Java 绑定把 C++ 的资源对象一一映射成 `RmmDeviceMemoryResource`（接口里只有 `getHandle()` 和 `close()` —— 它就是一个 C++ 指针的壳）。能直接调的 API 其实很少：

```scala
// 初始化：按 bitmask 模式一次性建好整条资源链
Rmm.initialize(allocationMode, logConf, poolSize)

// 显式分配，返回引用计数的 AutoCloseable
val buf: DeviceMemoryBuffer = Rmm.alloc(size, stream)

// 查询
Rmm.getTotalBytesAllocated        // 当前已分配
Rmm.getMaximumTotalBytesAllocated // 历史峰值
Rmm.getScopedMaximumBytesAllocated// 区间峰值，配合 reset 用
Rmm.getPoolSize / isPoolingEnabled

// 挂失败回调
Rmm.setEventHandler(handler)
```

真正有意思的是**资源是可堆叠的装饰器**。`RmmWrappingDeviceMemoryResource` 是装饰基类，每一层包住下一层，各管一件事：

| 类 | 职责 |
|---|---|
| `RmmCudaMemoryResource` | 终端：直接 `cudaMalloc` |
| `RmmManagedMemoryResource` | 终端：统一内存 |
| `RmmPoolMemoryResource` | 池化子分配 |
| `RmmArenaMemoryResource` | 按线程分 arena，PTDS 友好 |
| `RmmCudaAsyncMemoryResource` | `cudaMallocAsync` |
| `RmmTrackingResourceAdaptor` | 记账，`getTotalBytesAllocated` 的数据来源 |
| `RmmLoggingResourceAdaptor` | 把每次分配写日志 |
| `RmmLimitingResourceAdaptor` | 额外限额（async 模式下用它兜住 poolSize） |
| `RmmEventHandlerResourceAdaptor` | 失败时回调到 JVM |

#### ③ `RmmSpark` —— cuDF-Spark 实际编程的那一层

这是本节的重点。把整个仓库 grep 一遍会看到一个很说明问题的分布：`Rmm.*` 的调用只有十余处，而且几乎全在 `GpuDeviceManager` 的初始化里；`RmmSpark.*` 则散布在几十个地方。**算子代码从不调 RMM，它调的是 RmmSpark。**

而 `SparkResourceAdaptor`（RmmSpark 背后的那个类）本身就继承 `RmmWrappingDeviceMemoryResource` —— 它不是另起炉灶的系统，**就是资源链最上面的那一环**：

![图 2](fig-02.png)

**图 2** — 一条自下而上搭起来的装饰器链。`Rmm.initialize` 先建好下面四层，`RmmSpark.setEventHandler` 再在顶上套两层 —— 所以 Spark 的重试语义不是加在 RMM 旁边，而是**加在 RMM 的分配路径里**：每一次 `cudaMalloc` 之前都先经过它。

**为什么非要多这一层？**因为 RMM 原生的 event handler 只知道"这次分配失败了，要不要重试"，不知道**是谁在分配**。而所有有用的决策都需要这个信息：该让哪个线程回滚、哪个线程阻塞、哪个 task 的数据先被 spill、某个 task 到底用了多少显存。`SparkResourceAdaptor` 在分配路径里维护了一张"线程 → task → 状态"的表，把这些答案补齐。

于是 `RmmSpark` 的 API 可以分成五组：

| 分组 | 主要方法 | 用途 |
|---|---|---|
| 线程注册 | `currentThreadIsDedicatedToTask`<br>`poolThreadWorkingOnTask`<br>`shuffleThreadWorkingOnTasks`<br>`removeAllCurrentThreadAssociation` · `taskDone` | 告诉分配器"我这个线程在为哪个 task 干活"。区分**专属线程**（阻塞它不会连累别的 task）和**池线程**（阻塞它可能传递性地卡住其他 task），两者的阻塞策略不同 |
| 重试协议 | `blockThreadUntilReady`<br>`currentThreadStartRetryBlock` / `…End` | catch 到 RetryOOM 后，先把输入标成可 spill，再调 `blockThreadUntilReady` 让出，醒来后重试。`RmmRapidsRetryIterator` 封装的就是这套流程 |
| CPU 侧钩子 | `preCpuAlloc` · `postCpuAllocSuccess`<br>`postCpuAllocFailed` · `cpuDeallocate` | `HostAlloc` 把主机内存的分配也报给同一个状态机，这样 CPU 和 GPU 两边的 OOM 能共用一套重试语义 |
| 状态与指标 | `getStateOf` · `getMaxGpuTaskMemory`<br>`getAndResetNumRetryThrow`<br>`getAndResetComputeTimeLostToRetryNs` | per-task 的峰值显存、重试次数、因重试损失的计算时间 —— `GpuTaskMetrics` 的数据来源 |
| 协作标记 | `spillRangeStart` / `spillRangeDone`<br>`waitingOnPool` / `doneWaitingOnPool` | 告诉状态机"我正在 spill"或"我在等 bounce buffer"，避免把这些等待误判成死锁 |

### 线程状态机

这张表在排查 GPU OOM 时很有用 —— RmmSpark 的日志里打的就是这些状态之间的跃迁：

| 状态 | 含义 |
|---|---|
| `THREAD_RUNNING` | 正常跑 |
| `THREAD_ALLOC`<br>`THREAD_ALLOC_FREE` | 正在分配中；后者表示分配期间发生了一次 free（说明还有希望，可以重试） |
| `THREAD_BLOCKED` | 临时阻塞，等别人释放 |
| `THREAD_BUFN_THROW`<br>`THREAD_BUFN_WAIT`<br>`THREAD_BUFN` | **BUFN = Block Until Further Notice**。三步走：先抛异常让这个 task 回滚（`THROW`），回滚完等下一次分配时真正阻塞（`WAIT`），然后一直挂着（`BUFN`）直到**有别的 task 跑完**才放行 |
| `THREAD_SPLIT_THROW` | 该抛 `SplitAndRetryOOM` 了 —— 切小再来 |
| `THREAD_REMOVE_THROW` | 线程正在被摘除，需要抛异常退出 |

> `BUFN` 这组状态是整套机制里最"狠"的一招，设计意图写在 jni 的注释里：**当 spill 和暂停大分配线程都不足以避免 OOM 时，就不要再让这个线程重新开始了** —— 一直挂到确认有别的 task 真正完成、系统取得了进展为止。这和第 07 节"按 task 优先级挑 spill 受害者"是同一个思路：宁可部分串行，也要保证有进展。

### 所有权：ARM 模式

回到日常代码。Scala 侧很少出现 `Rmm.alloc`（全仓库只有 UCX shuffle 的接收路径直接用它）。显存的所有权跟着 cuDF 对象走（`Table` / `ColumnVector` / `DeviceMemoryBuffer`），而这些对象是 JVM 外的资源 —— **GC 不认识它们**。所以 cuDF-Spark 用 ARM（Automatic Resource Management）模式手工管生命周期：

```scala
// 异常安全：无论 process 抛不抛，col 都会被关
withResource(GpuColumnVector.from(batch)) { col =>
  process(col)
}

// 成功时所有权交给调用方，失败时就地关闭
closeOnExcept(new ColumnarBatch(...)) { b =>
  populate(b)
  b
}

// 集合语义
batches.safeClose()
cols.safeMap(_.copyToHostAsync(stream))

// 反例：process 抛异常就泄漏
val col = GpuColumnVector.from(batch)
val r = process(col)
col.close()
```

> 引用计数在这里有**双重职责**。它防泄漏，这是显然的；更重要的是，它是 spill 框架判断"这块数据能不能被挪走"的唯一依据 —— 引用计数高于预期，就说明有人正在用，这块数据不能动。一次泄漏不只是漏显存，还会让这块 buffer **永久不可 spill**。

### 碎片：GPU 上同样存在，而且更要命

答案是肯定的，而且比 CPU 上更难受。原因回到第 01 节那条：**CPU 有虚拟内存，GPU 的池没有。**

在 CPU 上，物理内存的碎片对应用基本透明 —— 页表可以把一堆不连续的物理页映射成一段连续的虚拟地址，`malloc` 拿不到大块时还能 `mmap` 一段新的虚拟地址。RMM 的池则是启动时一次性拿到的**一整段连续设备地址**，池内部的空洞无处可藏，而且池不能增长 —— **碎片直接等于可用容量的损失**。

#### 三个来源

![图 3](fig-03.png)

**图 3** — 碎片的三个来源。前两种 CPU 分配器也有；第三种是 GPU 特有的 —— 流的存在把一个空闲链表切成了 N 个。

#### 谁来管：四层，各管一段

| 层 | 机制 | 管什么 |
|---|---|---|
| free list | `coalescing_free_list` | 插入时按地址序找位置，**自动合并前后相邻的空闲块** |
| arena | superblock 的 `split` / `merge` | 大粒度的分裂与合并，整块在全局 arena 和各线程 arena 之间流转 |
| 跨流 | `get_block_from_other_stream` | 把别的流闲着的块借过来，必要时合并两条流的 free list |
| OOM 之前 | `defragment()` | 最后一搏：全设备同步，把所有 arena 的 superblock 全还给全局 arena |

#### 代码里的几个决定

**为什么是 first-fit 而不是 best-fit。**RMM 在注释里直接给了论文出处：

```cpp
// Address-ordered first-fit has shown to perform slightly better than best-fit
// when it comes to memory fragmentation, and slightly cheaper to implement.
// It is also used by some popular allocators such as jemalloc.
// \see Johnstone, M. S., & Wilson, P. R. (1998). The memory fragmentation problem: Solved?
block first_fit(std::size_t size) { ... }
```

直觉上 best-fit 更省空间，但它会在大块旁边留下一堆难用的小零头；地址序 first-fit 倾向于把分配集中在低地址，高地址更容易保持大块完整。

**合并是三分支的。**`superblock::coalesce` 插入一个空闲块时要同时看前后：

```text
bool const merge_prev = previous != cend() && previous->is_contiguous_before(blk);
bool const merge_next = next != cend() && blk.is_contiguous_before(*next);

if (merge_prev && merge_next) {          // 前后都连，三块并一块
  auto const merged = previous->merge(blk).merge(*next);
  free_blocks_.erase(previous);
  auto const iter = free_blocks_.erase(next);
  free_blocks_.insert(iter, merged);
} else if (merge_prev) { ... }
  else if (merge_next) { ... }
  else free_blocks_.insert(next, blk);   // 孤立的空洞，只能原样插入
```

**尺寸类是 jemalloc 风格的。**arena 模式下（编译期开 `RMM_ARENA_USE_SIZE_CLASSES`）请求会被对齐到 117 个预设尺寸之一 —— 256 B 起步、间距逐级翻倍。这是拿**块内碎片换块外碎片**：同尺寸类的块可以互相复用，代价是每次分配多浪费一点。

**跨流借用是两段式的。**这是 GPU 分配器最特别的一段逻辑：

```text
// 1. 本流的 free list —— 命中就直接用，不需要任何同步
// 2. 别的流的 free list —— 命中则插一个 cudaStreamWaitEvent，不阻塞 CPU
block_type block = get_block_from_other_stream(size, stream_event, blocks, false);
if (block.is_valid()) return block;

// 3. 还不行：同步合并别的流的整条 free list 进来，再找
block = get_block_from_other_stream(size, stream_event, blocks, true);
if (block.is_valid()) return block;

// 4. 都不行才扩池
return this->underlying().expand_pool(size, blocks, stream_event.stream);
```

第 2 步和第 3 步的差别是成本：前者只在 GPU 时间线上插一个等待，后者要真的把两条流的空闲链表合并（并记录新 event）。先试便宜的。

**`defragment()` 是 OOM 前的最后一搏。**arena 分配器的 `allocate` 是两段式的：

```cpp
{ std::shared_lock lock(mtx_);          // 快路径：共享锁，大家并发分配
  void* p = arena.allocate_sync(bytes);
  if (p != nullptr) return p; }

{ std::unique_lock lock(mtx_);          // 慢路径：独占锁
  defragment();                        // cudaDeviceSynchronize + 归还所有 superblock
  void* p = arena.allocate_sync(bytes);
  if (p == nullptr) {
    if (dump_log_on_failure_) dump_memory_log(bytes);
    RMM_FAIL("Maximum pool size exceeded ... No room in arena.", rmm::out_of_memory);
  }
  return p; }
```

`defragment()` 内部做的事很暴力 —— `cudaDeviceSynchronize()`，然后遍历所有线程 arena 和流 arena，把它们手里的 superblock 全部 `clean()` 还给全局 arena 重新合并。代价是一次全设备同步，所以只在即将 OOM 时才做。

> 注意这条链路和 cuDF-Spark 的 spill 是**两件不同的事，发生在不同层**。`defragment` 是在"总量够、但没有连续大块"时重新排列已有的空闲空间；spill 是在"总量就不够"时把别人的数据挪走。RMM 先试前者，失败后才通过 `onAllocFailure` 把问题交给后者。

#### 一条实用结论

三种池模式对碎片的处理方式完全不同：

| 模式 | 碎片由谁管 |
|---|---|
| `arena` | RMM 自己：superblock + 尺寸类 + `defragment`。多线程 / PTDS 场景下最稳 |
| `default`(pool) | RMM 自己：按流的 coalescing free list，跨流借用。结构更简单，但流多的时候 free list 分裂更明显 |
| `async` | **交给 CUDA 驱动的内存池**。RMM 只是包一层 `cudaMemPool_t`，整理工作在驱动里做 |

如果日志里反复出现"总 allocated 远小于池大小却依然 OOM"，基本可以判定是碎片而不是容量问题 —— 这时候换 `arena` 或 `async`、或者调小 `batchSizeBytes` 让请求粒度更均匀，通常比加大 `allocFraction` 有效。

## 03. 主机内存：独立记账，统一协议

cuDF-Spark 里的主机内存分三块，归属完全不同：

| 区域 | 谁管 | 装什么 |
|---|---|---|
| JVM heap | Spark / JVM GC | 执行计划、元数据、Spark 自己的 shuffle 结构 |
| Pinned 池 | `PinnedMemoryPool`，启动时一次分配 | D2H / H2D 拷贝的落点；spill 的中转 |
| Non-pinned | `HostAlloc` 记账，上限由 off-heap limit 决定 | pinned 不够时的退路、读文件的缓冲 |

后两块由 `HostAlloc` 统一出口。它被注册成 cuDF 的 `DefaultHostMemoryAllocator`，所以**连 cuDF 内部的主机分配也会经过它**。记账方式很直接：分配时自增计数器，并给 buffer 挂一个 `MemoryBuffer.EventHandler`，`onClosed` 时自减。

### pinned 还是 non-pinned

判断依据在第 02 节末尾那条：pinned 内存同时是 `host_accessible` 和 `device_accessible`，GPU 能直接寻址它，于是 DMA 引擎可以不经 CPU 参与地搬运 —— `cudaMemcpyAsync` 真异步。pageable 内存 GPU 寻址不到，驱动必须先拷进一个内部 staging buffer，**拷贝退化成同步**。

代价是页锁定内存不可换出，分配和释放都很贵。所以它只能以定长池的形式存在，总量要提前定死 —— 这就带来了取舍：

| 该用 pinned | 该用 non-pinned |
|---|---|
| spill 的 device → host 中转缓冲 | 纯 CPU 解析的元数据（Parquet footer、header） |
| shuffle 写侧的 D2H 落点 | 生命周期很长、会一直占着的缓冲 |
| shuffle 读侧攒批后上卡的 buffer | 单次超大分配 —— 定长池容易被一个请求打满 |
| 任何 `copyToHostAsync` 的另一端 | pinned 池已经紧张时的任何非传输用途 |

一句话概括：**会参与 PCIe 传输的用 pinned，只在 CPU 上处理的用 non-pinned。**

代码里的体现是 `HostAlloc.alloc(amount, preferPinned = true)` —— 默认 `true`，先试 pinned，失败回落 non-pinned；传 `false` 则顺序相反。两边互为 fallback，所以这不是硬性选择，只是一个偏好。全仓库**只有两处显式传 `false`**：读 Parquet footer 和 trailer 的那两个小 buffer，它们只被 CPU 解析，从不上卡。

> ⚠️ `spark.rapids.memory.pinnedPool.size`**默认是 0**，也就是默认根本不开 pinned 池。这种配置下上面整套偏好逻辑全部走 fallback 分支，所有主机内存都是 pageable，`copyToHostAsync` 实际上是同步的 —— `GpuPartitioning` 的注释里明确承认了这一点。如果 profile 显示 D2H 拷贝时间异常高，先检查这个配置。

### 联动：两条独立的记账线，一个统一的重试协议

显存和主机内存各有各的上限、各有各的计数器，但它们共用一套失败语义。这是整个设计里最省事的一个决定：

|  | 显存 | 主机内存 |
|---|---|---|
| 记账 | RMM 池 + `GpuTaskMetrics` | `HostAlloc` 计数器 + `GpuTaskMetrics` |
| 失败回调 | `onAllocFailure` | `spillAndCheckRetry` |
| 腾空间 | device store → host / disk | host store → disk |
| 腾不出来 | 抛 `GpuRetryOOM` / `GpuSplitAndRetryOOM` | 抛 `CpuRetryOOM` / `CpuSplitAndRetryOOM` |
| 谁接住 | `RmmRapidsRetryIterator` —— 同一个 |

也就是说，算子代码写一遍 `withRetry`，同时获得了显存和主机内存两边的保护。`HostAlloc` 还有一个小优化：`canNeverSucceed` 会先判断"这次请求本身就大于总上限吗"，如果是，直接失败，不做无谓的 spill。

### 数据什么时候在主机、什么时候在显存

这条线比看起来重要。读 Parquet、拉 shuffle 数据，产物最初都在主机内存上；cuDF-Spark 会尽量**让数据在主机侧多待一会儿**，攒够一批、拿到 GPU 许可之后再一次性上卡。写 shuffle 时反过来：数据切好、同步完，立刻释放 GPU 许可，后续的序列化和落盘都在主机侧完成。第 06 节和第 08 节会展开这个节奏。

## 04. Spill 框架：五个方法，三种策略

前面几节一直在说"spill 把数据挪走"，但挪的是什么、由谁决定、怎么保证挪的时候没人在用 —— 这些才是框架的本体。它的职责可以拆成三条：

1. **跟踪** —— 哪些对象还活着、各自多大、属于哪个 task；
2. **判定** —— 此刻哪些可以被安全地挪走；
3. **搬运与释放** —— 挪到哪一级、怎么挪、什么时候才算真的还给了 RMM。

### Handle：框架的基本单位

任何交给框架的对象都被包进一个 **handle**，**所有权随之转移** —— 原来的引用不能再用了。要用数据就调 `materialize`，拿回来的东西由调用方负责关闭。

handle 不是一个类，是一个家族。每种底层对象的打包方式和落点都不一样：

| Handle | 装什么 | spill 到哪、怎么搬 |
|---|---|---|
| `SpillableDeviceBufferHandle` | `DeviceMemoryBuffer` | 整块拷到一个 host handle |
| `SpillableColumnarBatchHandle` | 普通 `ColumnarBatch` | `chunked_pack` 分块打包 → host handle |
| `SpillableColumnarBatchFromBufferHandle` | 连续 buffer 支撑的 batch | 已经是连续的，整块拷 |
| `SpillableCompressedColumnarBatchHandle` | 压缩过的 shuffle batch | 整块拷，保持压缩态 |
| `SpillableTableHandle` | cuDF `Table` | `chunked_pack` |
| `SpillableHostBufferHandle` | `HostMemoryBuffer` | 流式写进一个磁盘文件 |
| `SpillableHostColumnarBatchHandle` | 主机列式 batch | 序列化落盘 |
| `SharedRecomputableHandle` | 共享的可重算设备对象<br>（join 哈希表） | **不拷不写，直接丢弃，用时重建** |
| `DiskHandle` | 磁盘文件里的一段 | 终点，不可再 spill |

### 五个核心方法

| 方法 | 谁调 | 职责 |
|---|---|---|
| `spillable: Boolean` | store 挑受害者时 | 此刻能不能挪。**调用时不持 store 锁** —— 它要拿 handle 自己的锁，持 store 锁去拿会 AB-BA 死锁 |
| `spill(): Long` | store，或直接对 handle 调 | 把数据搬到下一级，返回 `approxSizeInBytes`。**不释放原来那份**。线程安全：多线程同时调，只有一个真的干活，其余返回 0 |
| `releaseSpilled()` | 设备 store 的 `postSpill` | 真正 free 设备内存。只有设备侧 handle 有这个方法 |
| `materialize(...)` | 用户代码 | 把数据拿回来。还在原地就 `incRefCount` 零拷贝；已经 spill 了就新分配一份、从 host 或 disk 拷回 |
| `doClose()` | 框架内部 | 真正关闭底层 buffer。注意 `close()` **只打标记**，正在 spill 时把 `doClose` 推迟给 spill 线程 |

![图 4](fig-04.png)

**图 4** — 设备侧 handle 的生命周期。中间那个"已拷走但还没释放"的状态是整个框架最关键的设计 —— 它存在的唯一理由就是给 CUDA 同步留出位置（见 [issue #8610）。

### `spillable` 不是一条规则，是每种 handle 一条

判定是层层收紧的：基类只要求 `approxSizeInBytes > 0`，设备侧再加 `dev.isDefined`，到了具体 handle 才是真正的核心 —— **有没有别人正在用**。

| Handle | 额外条件 |
|---|---|
| `SpillableHandle`（基类） | `approxSizeInBytes > 0` —— 零字节的只被跟踪，不参与 spill |
| `DeviceSpillableHandle` | `dev.isDefined` —— 还没被挪走 |
| `SpillableDeviceBufferHandle` | `dmb.getRefCount == 1` —— 只有框架自己持有 |
| `SpillableColumnarBatchHandle` | 每一列的 refCount 等于它**在这个 batch 里出现的次数** |
| `SpillableHostBufferHandle` | `host.getRefCount == 1`，且还没落盘 |
| `SharedRecomputableHandle` | `pinCount == 0` —— 自己维护的租约计数 |

第四行值得单独说。一个 `ColumnarBatch` 可以**多次引用同一个底层列**（比如 project 出两个指向同一数据的列），所以不能简单判断 `refCount == 1`：

```scala
val dcvs = GpuColumnVector.extractBases(dev.get)
val colRepetition = mutable.HashMap[ColumnVector, Int]()
dcvs.foreach { hcv =>
  colRepetition.put(hcv, colRepetition.getOrElse(hcv, 0) + 1)
}
// 每列的实际引用数，必须正好等于它在本 batch 里出现的次数
dcvs.forall(dcv => colRepetition(dcv) == dcv.getRefCount)
```

最后一行的 `SharedRecomputableHandle` 则是因为 cuDF 的哈希表**根本没有引用计数**，框架只好自己维护一个 pin 计数，由 `acquire` / 租约的 `close` 配对增减。

### 三种 spill 策略

![图 5](fig-05.png)

**图 5** — 三种策略，差别在于"恢复时要付多少钱"。框架用这个代价排优先级：重算最便宜，所以最先牺牲。

第三种策略最值得展开。join 的哈希表是设备上的一大块结构，**拷到主机再拷回来，比直接重建还慢**。所以 `SharedRecomputableHandle` 的 `spill()` 什么 IO 都不做：

```scala
override def spill(): Long = {
  val thisThreadSpills = synchronized {
    if (!closed && dev.isDefined && pinCount == 0 && !spilling) {
      spilling = true
      evicted = dev
      dev = None                              // 逻辑上立刻失效
      SpillFramework.removeFromDeviceStore(this)
      true
    } else false
  }
  if (thisThreadSpills) {
    executeSpill { pendingRelease ++= evicted; 0L }   // 真正的 close 推迟到同步之后
    approxSizeInBytes                                 // 但账要记上
  } else 0L
}
```

之后第一个调 `acquire()` 的线程负责 `rebuild()`，其余线程在 handle 上 `wait()` 等它建完。`HashBuild` 还给这件事挂了个 `rebuilds` 指标，可以直接看到重建发生了多少次。

> 一个容易踩的死锁：`acquire()` 的第一行是 `GpuSemaphore.acquireIfNecessary`，而且**必须在拿 handle 锁之前**。因为等待重建的线程是攥着 GPU 许可在等的，如果重建者自己还没拿到许可，就会出现"持锁者等许可、持许可者等锁"。

### 级联：一次 spill 可能走两级

device 侧的 spill 目的地不一定是主机。真正的决策点在 `SpillableHostStore.makeBuilder` —— 它先看这块数据是否超过 host store 的上限，再试 `HostAlloc.tryAlloc`，两关都过才给一个"写主机"的 builder；否则直接给一个"写磁盘"的 builder，**整个跳过主机这一级**。

### 三级级联

![图 6](fig-06.png)

**图 6** — 三级 spill。注意两条细节：`chunked_pack` 让搬运一张 10 GB 的表不需要额外 10 GB 的连续空间，只需要几个固定大小的 bounce buffer；两阶段释放确保没有 kernel 还在读这块数据时才真正 free。

### 两个容易被忽略的实现细节

#### spill 和 close 的竞争

一个 handle 正在被 spill 时，持有它的 task 完全可能同时决定关掉它。框架用一个 `spilling` 标志和一次自保的 `incRefCount` 处理这个竞争：

- **spill 开始时**给底层 buffer 自增一次引用 —— 这样即使 `close()` 在搬运途中发生，buffer 也不会被释放掉；
- **`close()` 看到 `spilling == true`** 就只打一个 `closed` 标记，把真正的 `doClose()` 留给 spill 线程；
- **搬完之后**再检查一次 `closed`：若已被关，就把刚建好的 host handle 直接丢掉，否则才挂上去。

整段逻辑的效果是：**搬运途中的 handle 既不会被提前释放，也不会在搬完之后变成一个没人回收的孤儿。**

#### host store 的两个入口

`track` 和 `trackNoSpill` 行为完全不同，这是读这段代码时最容易困惑的地方：

| 入口 | 谁用 | 行为 |
|---|---|---|
| `track` | device → host 的 spill 路径 | 装不下就先 spill 到磁盘腾地方，最多重试 5 轮；腾不出来就失败，让调用方改走磁盘 |
| `trackNoSpill` | 用户代码直接创建的 host handle | 直接记账，**不触发任何级联** |

后者意味着一件反直觉的事：如果 host store 限额是 1 GB，而你用工厂方法直接塞进一个 1.5 GB 的主机 buffer，store 里就真的有 2.5 GB —— **限额只约束 spill 进来的数据，不约束直接放进来的**。这段行为是为了兼容框架重构之前的语义，文件头的注释里写明了。

### 优先级：代码里写的和实际生效的不是一回事

算子代码里到处是这样的写法：

```text
SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
```

看上去像是在指定优先级。但跟进 `SpillableColumnarBatch.apply` 会发现，**这个 `priority` 参数从头到尾没有被用到** —— 它没有传给任何 handle。handle 的 `taskPriority` 来自完全另一个地方：

```text
var taskPriority: Long =
  taskId.map(TaskPriority.getTaskPriority).getOrElse(Long.MaxValue)
```

全仓库真正写入 `taskPriority` 的只有三处：`SharedRecomputableHandle` 把自己钉死在 `Long.MinValue`；`SpillablePartialFileHandle` 显式传值；以及 spill 过程中产生的下级 handle（host、disk）**继承上级的 taskPriority**，这样一份数据无论落在哪一级，在队列里的位置都不变。

> 所以实际的 spill 顺序是：**先牺牲可重算的对象，然后严格按 task 年龄从新到老。**`SpillPriorities` 里那几个常量（`ACTIVE_ON_DECK`、`INPUT_FROM_SHUFFLE` 等）对设备 store 的选择没有影响 —— 读代码时别被它们误导。

## 05. 五个算子的退路

算子层的通用模式只有两件事：**把不在用的数据包成 `SpillableColumnarBatch`**（交给 spill 框架托管），**把 GPU 计算包进 `withRetry`**（接住 OOM）。

```scala
// 不切分：失败就原样重跑，靠 spill 腾空间
withRetryNoSplit(spillableBatch) { attempt =>
  withResource(attempt.getColumnarBatch()) { batch => doGpuWork(batch) }
}

// 可切分：接到 GpuSplitAndRetryOOM 就把输入切半再跑
withRetry(spillableBatch, splitSpillableInHalfByRows) { attempt => ... }

// 有状态：checkpoint / restore，回滚到重试前的位置
withRestoreOnRetry(retryable) { ... }
```

两条硬性前提，写算子时必须满足：

- **`fn` 必须幂等** —— 它可能被调用多次。带状态的表达式要实现 `Retryable` 的 `checkpoint` / `restore`。
- **进入 `withRetry` 之前，调用方持有的数据必须已经是 spillable 的。** 这样这个线程才可以被安全地阻塞、它的数据才可以被别的 task 挪走。

> 真正有意思的差别在于：**每个算子"能切的维度"不一样。** 有的能切输入，有的只能切输出，有的两头都不能切、只能重新组织数据布局。下面逐个看。

### Hash Aggregate 切输入 → 不行就重分区

**数据驻留在哪。**三层，全部托管给 spill 框架：输入批是 `SpillableColumnarBatch`；第一轮部分聚合的结果由 `CloseableBufferedIterator` 串起来，也全是 spillable；重分区之后是 `ArrayBuffer[AutoClosableArrayBuffer[SpillableColumnarBatch]]` —— **整个桶数组全程可被挤走**。

**第一道退路：切输入。**前处理、聚合、后处理三段都包在 `withRetry(sb, splitSpillableInHalfByRows)` 里。按行切在这里是安全的，因为后面还有 merge 阶段把部分结果合起来。唯一不能切的是 `concatenateBatches` —— 它的语义就是"把这些合成一个"，所以用 `withRetryNoSplit`。

**第二道退路：hash 重分区。**当多批部分聚合结果合不进一个 batch 时：

```scala
def repartitionAndClose(batch: SpillableColumnarBatch): Unit = {
  // 只有一批、且所有桶都空 → 直接丢进 0 号桶，跳过重分区
  if (!aggregatedBatches.hasNext && batchesByBucket.forall(_.size() == 0)) {
    batchesByBucket.head.append(batch); return
  }
  withResource(new GpuBatchSubPartitioner(
      Iterator(withRetryNoSplit(batch)(_.getColumnarBatch())),
      hashKeys, hashBucketNum, hashSeed, "aggRepartition")) { partitioner =>
    (0 until partitioner.partitionsCount).foreach { id =>
      partitioner.releaseBatchesByPartition(id).foreach { nb =>
        if (nb.numRows() > 0) batchesByBucket(id).append(nb)  // 立刻变 spillable
        else nb.safeClose()
      }
    }
  }
}
```

桶还是太大就**递归再分**，并且每层换一个 hash 种子（`baseHashSeed + recursiveDepth * 7`）—— 同一个种子再切一次只会得到同样的倾斜。递归有上限 `maxRecursiveDepth`，超了只打 warning 不报错，因为理论上部分聚合之后不该剩那么多重复 key（[issue #11834）。

> 内存特征：名义峰值是"所有桶之和"，但桶里每一批都是 spillable 的，真正必须驻留显存的只有**当前正在合并的那一个桶**。这就是为什么 agg 在数据远大于显存时仍然能跑完。

### Hash Join 子分区 + 切输出

Join 是五个算子里内存结构最复杂的一个 —— **三块内存，三种完全不同的管法**。

#### ① build 侧数据 → `LazySpillableColumnarBatch`

注意它不是一进来就 spillable 的。内部有两个字段：`cached`（裸 batch）和 `spill`（`SpillableColumnarBatch`），只有调用 `allowSpilling()` 时才把前者转成后者：

```scala
override def allowSpilling(): Unit = {
  if (spill.isEmpty && cached.isDefined) {
    try {
      spill = Some(SpillableColumnarBatch(cached.get, ACTIVE_ON_DECK_PRIORITY))
    } finally { cached = None }   // 所有权转移，裸引用必须清掉
  }
  cached.foreach(_.close()); cached = None
}
```

为什么要 lazy：build batch 在整个 probe 过程中被反复读取，如果一直挂在 store 里，每轮 spill 挑受害者时都要白查一次它的 refCount。显式地在"这轮确定不用了"的时刻交出去，更省事也更准确。`getBatch` 反过来会在需要时从 `spill` 里再取回来。

#### ② 哈希表 → `SharedRecomputableHandle`，丢弃重算

就是上一节讲的第三种策略。`HashBuildCache` 让同一个 build 侧在多次 probe 之间共享一张表，`ReacquirableLease` 则允许持有者在等待重试时**主动交出租约**，让表可以被挤走，醒来后再重新 `acquire`（必要时触发重建）。

#### ③ 输出 → `JoinGatherer`，切的是产出粒度

Join 的输出可能比输入大几个数量级，所以结果不是一次性物化的。`JoinGatherer` 持有 cuDF 的 gather map，按目标字节数一批批 gather。这里的 retry 切的**不是输入，是"这次取多少行"**：

```scala
val minTargetSize = Math.min(targetSize, 64L * 1024 * 1024)
val targetSizeWrapper = AutoCloseableTargetSize(targetSize, minTargetSize, ...)
withRetry(targetSizeWrapper, splitTargetSizeInHalfGpu) { attempt =>
  val nextRows = JoinGatherer.getRowsInNextBatch(gather, attempt.targetSize, ...)
  ...
}
```

之所以能这么玩，是因为 **gatherer 自己记录了取到哪一行**。配合 `withRestoreOnRetry`，重试时能精确回到上次的位置 —— 目标尺寸减半意味着这次从 gatherer 少取一点，但不会重复也不会漏。不活跃时还会调 `gathererStore.foreach(_.allowSpilling())` 把自己也交出去。

#### build 侧装不下时：两侧一起子分区

```scala
protected def partitionBatch(inputBatch: ColumnarBatch): Unit = {
  val spillableBatch = SpillableColumnarBatch(inputBatch, ACTIVE_ON_DECK_PRIORITY)
  withRetryNoSplit(spillableBatch) { _ =>
    val partsTable = hashPartitionAndClose(spillableBatch.getColumnarBatch(), numPartitions, ...)
    val contigTables = partsTable.getTable.contiguousSplit(partsTable.getPartitions.tail: _*)
    withResource(contigTables) { _ =>
      contigTables.zipWithIndex.foreach { case (ct, i) =>
        if (ct.getRowCount > 0) { contigTables(i) = null; addPartition(i, ct) }
      }
    }
  }
}
```

每个 `JoinPartition` 持有一组 `SpillableColumnarBatch` 和一个 `totalSize`。相同 key 必定落进同一个分区，所以 N 个小 join 串行跑的结果和一个大 join 等价。

还有一层容易被忽略：probe 阶段为了量两侧大小，已经从 shuffle 拉回来的主机数据也是托管的 —— `SpillableHostConcatResult` / `KudoSpillableHostConcatResult`，之后用 `HostQueueBatchIterator` 原样回放。**探测不会浪费已经读进来的数据，也不会让它白占主机内存。**

### Sort 外部归并排序

全局排序没法靠切批解决，所以装不下时直接换算法。两个队列，**里面全是 `SpillableColumnarBatch`**：

| 队列 | 结构 | 装什么 |
|---|---|---|
| `pending` | `PriorityQueue[OutOfCoreBatch]`，按首行排序 | 还没归并完的数据，附带 `pendingSize` 计数 |
| `sorted` | `LinkedList[SpillableColumnarBatch]` | 已确定全局有序、等待输出，附带 `sortedSize` |

关键设计在 `OutOfCoreBatch` 这个结构上：

```scala
case class OutOfCoreBatch(buffer: SpillableColumnarBatch, firstRow: UnsafeRow)
```

> **排序的元数据和数据是分离的。**`firstRow` 是一个主机侧的 `UnsafeRow`，始终驻留内存；`buffer` 才是可能被挤到磁盘上的那份。于是即使所有批都被 spill 了，优先队列**仍然能正常比较和出队**，完全不需要把数据读回来。这是整个 out-of-core sort 能成立的前提。

唯一必须回到 CPU 的地方是 `convertBoundaries`：把每批的首行 `copyToHost` 再转成 `UnsafeRow`。每批只做一行，代价可以接受。

每轮取若干批做归并，用"下一个 pending 批的首行"当切分点 —— 之前的部分已全局有序，进 `sorted`；之后的部分由 `splitAfterSort` 按 `targetSize / 8` 切小再塞回 `pending`。那个 `/ 8` 是个经验值：实际场景里通常只有一两批数据，为这个常见情况优化比精确计算更划算。

### Window 不能切 · 只能等

窗口函数要求"同一个 partition-by 分组的所有行在同一批，且按 order-by 排好"。**切批就会算错**，所以这里只能 `withRetryNoSplit` —— 失败了原样重跑，指望 spill 框架从别的 task 那里腾出空间。

能做的优化只有两件。一是把预读但还没处理的那批也托管出去：

```scala
var onDeck: Option[SpillableColumnarBatch] = None

override def next(): ColumnarBatch = {
  val cbSpillable = onDeck match {
    case Some(x) => onDeck = None; x
    case _ => SpillableColumnarBatch(input.next(), ACTIVE_BATCHING_PRIORITY)
  }
  withRetryNoSplit(cbSpillable) { _ => ... }
}
```

二是**按窗口形态分化出不同的迭代器**，让尽可能多的 case 避开"整组必须驻留"的最坏情况：

| 迭代器 | 适用形态 | 内存代价 |
|---|---|---|
| `GpuWindowIterator` | 通用 | 整批驻留，最坏情况 |
| `GpuRunningWindowIterator` | running window<br>（unbounded preceding → current） | 可增量，只需保留上一批的收尾状态（`cachedBatch`） |
| `GpuBatchedBoundedWindowIterator` | 有界偏移 | 只需前后各 N 行的重叠部分 |
| `GpuCachedDoublePassWindowIterator` | unbounded → unbounded | 两遍扫描 + `FixerPair` 事后修正，避免一次性持有整组 |
| `GpuUnboundedToUnboundedAggWindowIterator` | 整组一个值 | 退化成聚合，代价最低 |

换句话说，window 没有"退路"，只有"别走到最坏情况"。规划阶段选对迭代器，比运行期的任何补救都重要。

### Shuffle 尽早离开 GPU

Shuffle 不是"怎么在显存里撑住"，而是"怎么尽快不占显存"：写侧数据一落主机就 `releaseIfNecessary`，读侧在主机内存里攒批、最后一刻才 `acquireIfNecessary`。

两侧的数据同样全程托管：写侧用 `SpillablePartialFileHandle`（可以配置成直接写文件，或者先放主机 buffer 再按需落盘），读侧用 `SpillableKudoTable` 把每个收到的 block 包进 `SpillableHostBuffer`。

写侧三条路径、读侧三种拼装方式、以及 Kudo 序列化为什么能消掉最大的一次显存峰值 —— 第 06 节单独展开。

### 一张表总结

| 算子 | 硬约束 | OOM 时切什么 | 兜底策略 |
|---|---|---|---|
| Aggregate | 同 key 同批 | 输入行数 | hash 重分区成桶，可递归 |
| Join | build 侧完整 | 输出目标字节数 | 两侧按 key hash 子分区，拆成 N 个小 join |
| Sort | 全局有序 | 归并后的切分点 | 外部归并排序，两个队列全 spillable |
| Window | 同组同批 + 有序 | 切不了 | 原样重试，等别人 spill；按窗口形态选专用迭代器 |
| Shuffle | 跨 device/host 边界 | 攒批的表数量 | 尽早释放许可、数据滞留主机侧；详见 第 06 节 |

## 06. Shuffle 与 Kudo 序列化

Shuffle 是整条流水线上唯一**必须两次跨越 device / host 边界**的环节：数据在 GPU 上分区，序列化后落到主机内存或磁盘，对端再反序列化回 GPU。两次跨越之间的那段时间，数据既不该占显存，也不该占着 GPU 许可。

所以这一节的主线不是"怎么撑住"，而是三个问题：**切分在哪做**、**序列化成什么格式**、**许可什么时候还、什么时候拿**。

### 写侧：三条路径

入口是 `GpuPartitioning.sliceInternalGpuOrCpuAndClose`。算完 hash 分区索引之后，按配置分派到三条完全不同的路：

![图 7](fig-07.png)

**图 7** — 三条写侧路径的差别不在"要不要还许可"，而在"切分在哪做"。切得越靠前（GPU 侧），D2H 的数据量越小；切得越靠后（主机侧），显存峰值越低。

| 路径 | 触发条件 | 显存峰值 | 代价 |
|---|---|---|---|
| Kudo GPU 切分 | `kudo.serializer.write.mode=GPU` | 中 | 序列化结果受 2 GB 限制（offsets 是 Int），超了要调小 `batchSizeBytes` |
| contiguousSplit | UCX 模式（`shuffle.mode=UCX`） | 高（约 2×） | 切分是真实拷贝；换来的是数据不离开显存，可直接 RDMA |
| CPU 切分 | 默认（MULTITHREADED 模式） | 低 | 整批 D2H，主机内存和 PCIe 带宽压力大 |

第二条路里那次 `Cuda.DEFAULT_STREAM.sync()` 值得单独说一句：下游的 shuffle manager 会把上百个 buffer 逐个注册进 spill 框架，而每个 spillable handle 都要求"内容已经在设备上完全物化"。在这里统一同步一次，比每个 buffer 各自同步便宜得多。

### Kudo 解决的是什么问题

在 Kudo 之前，shuffle 用的是 cuDF 自带的 `JCudfSerialization`。它的输入必须是一张**已经物化好的表** —— 序列化就是把 header 和各个 buffer 顺序写出去。这个前提直接决定了写侧必须先 `contiguousSplit`：你得先把 N 个分区真的切成 N 张独立的表，才能分别序列化。

而 `contiguousSplit` 是一次完整的 GPU 拷贝。原表还在、新表已经生成，**瞬时需要大约两倍的显存**。

Kudo 改掉的就是这个前提。它的 `WriteInput` 带 `rowOffset` 和 `numRows`：

```scala
val writeInput = WriteInput.builder
  .setColumns(columns)        // 完整的列，没有被切过
  .setOutputStream(out)
  .setNumRows(numRows)
  .setRowOffset(startRow)    // 只写出 [startRow, startRow + numRows) 这个行区间
  .setMeasureCopyBufferTime(measureBufferCopyTime)
  .build
kudo.writeToStreamWithMetrics(writeInput)
```

配合 `SlicedGpuColumnVector`（只记录 `start` / `end` 两个下标，不持有任何独立数据），写侧**根本不需要把每个分区物化出来**。序列化器看到的是"完整的列 + 一个行区间"，直接按区间写字节流。

![图 8](fig-08.png)

**图 8** — 两种序列化的输入契约不同：JCudf 要"一张表"，Kudo 要"一组列 + 一个行区间"。这一个契约差异，消掉了写侧最大的一次显存峰值。

写侧还有一个细节：当输入的列已经是 `SlicedGpuColumnVector` 时，序列化器直接取 `getStart` 和 `getBase`，连 D2H 都不用做（数据早就在主机上了）；否则才对每列 `copyToHostAsync`，然后统一 `Cuda.DEFAULT_STREAM.sync()` 一次 —— 又是"多个异步拷贝，只同步一次"的套路。

### 读侧：三种拼装方式

读侧的任务是把几十上百个小的 shuffle block 拼成一个目标大小的 batch。拼装在哪做，决定了内存落在哪。

![图 9](fig-09.png)

**图 9** — 三条读路径。Kudo GPU 模式的关键优化藏在注释里：每个 spillable table 直接拷进**同一个**暂存主机缓冲，而不是各自物化成独立的主机 buffer —— 一批一次分配，因为 pinned 内存的分配会在同一把锁上串行。

### 读侧的三重内存保护

- **攒批期间可 spill。**每个收到的 kudo table 的主机 buffer 都包成 `SpillableHostBuffer`（`SpillableKudoTable`，优先级 `ACTIVE_BATCHING_PRIORITY`）。等待攒够目标批大小的这段时间里，这些数据随时可以被挤到磁盘上。
- **拼装失败可切半。**`concatenateTables` 外面包着 `withRetry`，切分策略是 `createSplitPolicyByTargetSize` —— **按字节**把这一组表切成两半（不是按表数量），下限 10 MB。切到只剩一张表还不够，抛 `GpuSplitAndRetryOOM`。切出来的第二半存进 `pendingResultIter`，下次 `next()` 直接返回，不重新攒批。
- **许可拿得尽量晚。**Kudo GPU 模式在 `concatenateTablesInGpu` 之前 acquire（因为拼装本身在 GPU 上）；主机拼装的两条路径在 `toGpuBatch` 之前才 acquire。注意即使是空 batch 也会 acquire —— 下游算子假定这个迭代器一定持有许可。

### Shuffle manager 的三种模式

| 模式 | 数据落点 | 对显存的影响 |
|---|---|---|
| MULTITHREADED<br>默认 | 主机内存 / 磁盘 | 写读都用线程池并行，数据尽快离开显存。配合 `skipMerge` 可以跳过分片合并，但要求关掉 ESS 并开启 off-heap 限额，否则 buffer 会无界增长 |
| UCX | 显存 | 走 RDMA 直传，数据不下卡 —— 延迟最低，但显存占用最高，而且要从 RMM 池里预留 bounce buffer |
| CACHE_ONLY | 内存 | 单 executor 短路，测试用 |

压缩（`shuffle.compression.codec`，支持 zstd / lz4）只在 `contiguousSplit` 那条路上生效，默认关闭。它减少的是传输量和主机内存占用，但压缩本身要在 GPU 上跑，会抬高瞬时显存。

> 一句话总结这一节：**Kudo 把"切分"从一次物理拷贝变成了一组偏移量**。写侧因此省掉了 shuffle 路径上最大的一次显存峰值，读侧则把拼装的代价挪到了 GPU 上 —— 默认配置 `write.mode=CPU, read.mode=GPU` 正是这两件事的组合。

## 07. Task 之间：共享什么，独享什么

一个 executor 进程里，所有 task 线程**共享同一个 RMM 池、同一套主机内存上限、同一组 spill store**。没有 cgroup 式的隔离，也没有 per-task 的显存配额 —— 唯一的"配额"是 GpuSemaphore 的准入许可，而那是软约束（见下节）。

|  | 内容 |
|---|---|
| 共享 | RMM 显存池、pinned / non-pinned 上限、三个 spill store、GpuSemaphore 的许可总量 |
| 独享 | 每个 `SpillableColumnarBatch` 的所有权、`GpuTaskMetrics` 的计数（峰值显存、spill 字节数、spill 耗时、信号量等待时间） |

### 内存不够时的升级链

从最轻到最重，四步：

1. **Spill 别人的数据。** `onAllocFailure` 第一反应是从 device store 挑受害者。挑谁有讲究，见下。
2. **同步后重试。** 一个字节都 spill 不出来时，`Cuda.deviceSynchronize()` 再试 —— 可能有流还没跑完、buffer 还没来得及还。最多重试 `gpuOomMaxRetries` 次，计数按线程维护。
3. **让这个 task 回滚。** 抛 `GpuRetryOOM`，`withRetry` 接住，线程阻塞，它持有的 spillable 数据这时可以被别人挪走。
4. **让这个 task 变小。** 抛 `GpuSplitAndRetryOOM`，算子按自己的切分策略把规模减半重来。切到不能再切 → 真 OOM，task 失败。

### 挑谁 spill：按 task 优先级，不是按 LRU

这是个容易被忽略但很关键的设计。`SpillableDeviceStore` 里的 handle 不是按最近使用时间排的，而是按 `taskPriority`（`HandleComparator`）—— 每个 handle 在创建时记下所属 task 的优先级，spill 时从优先级最低的开始挑。

> 为什么不用 LRU：显存紧张时，如果所有 task 都被公平地削一刀，结果是**谁也做不完** —— 每个 task 都在反复 spill、重算、再 spill。按优先级挑受害者，等价于"让老 task 先跑完腾出空间"，把并行退化成部分串行，但保证有进展。这是避免活锁的标准做法。

同一个优先级体系还用在别处：`GpuSemaphore` 底层是 `PrioritySemaphore`，排队时也按 task 优先级；异步 IO 线程池（`AsyncRunners`）、CPU bridge 线程池同样继承这个优先级。整个 executor 对"谁更重要"有一致的判断。

### 一个 task 的多个线程

一个 Spark task 可能有多个线程碰 GPU（比如多线程读文件）。`SemaphoreTaskInfo` 把它们组织起来：**整个 task 只持有一份许可**，内部用 `blockedThreads` / `activeThreads` 两个集合管理。只有队首的那个线程真正阻塞在信号量上，其余的在 `SemaphoreTaskInfo` 上 `wait()`，拿到许可后一起放行。这避免了同一个 task 的线程之间互相抢自己的配额。

## 08. GpuSemaphore：把 OOM 挡在发生之前

前面所有机制都是**事后的**：分配失败了才开始补救。GpuSemaphore 是唯一事前的那个 —— 它的作用是让"同时在 GPU 上干活的 task 数"保持在显存撑得住的范围内。

与其让 16 个 task 同时上卡然后一起 OOM、一起 spill、一起重试，不如先让 4 个上去跑完。**Spill 和 retry 都不是免费的**，它们只是比失败好；最优解是压根不触发它们。

### 它不是一个计数信号量

直觉上这应该是 `new Semaphore(concurrentGpuTasks)`。实际实现是**按内存配额发许可**：

![图 10](fig-10.png)

**图 10** — 许可是内存份额而不是名额，所以"并发度"是动态的：轻量 task 可以挤进来更多个，重 task 自然互斥。下半部分的时间线是算子代码里 `acquire` / `release` 放置位置的依据 —— 上一节 shuffle 的三条写路径和三条读路径，放置的都是这两个点。

- **许可总量**由 RMM 池大小换算而来（`computeMaxPermits`）。
- **每个 task 申请的份额**是 `memToPermitsWithMax(estimator.estimate())` —— 估计这个 task 要用多少显存，就申请多少。
- **估计值会自我修正**。打开 `concurrentGpuTasks.dynamic` 后，`GpuTaskMemoryEstimator` 用一个时间窗把"配置的默认估计"和"同 stage 其他 task 的实测峰值"加权混合：task 刚起步时信默认值，跑久了信实测值。
- **排队是有优先级的**。底层是 `PrioritySemaphore`，按 task 优先级而不是 FIFO，和 spill 的受害者选择用同一个优先级体系。

### 什么时候持有，什么时候放掉

规则很简单：**只要离开 GPU 做别的事，就还回去**。

- `acquireIfNecessary` —— 数据要上卡之前的最后一刻（shuffle 读攒够批、文件读完成解码）。
- `releaseIfNecessary` —— 结果已经落到主机内存、准备做 IO 之前（shuffle 写的三条路径都有这一步）。
- task 结束时由 `onTaskCompletion` 兜底释放，并把峰值内存、重试次数这些指标写进 `GpuTaskMetrics`。

### 一个副作用：它是 OOM 现场的第一手资料

因为信号量准确知道"此刻有哪些线程在 GPU 上"，`DeviceMemoryEventHandler` 在彻底放弃前会调 `GpuSemaphore.dumpActiveStackTracesToLog()`，分开打印"持有许可的线程"和"其他线程"的栈。查 GPU OOM 时这段日志通常比堆 dump 更有用 —— 它直接告诉你是哪几个算子在同时吃显存。

## 09. 与 Spark 内存模型的对照

读到这里会很自然地产生一个疑问：Spark 自己就有一套成熟的内存管理 —— `UnifiedMemoryManager` 配 `ExecutionMemoryPool`，有 per-task 的下限保证、上限、还有 storage 与 execution 之间的相互借用。cuDF-Spark 为什么不照搬？

把两边放在一起看，差异可以归结成一句话：**Spark 是"授予式硬记账"，cuDF-Spark 是"准入式软限流"。**

### Spark 那一侧

`ExecutionMemoryPool.acquireMemory` 的逻辑很直接 —— consumer 必须先申请才能用：

```scala
val maxMemoryPerTask = maxPoolSize / numActiveTasks
val minMemoryPerTask = poolSize / (2 * numActiveTasks)

val maxToGrant = math.min(numBytes, math.max(0, maxMemoryPerTask - curMem))
val toGrant    = math.min(maxToGrant, memoryFree)

// 连 1/2N 都给不了，就阻塞等别人释放
if (toGrant < numBytes && curMem + toGrant < minMemoryPerTask) {
  lock.wait()
} else {
  memoryForTask(taskAttemptId) += toGrant
  return toGrant   // 可能小于申请量 —— consumer 自己负责 spill
}
```

三个要点：**下限** `poolSize / 2N` 是硬保证，拿不到就挂起；**上限** `maxPoolSize / N` 随活跃 task 数动态变化，新 task 一进来所有人的份额立刻缩水（靠 `notifyAll()` 唤醒重算）；而**授予量小于申请量是正常返回值**，不是错误 —— consumer 收到之后按契约自己 spill。

另外 `maybeGrowExecutionPool` 允许 execution 侧驱逐 storage，把 storage pool 压缩到 `storageRegionSize` 为止；反方向不行。这就是"软边界"。

### cuDF-Spark 那一侧

GpuSemaphore 只在**进入 GPU 之前**要一次许可，份额是 `memToPermitsWithMax(estimator.estimate())`。拿到之后，这个 task 实际用了多少显存**没有任何人检查**。超了的后果不是"申请被拒绝"，而是 RMM 分配失败 → 触发 spill → 抛 RetryOOM。

![图 11](fig-11.png)

**图 11** — 同一个量的两种管法。Spark 把"能用多少"变成一个被强制执行的区间；cuDF-Spark 只把它当成一个准入时的估计值，真正的约束推迟到分配失败那一刻才生效。

|  | Spark UnifiedMemoryManager | cuDF-Spark |
|---|---|---|
| 申请方式 | 增量，每次 `acquireExecutionMemory(n)`，返回 granted ≤ n | 一次性拿够整个 task 的估计份额 |
| 下限保证 | `poolSize / (2N)`，拿不到就 `lock.wait()` | 没有 |
| 上限 | `maxPoolSize / N`，**随 N 动态变化** | 封顶在池总量，**不随 N 变化** |
| N 变化时 | `notifyAll()`，所有 task 重算份额 | 不重算，已持有许可的不受影响 |
| 超额 | 授予量不足是正常返回值，consumer 自己 spill | 实际用量无人检查，超了由失败回调兜底 |
| 排队顺序 | 近似公平：唤醒后重新竞争 | 按 task 优先级，老 task 优先 |
| 两类内存 | storage ⇄ execution 软边界，execution 可驱逐 storage 到 `storageRegionSize`，反向不行 | 没有 storage 概念 |

### 为什么不照搬

> 因为**在 GPU 上，算子没法在分配之前知道自己要用多少**。cuDF 一次 join 或 agg 的峰值取决于 key 的基数、null 比例、字符串的实际长度 —— 这些要跑完才知道。Spark 的 `UnsafeExternalSorter` 可以说"我要 n 字节"，GPU 算子说不出这句话。硬记账的前提不成立，就只能换成"预测 + 事后补救"。

这也解释了前面几节的几个设计为什么长成那样：动态内存估计（用同 stage 其他 task 的实测峰值修正）是在补"预测"这一半；三级 spill 和重试框架是在补"事后补救"那一半。两者合起来，近似地实现了 Spark 那套硬约束想达到的效果。

### 一个容易混淆的点：spill store 不是 storage pool

Spark 的 storage pool 是**预留出来给缓存用的区域**，有 `storageRegionSize` 保底，execution 再饿也不能把它挤到零。cuDF-Spark 的 device store 性质完全不同 —— 它装的是**已经在用、但碰巧可以被驱逐**的数据。没有预留，也就没有保底：只要有人要内存，store 里所有 spillable 的东西都是候选受害者。

所以"storage 和 execution 可以相互转换"这件事在 cuDF-Spark 里**没有对应物**，倒不是因为做不到，而是因为这里压根没有划出两个区域。显存池是一整块，谁拿到算谁的，拿不到就从别人那里挤 —— 挤谁由 task 优先级决定。

## 10. RMM 与 cuCascade

先澄清一个最容易误会的点：**cuCascade 不是 RMM 的替代品，它建在 RMM 之上。**它的 README 把 RMM 列为核心依赖，GPU tier 的默认分配器就是 RMM 的 `cuda_async_memory_resource`，而它的核心类 `reservation_aware_resource_adaptor` 包住的也正是一个 `rmm::mr::device_memory_resource*`。

所以这不是"两个分配器二选一"，而是两个**不同层次**的东西：

> RMM 回答的是：**这一次分配，从哪里拿到这些字节。**<br>
>  cuCascade 回答的是：**这份数据此刻该待在哪一层，谁有权占多少。**

### 逐项对照

|  | RMM | cuCascade |
|---|---|---|
| 层次 | 分配器 | 数据与配额的编排层，构建在 RMM 之上 |
| 抽象单位 | 字节：`void*` + size + stream | `data_batch` —— 一份有身份、有状态、能在层间移动的数据 |
| 分配模型 | 直接要，要不到抛 `rmm::out_of_memory` | **先 reserve 再 alloc**；reserve 不到就在条件变量上等别人释放 |
| 层级 | 单层 —— device 或 host，由你选哪个 resource | 内建三层：GPU / pinned host / disk，带自动降级 |
| 失败处理 | 一个 `RmmEventHandler.onAllocFailure` 回调 | **可插拔 policy**：`reservation_limit_policy` 管超额，`oom_handling_policy` 管分配失败 |
| 并发粒度 | per-stream free list | per-stream 或 per-thread 的 reservation 追踪，二选一 |
| 拓扑感知 | 无 | NUMA 区域发现、GPU-CPU 亲和、pool peer access 的双向字节验证 |
| 防抖动 | 无 | **滞回带**：85% 触发降级，一直降到 65% 才停 |
| 碎片 | 自己管（见 第 02 节） | GPU tier 委托给 RMM；host tier 用**固定 1 MB block** 从根上消除外部碎片 |

### reservation 模型长什么样

最本质的差别是这一条。RMM 的调用方直接伸手要内存；cuCascade 的调用方必须先拿到一张"票"：

```text
// 策略 1：任意一块 GPU
auto res = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), 1ULL << 30);

// 策略 3：先试 GPU，不行退到 host
auto res = manager.request_reservation(
    any_memory_space_in_tiers({Tier::GPU, Tier::HOST}), 512ULL << 20);

// 策略 4：明确要求从 GPU 降级到 host
auto res = manager.request_reservation(
    any_memory_space_to_downgrade(gpu_space_id, Tier::HOST), data_size);
```

拿不到票就阻塞在条件变量上，直到系统里有人释放 reservation 并发出通知。票的生命周期是 RAII 的：`reservation` 析构 → arena 释放预留字节 → `notification_channel` 唤醒等待者。

超额使用也不是无人过问的 —— 它由 policy 决定怎么办：

```text
fail_reservation_limit_policy      // 默认：超了就抛
ignore_reservation_limit_policy    // 软限制：允许超，只记账
increase_reservation_limit_policy(1.25, false)  // 自动把预留扩大 1.25 倍
```

### 和 cuDF-Spark 是什么关系

这才是真正有意思的地方。把前面九节讲的东西和 cuCascade 并排看，会发现**它们在解决同一类问题，只是一个长在 JVM 里、一个长在 C++ 里**：

![图 12](fig-12.png)

**图 12** — 两个栈的形状几乎一样。差别在于 cuDF-Spark 的中间两层是为 Spark 的 task 模型量身写的，而 cuCascade 把同一组职责做成了与引擎无关的库。

| cuDF-Spark | cuCascade |
|---|---|
| `SpillableHandle` + 三个 store | `data_batch` + 三个 `memory_space` |
| `GpuSemaphore` 的内存许可 | `reservation` + `memory_reservation_manager` |
| `DeviceMemoryEventHandler.onAllocFailure` | `oom_handling_policy`（可插拔） |
| `taskPriority` 决定 spill 顺序 | 可插拔的降级选择 policy |
| `materialize()` + 手工 `withResource` | `read_only_data_batch` / `mutable_data_batch` RAII accessor |
| 两阶段释放 + `spilling` 标志 | `std::shared_mutex` 守护的 idle / read_only / mutable 状态机 |
| 无 | 滞回带防抖动、NUMA 感知 |

### 两处值得借鉴的设计

**滞回带。**第 09 节末尾我提到 cuDF-Spark 的 device store 没有"保底区"，所有 spillable 的东西都是候选受害者，缺少抖动保护。cuCascade 用两个阈值解决这件事：`downgrade_trigger_fraction = 0.85` 触发降级，`downgrade_stop_fraction = 0.65` 才停手。中间这 20 个百分点就是滞回带 —— 一旦开始降级就降到位，避免"刚降下去又立刻升回来"的反复搬运。

**固定尺寸块消除碎片。**cuCascade 的 host tier 用 `fixed_size_host_memory_resource`：所有分配都是 1 MB 的定长块，要 10 MB 就给 10 个块（`multiple_blocks_allocation`，RAII 归还）。代价是最多浪费不到 1 MB，换来的是 host tier **结构上不可能有外部碎片**。对比之下，cuDF-Spark 的 `HostAlloc` 是变长分配，pinned 池被打散之后只能靠 fallback 到 non-pinned 来兜底。

> 需要说清楚的是，这两处差异不代表 cuDF-Spark 做错了 —— 它背着一个 cuCascade 没有的约束：必须贴合 Spark 的 task 生命周期和 JVM 的对象所有权。reservation 模型要求调用方"先说出要多少"，而第 09 节解释过，GPU 算子恰恰说不出这句话。cuCascade 把这个义务交还给了应用层，代价是应用层要自己估。

## 附. 相关配置速查

项目已更名为 **NVIDIA cuDF plugin for Apache Spark**（仓库 `NVIDIA/cudf-spark`），原名 RAPIDS Accelerator for Apache Spark。但 **artifact 名、配置前缀 `spark.rapids.*` 和 Scala 包名 `com.nvidia.spark.rapids` 暂时保持不变**，所以下面的配置项仍然是 `rapids` 开头。

| 配置项 | 作用 |
|---|---|
| spark.rapids.memory.gpu.pool | RMM 分配器模式：`default`(pool) / `arena` / `async` / `none` |
| spark.rapids.memory.gpu.allocFraction | 池占可用空闲显存的比例 |
| spark.rapids.memory.gpu.minAllocFraction | 池大小下限（占总显存比例），算出来更小就报错 |
| spark.rapids.memory.gpu.maxAllocFraction | 池大小上限（占总显存比例） |
| spark.rapids.memory.gpu.reserve | 留给 cuDF / cuBLAS 等库直接 `cudaMalloc` 的量 |
| spark.rapids.memory.gpu.oomMaxRetries | spill 不出东西时，同步后重试的次数 |
| spark.rapids.memory.gpu.oomDumpDir | GPU OOM 时的 heap dump 目录 |
| spark.rapids.memory.uvm.enabled | 启用 CUDA 统一内存（与 `async` 模式互斥） |
| spark.rapids.memory.pinnedPool.size | pinned 主机内存池大小 |
| spark.rapids.memory.host.offHeapLimit.enabled | 是否对堆外主机内存做上限跟踪 |
| spark.rapids.memory.host.offHeapLimit.size | 堆外主机内存上限（不设则从 executor overhead 推算） |
| spark.rapids.memory.host.spillStorageSize | host spill store 的容量上限 |
| spark.rapids.sql.concurrentGpuTasks | 并发 GPU task 的基准值，换算成许可份额 |
| spark.rapids.sql.concurrentGpuTasks.dynamic | 按实测峰值动态修正每个 task 的内存估计 |
| spark.rapids.sql.batchSizeBytes | 目标批大小，几乎所有算子的切分基准 |

### Shuffle 与 Kudo

| 配置项 | 作用 |
|---|---|
| spark.rapids.shuffle.mode | `MULTITHREADED`（默认）/ `UCX` / `CACHE_ONLY` |
| spark.rapids.shuffle.kudo.serializer.enabled | 启用 Kudo 序列化，默认 `true` |
| spark.rapids.shuffle.kudo.serializer.write.mode | 写侧序列化在 `CPU`（默认）还是 `GPU` |
| spark.rapids.shuffle.kudo.serializer.read.mode | 读侧拼装在 `CPU` 还是 `GPU`（默认） |
| spark.rapids.shuffle.kudo.serializer.debug.mode | `NEVER` / `ALWAYS` / `ONFAILURE`，dump kudo table 排查序列化问题 |
| spark.rapids.shuffle.compression.codec | `zstd` / `lz4` / `copy` / `none`（默认），仅 `contiguousSplit` 路径生效 |
| spark.rapids.shuffle.multiThreaded.writer.threads | MULTITHREADED 模式的写线程数 |
| spark.rapids.shuffle.multiThreaded.reader.threads | MULTITHREADED 模式的读线程数，默认 20 |
| spark.rapids.shuffle.multithreaded.skipMerge | 跳过分片合并；需关闭 ESS 且开启 off-heap 限额 |
| spark.rapids.shuffle.partitioning.maxCpuBatchSize | CPU 切分路径下单个分片的上限，超了会 reslice |

### 排查 GPU OOM 的顺序

1. 先看 `gpuSpillToHostBytes` / `gpuSpillToHostTime` 这组 task 指标 —— spill 量大说明是真不够，spill 量为零但 OOM 说明数据没被正确纳入 spill 托管（漏了 `SpillableColumnarBatch`，或者某处引用计数没还）。
2. 看 `gpuMaxDeviceMemoryBytes` 的分布 —— 是所有 task 均匀偏高（该降并发），还是个别 task 尖峰（该查数据倾斜）。
3. 看 OOM 日志里 `dumpActiveStackTracesToLog` 打出的栈 —— "持有许可的线程"那一组直接指出是哪些算子在同时吃显存。
4. 如果 OOM 发生在 shuffle 写侧，看 `rapidsShuffleSerializationStreamTime` 和 `shuffle.mode` —— UCX 模式下数据留在显存，峰值天然比 MULTITHREADED 高；`contiguousSplit` 的 2× 峰值也只在这条路上出现。
5. 调参的顺序通常是：先降 `batchSizeBytes`，再降 `concurrentGpuTasks`，最后才考虑动 `allocFraction`。

---

本文的网页版（带可交互目录、深色模式和矢量图）：

<https://claude.ai/code/artifact/3efd0729-0f80-4a77-8426-6dfab4326b2a>
