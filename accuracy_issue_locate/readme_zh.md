# hip-ep 精度对数指南

## 用法

本文有两种用法。第 1–5 节和第 7 节两种用法共用。差别从判定出第一个异常算子之后开始。

| 用法 | 做到哪一步 | 之后 |
|---|---|---|
| 定位模式 | 找到第一个异常算子 | 在对话中指出该层和原因，询问是否结合代码分析，由用户决定 |
| 自动模式 | 找到第一个异常算子 | 直接结合代码修复并复测，直到整网正常或达到时间上限 |

开始时说明用哪一种。自动模式同时给出连续工作的时间上限。

CPU EP 的 dump 不重复生成，除非已有结果不可用。hip-ep 的 dump 是否重新生成，见第 6 节自动模式。

## 1. 原理

误差会沿数据流向下游传递：第一个精度出错的节点，其误差会传递给之后所有依赖它的节点。因此下游节点的误差可能并不是它自己造成的，只有按执行顺序 **第一个** 误差很大的节点才是真正的根因。

整网完全对数的方法是循环：

1. 按执行顺序找到第一个误差很大的节点。
2. 修复该节点。
3. 按第 6 节自动模式决定是否重新 dump，然后对比，回到第 1 步。
4. 直到整网没有误差很大的节点为止。

- GPU 侧：hip-ep 在第一次推理时把每个 ONNX tensor 写成 `.npy`（hip-ep commit `91fa0eef3`）。
- CPU 侧：ONNX Runtime CPU EP 自带的 node dump。
- 两边文件名使用同一规则（tensor 名中非字母数字字符替换为 `_`），可按文件名直接配对。

## 2. 生成 CPU 参考 dump

1. 先确认当前环境里、实际会加载的那份 ONNX Runtime 是否已经带 node dump。已带则直接使用，不必重新编译。

   这是编译期选项 `onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON`。打开后，构建信息中含有 `dump-node=1`。

   - Python 包：

     ```bash
     python -c "import onnxruntime as ort; print(ort.get_build_info())"
     ```

     输出中含 `dump-node=1` 即已打开。

   - 本地编译的动态库（`helper/main.py` 里 `CPUEP_LIB_DIRS` 指向的那一份）：看对应构建目录的 `CMakeCache.txt` 是否为 `onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS:BOOL=ON`，或在 `onnxruntime.dll` / `libonnxruntime.so` 中查找字符串 `dump-node=1`。

   要跑 CPU dump 的那一份库没有 `dump-node=1` 时，才重新编译并打开 node dump：

   ```bash
   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON
   ```

2. 运行前设置环境变量：

   ```bash
   export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1
   export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
   export ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION=files
   export ORT_DEBUG_NODE_IO_OUTPUT_DIR=dump   # see note below
   export ORT_DEBUG_NODE_IO_DUMPING_DATA_TO_FILES_FOR_ALL_NODES_IS_OK=1

   # Reduce log volume
   export ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT=0
   export ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA=0
   ```

   备注：`ORT_DEBUG_NODE_IO_OUTPUT_DIR` 是保存目录。相对路径相对于运行进程的当前工作目录，上例即 `<cwd>/dump/`。ONNX Runtime 不会自动创建该目录，需要提前 `mkdir`。每个 tensor 保存为 `<tensor名>.tensorproto`。

3. 格式转换：CPU 侧原始文件是 `<tensor名>.tensorproto`，对比前要变成同名 `.npy`。用 `helper/main.py` 的 `dump_cpuep` 一次完成运行和转换。

   先在 `helper/main.py` 里填好 `CPUEP_LIB_DIRS`（须是带 `dump-node=1` 的那份库）和 `RUNNER`（留空则使用 `helper/test_onnx_runner/build` 下的 `test_onnx_runner`）。输入列表是文本文件，每行一个 `.npy`，顺序与模型输入一致。

   在 `accuracy_issue_locate` 目录下：

   ```bash
   python helper/main.py dump_cpuep \
       --model model.onnx \
       --inputs inputs.txt \
       --dir helper/dump_cpuep
   ```

   相对路径按当前工作目录解析。`--dir` 必须是空目录；省略时写入 `helper/dump_cpuep`。该命令自己设置第 2 步的环境变量，把 `.tensorproto` 写到临时目录，再逐个转成 `.npy`，然后删除临时目录。日志在 `helper/lastrun_dump_cpuep.log`。

   转换与 `tensorproto_to_npy` 相同：按 ONNX `TensorProto` 解析文件，用 `onnx.numpy_helper.to_array` 得到数组，再 `numpy.save` 成 `<原 stem>.npy`。stem 保持不变，以便和第 4 节的 GPU `.npy` 按文件名配对。

   第 2 步已经留下 `.tensorproto`、且不需要重跑推理时，对其中每个文件做同样的转换，把 `.npy` 写到另一个目录，再作为对比的 `--ref`。

## 3. hip-ep tensor dump 功能（commit `91fa0eef3`）

该 commit 为 hip-ep 增加逐层 dump：在第一次推理时，把 HIP 图计算出的每个 ONNX tensor 写到磁盘，文件名与 ONNX Runtime CPU node dump 一致，便于逐个配对。设计文档见 hip-ep `docs/design/tensor-dump.md`。

工作流程：

1. **记录名字**：`hip-tag-onnx-output-locs`（在 `simplify-onnx` 之前）把每个 `onnx.*` op 的输出 tensor 名写进它的 Location，使名字能穿过 ONNX→HIP 转换。
2. **插入 dump**：`hip-insert-tensor-dump`（在内存规划之后、HIP→LLVM 之前）在每个带名字的 HIP op 后面插入 `hip.dump_tensor`。插在内存规划之后，buffer 布局与正常编译完全一致，不增加分配。
3. **运行时写文件**：`hip.dump_tensor` 降级为 `hipdnn_ep_dump_tensor` 调用，同步 stream、拷回 host，写出 `.npy` 并追加一行到 `index.jsonl`。

两个 pass 只在编译期设置了 `HIPDNN_EP_DUMP_DIR` 时才加入 pipeline，默认编译不受影响。

## 4. 生成 GPU dump

| 环境变量 | 含义 |
|---|---|
| `HIPDNN_EP_DUMP_DIR` | dump 目录；编译期和运行期都要设置 |
| `HIPDNN_EP_DUMP_START` | 写出的起始 index（含），默认 0 |
| `HIPDNN_EP_DUMP_END` | 写出的结束 index（含），默认不限 |

```bash
export HIPDNN_EP_DUMP_DIR=dump_gpu
export HIPDNN_EP_DUMP_START=0
export HIPDNN_EP_DUMP_END=6666
```

建议使用空目录，避免旧文件混入对比。

输出：每个 tensor 一个 `<tensor名>.npy`，外加 `index.jsonl`，按执行顺序记录 `index`/`name`/`dtype`/`shape`。

注意：
- dump 算子在编译期插入。开启前要清模型缓存，否则命中不带 dump 的缓存不会写文件。
- 只写每个 session 的第一次推理。
- 开启 dump 会同步拷贝，不要用来测性能。

## 5. 对比

通过 `helper/main.py` 的 `accuracy` 子命令调用 `helper/accuracy_dirs.py`，把 CPU dump 与 GPU dump 按 ONNX 拓扑顺序对比，并写出 CSV。该子命令固定传入 `--order topo`。

在 `accuracy_issue_locate` 目录下：

```bash
python helper/main.py accuracy \
    --model model.onnx \
    --ref  dump \
    --test dump_gpu \
    --csv  report.csv
```

`--ref` 是 CPU 参考目录，里面应是 `.npy` 或 `.bin`（`dump_cpuep` 直接写出 `.npy`；第 2 步手工得到的 `.tensorproto` 需先按该节转换成 `.npy`）。`--test` 是第 4 节的 GPU 目录。`--model` 提供拓扑顺序：图输入在前，其后按 `graph.node` 排列，同一节点的多个输出共用该节点的序号。不提供可用的模型时，脚本会改按 `l2` 排序，得到的 CSV 不能用来找第一个错误节点。

`main.py accuracy` 比较进程的工作目录是 `helper/`，传入的相对路径按 `helper/` 解析。省略 `--ref`、`--test`、`--csv` 时，默认比较 `helper/dump_cpuep` 和 `helper/dump_hipep`，CSV 写到 `helper/report.csv`。

脚本按文件名配对，并自动处理 `InsertedPrecisionFreeCast_` 前缀和多输出算子的 `_out{N}` 后缀。CSV 与终端表格使用去掉这些前后缀后的逻辑 tensor 名，行顺序即拓扑顺序（`idx` 从 0 开始，`topo_idx` 递增）。

从上往下读 CSV。误差会传到下游，所以主指标第一次变差的那一行是根因，它的 `op_type` 就是要修的节点。后续是否重新 dump 再对比，按第 6 节的用法。

| 列 | 含义 |
|---|---|
| `topo_idx` | 拓扑序号，越小越靠前 |
| `op_type` | 产生该 tensor 的算子；图输入为 `<input>` |
| `orig_name` | ONNX tensor 名 |
| `l2` | `ref - test` 的 L2 范数 |
| `cos` | 两侧向量的余弦相似度，同方向为 1 |
| `status` | `EXACT` / `PASS` / `WARN` / `FAIL` / `NAN/INF` |

`a_*` 是 ref（CPU），`b_*` 是 test（GPU）。`status` 按 `rel_l2` 和 `max_abs` 分档：`l2 == 0` 为 `EXACT`，`rel_l2 < 1e-3` 或 `max_abs < 1e-3` 为 `PASS`，`rel_l2 < 1e-2` 为 `WARN`，否则为 `FAIL`；任一侧含 NaN 或 Inf 为 `NAN/INF`。`NAN/INF` 与模型类型无关，该行即出错。其余行按模型看主指标：

- **CNN**：主要看 `l2`。对齐的行 `l2` 为 0 或处于数值噪声；从上往下，`l2` 第一次明显变大的行就是第一个错误节点。
- **LLM**：主要看 `cos`。激活元素很多，`l2` 会随元素个数变大；`cos` 衡量方向是否一致。健康行的 `cos` 非常接近 1；从上往下，`cos` 第一次明显下降的行就是第一个错误节点。

## 6. 判定异常算子

每个算子的 GPU 结果与 CPU 参考都允许存在数值差。只有准确度出现较大偏差时，才断言某一层算子异常。偏差是否足够大，由 AI 综合判断：模型类型（CNN 看 `l2`，LLM 看 `cos`）、该行相对上游健康行的变化，以及误差是否从这一行开始向下游传递。

### 定位模式

在对话中指出异常层（`topo_idx`、`op_type`、`orig_name`），并说明原因。同时询问是否需要结合代码继续分析，由用户决定后再进行。

### 自动模式

不再询问，直接按下面的循环工作：

1. 按上面的标准确定第一个异常算子。整网已没有较大偏差时结束。
2. 在对话中记下该层和原因。
3. 结合代码修复该算子。
4. 再做第 5 节对比。CPU EP 的 dump 继续沿用，只有文件缺失或读不出来时才按第 2 节重新生成。hip-ep 酌情处理：本轮修改会改变该算子的 GPU 结果，或已有 dump 与当前编译结果不一致时，按第 4 节重新生成；已有 dump 仍对应当前实现时，直接拿它对比。
5. 回到第 1 步。

满足任一条件即停止，并汇报已修复的层、当前仍异常的层：

- 整网主指标恢复正常。
- 连续工作达到开始时给出的时间上限。
- 修复后第一个异常点仍是同一个 `orig_name`。这次修复视为未生效，停在该层，避免在时间上限内反复改同一处。

## 7. 限制

- 区域内嵌的 op（如 `hip.if` 分支）不会 dump。
- 纯 view（如 `Reshape`）和常量折叠的 tensor 没有 dump 文件。
- 一个 ONNX 节点拆成多个 HIP op 时，只 dump 最后一个。
