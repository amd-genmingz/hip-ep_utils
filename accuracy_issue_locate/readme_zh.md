# hip-ep 精度对数指南

## 1. 原理

误差会沿数据流向下游传递：第一个精度出错的节点，其误差会传递给之后所有依赖它的节点。因此下游节点的误差可能并不是它自己造成的，只有按执行顺序 **第一个** 误差很大的节点才是真正的根因。

整网完全对数的方法是循环：

1. 按执行顺序找到第一个误差很大的节点。
2. 修复该节点。
3. 重新 dump 并对比，回到第 1 步。
4. 直到整网没有误差很大的节点为止。

- GPU 侧：hip-ep 在第一次推理时把每个 ONNX tensor 写成 `.npy`（hip-ep commit `91fa0eef3`）。
- CPU 侧：ONNX Runtime CPU EP 自带的 node dump。
- 两边文件名使用同一规则（tensor 名中非字母数字字符替换为 `_`），可按文件名直接配对。

## 2. 生成 CPU 参考 dump

1. 编译 ONNX Runtime 时打开 node dump：

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

3. 格式转换：CPU 侧保存的是 TensorProto 格式，需要额外写脚本转换成 `.bin`/`.npy` 再与 GPU 侧对比。

   TODO: 转换脚本

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

TODO: 按 `index.jsonl` 的顺序逐个与 CPU 文件对比，报告第一个超出阈值的 tensor。

## 6. 定位到第一个错误节点后

TODO

## 7. 限制

- 区域内嵌的 op（如 `hip.if` 分支）不会 dump。
- 纯 view（如 `Reshape`）和常量折叠的 tensor 没有 dump 文件。
- 一个 ONNX 节点拆成多个 HIP op 时，只 dump 最后一个。
