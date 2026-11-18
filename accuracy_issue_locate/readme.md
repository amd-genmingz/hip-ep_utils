# hip-ep Accuracy Comparison Guide

## 1. Principle

Error propagates downstream along the data flow: the error of the first inaccurate node is passed on to every node that depends on it. A downstream node's error may therefore not be its own; only the **first** node with a large error, in execution order, is the real root cause.

Full-model alignment is a loop:

1. Find the first node, in execution order, with a large error.
2. Fix that node.
3. Re-dump and compare again, then go back to step 1.
4. Stop when no node in the whole model has a large error.

- GPU side: hip-ep writes every ONNX tensor as `.npy` during the first inference (hip-ep commit `91fa0eef3`).
- CPU side: ONNX Runtime CPU EP's built-in node dump.
- Both sides use the same file-name rule (non-alphanumeric characters in the tensor name become `_`), so files pair by name.

## 2. Generate the CPU reference dump

1. Build ONNX Runtime with node dump enabled:

   ```bash
   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON
   ```

2. Set these environment variables before running:

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

   Note: `ORT_DEBUG_NODE_IO_OUTPUT_DIR` is the save directory. A relative path is resolved against the running process's current working directory, so the example above writes to `<cwd>/dump/`. ONNX Runtime does not create the directory; `mkdir` it beforehand. Each tensor is saved as `<tensor name>.tensorproto`.

3. Format conversion: the CPU side saves TensorProto files, which need a separate script to convert them to `.bin`/`.npy` before comparing with the GPU side.

   TODO: conversion script

## 3. hip-ep tensor dump (commit `91fa0eef3`)

This commit adds per-node dumping to hip-ep: during the first inference it writes every ONNX tensor the HIP graph computes to disk, using the same file names as ONNX Runtime's CPU node dump so files pair one-to-one. Design doc: hip-ep `docs/design/tensor-dump.md`.

How it works:

1. **Record names**: `hip-tag-onnx-output-locs` (before `simplify-onnx`) stores each `onnx.*` op's output tensor names in its Location, so the names survive ONNX-to-HIP conversion.
2. **Insert dumps**: `hip-insert-tensor-dump` (after memory planning, before HIP-to-LLVM) inserts a `hip.dump_tensor` after each named HIP op. Running after memory planning keeps the buffer layout identical to a normal build and adds no allocation.
3. **Write files at run time**: `hip.dump_tensor` lowers to a `hipdnn_ep_dump_tensor` call, which synchronizes the stream, copies to host, writes the `.npy`, and appends a line to `index.jsonl`.

Both passes are added only when `HIPDNN_EP_DUMP_DIR` is set at compile time; a normal build is unaffected.

## 4. Generate the GPU dump

| Variable | Meaning |
|---|---|
| `HIPDNN_EP_DUMP_DIR` | Dump directory; must be set at both compile time and run time |
| `HIPDNN_EP_DUMP_START` | First index written (inclusive), default 0 |
| `HIPDNN_EP_DUMP_END` | Last index written (inclusive), default unbounded |

```bash
export HIPDNN_EP_DUMP_DIR=dump_gpu
export HIPDNN_EP_DUMP_START=0
export HIPDNN_EP_DUMP_END=6666
```

Use an empty directory so stale files never enter the comparison.

Output: one `<tensor name>.npy` per tensor, plus `index.jsonl` recording `index`/`name`/`dtype`/`shape` in execution order.

Notes:
- Dump ops are inserted at compile time. Clear the model cache first; a cached artifact compiled without dumps writes nothing.
- Only the first inference of each session is written.
- Dumping synchronizes and copies to host; never benchmark with it on.

## 5. Compare

Call `helper/accuracy_dirs.py` through the `accuracy` subcommand of `helper/main.py`. It compares the CPU dump with the GPU dump in ONNX topological order and writes a CSV. The subcommand always passes `--order topo`.

From the `accuracy_issue_locate` directory:

```bash
python helper/main.py accuracy \
    --model model.onnx \
    --ref  dump \
    --test dump_gpu \
    --csv  report.csv
```

`--ref` is the CPU reference directory and must contain `.npy` or `.bin` files (section 2 produces `.tensorproto`, which has to be converted first). `--test` is the GPU directory from section 4. `--model` supplies the topological order: graph inputs first, then `graph.node` order, with every output of one node sharing that node's index. Without a usable model the script falls back to sorting by `l2`, and that CSV cannot be used to find the first bad node.

`main.py accuracy` runs the comparison with `helper/` as its working directory, so relative paths passed to it resolve against `helper/`. Omitting `--ref`, `--test`, and `--csv` compares `helper/dump_cpuep` with `helper/dump_hipep` and writes `helper/report.csv`.

The script pairs files by name and handles the `InsertedPrecisionFreeCast_` prefix and the `_out{N}` suffix of multi-output ops. The CSV and the printed table use the logical tensor name with those affixes removed, in topological order (`idx` starts at 0, `topo_idx` increases).

Read the CSV from top to bottom. Error propagates downstream, so the first row whose primary metric gets worse is the root cause, and its `op_type` is the node to fix. After fixing it, dump again and compare again.

| Column | Meaning |
|---|---|
| `topo_idx` | Topological index; smaller is earlier |
| `op_type` | Op that produces the tensor; graph inputs are `<input>` |
| `orig_name` | ONNX tensor name |
| `l2` | L2 norm of `ref - test` |
| `cos` | Cosine similarity of the two vectors; 1 means the same direction |
| `status` | `EXACT` / `PASS` / `WARN` / `FAIL` / `NAN/INF` |

`a_*` is ref (CPU), `b_*` is test (GPU). `status` is bucketed by `rel_l2` and `max_abs`: `EXACT` when `l2 == 0`, `PASS` when `rel_l2 < 1e-3` or `max_abs < 1e-3`, `WARN` when `rel_l2 < 1e-2`, otherwise `FAIL`. `NAN/INF` means either side contains NaN or Inf; that row is already wrong for any model. For the other rows, use the primary metric for the model:

- **CNN**: read `l2`. A matched row has `l2` at 0 or at numerical noise. Walking downward, the first row whose `l2` jumps is the first bad node.
- **LLM**: read `cos`. Activations have many elements, so `l2` grows with the element count; `cos` measures whether the direction matches. A healthy row has `cos` very close to 1. Walking downward, the first row whose `cos` drops is the first bad node.

## 6. After finding the first bad node

Every op's GPU result may differ numerically from the CPU reference. Assert that a layer is wrong only when the accuracy shows a large deviation. Whether the deviation is large enough is a judgment the AI makes from the model type (CNN: `l2`, LLM: `cos`), how the row changes relative to the healthy rows above it, and whether the error starts at that row and then propagates downstream.

After that call, name the bad layer in the chat (`topo_idx`, `op_type`, `orig_name`) and give the reason. Also ask whether to continue by reading the code, and proceed only after the user decides.

## 7. Limitations

- Ops nested in regions (e.g. `hip.if` branches) are not dumped.
- Pure views (e.g. `Reshape`) and folded constants have no dump file.
- When one ONNX node lowers to several HIP ops, only the last one is dumped.
