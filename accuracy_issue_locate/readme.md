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

TODO: walk `index.jsonl` in order, compare each tensor with its CPU file, report the first one over threshold.

## 6. After finding the first bad node

TODO

## 7. Limitations

- Ops nested in regions (e.g. `hip.if` branches) are not dumped.
- Pure views (e.g. `Reshape`) and folded constants have no dump file.
- When one ONNX node lowers to several HIP ops, only the last one is dumped.
