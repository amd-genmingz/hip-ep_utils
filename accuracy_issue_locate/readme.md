# hip-ep Accuracy Comparison Guide

## Usage

This guide has two modes. Sections 1–5 and 7 are shared. The modes diverge after the first abnormal op has been decided.

| Mode | Stop after | Then |
|---|---|---|
| Locate | The first abnormal op is found | Name that layer and the reason in the chat, and ask whether to continue by reading the code. The user decides |
| Automatic | The first abnormal op is found | Fix it in the code and re-test, until the whole model is aligned or the time limit is reached |

State which mode to use at the start. For automatic mode, also give the time limit for continuous work.

Do not dump the CPU EP again unless the existing result is no longer usable. Whether to dump hip-ep again is decided in section 6, automatic mode.

## 1. Principle

Error propagates downstream along the data flow: the error of the first inaccurate node is passed on to every node that depends on it. A downstream node's error may therefore not be its own; only the **first** node with a large error, in execution order, is the real root cause.

Full-model alignment is a loop:

1. Find the first node, in execution order, with a large error.
2. Fix that node.
3. Decide whether to dump again as in section 6 (automatic mode), compare, then go back to step 1.
4. Stop when no node in the whole model has a large error.

- GPU side: hip-ep writes every ONNX tensor as `.npy` during the first inference (hip-ep commit `91fa0eef3`).
- CPU side: ONNX Runtime CPU EP's built-in node dump.
- Both sides use the same file-name rule (non-alphanumeric characters in the tensor name become `_`), so files pair by name.

## 2. Generate the CPU reference dump

1. First check whether the ONNX Runtime that this environment will actually load already includes node dump. If it does, use it and skip the rebuild.

   The feature is the compile-time option `onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON`. When it is on, the build info contains `dump-node=1`.

   - Python package:

     ```bash
     python -c "import onnxruntime as ort; print(ort.get_build_info())"
     ```

     `dump-node=1` in the output means the feature is present.

   - A locally built library (the one named by `CPUEP_LIB_DIRS` in `helper/main.py`): check that build's `CMakeCache.txt` for `onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS:BOOL=ON`, or search `onnxruntime.dll` / `libonnxruntime.so` for the string `dump-node=1`.

   Rebuild with node dump only when the library that will run the CPU dump does not contain `dump-node=1`:

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

3. Format conversion: the CPU side first writes `<tensor name>.tensorproto`. Comparison needs a `.npy` with the same stem. `dump_cpuep` in `helper/main.py` runs the model and converts the files in one step.

   Fill in `CPUEP_LIB_DIRS` in `helper/main.py` (the library that contains `dump-node=1`) and `RUNNER` (empty uses `test_onnx_runner` under `helper/test_onnx_runner/build`). The input list is a text file with one `.npy` path per line, in model-input order.

   From the `accuracy_issue_locate` directory:

   ```bash
   python helper/main.py dump_cpuep \
       --model model.onnx \
       --inputs inputs.txt \
       --dir helper/dump_cpuep
   ```

   Relative paths resolve against the current working directory. `--dir` must be empty; when omitted, files go to `helper/dump_cpuep`. The command sets the section 2 environment variables itself, writes `.tensorproto` files to a temporary directory, converts each one to `.npy`, then deletes the temporary directory. The log is `helper/lastrun_dump_cpuep.log`.

   Conversion is `tensorproto_to_npy`: parse the file as an ONNX `TensorProto`, turn it into an array with `onnx.numpy_helper.to_array`, and `numpy.save` it as `<original stem>.npy`. The stem is unchanged so the file pairs by name with the GPU `.npy` from section 4.

   When step 2 has already left `.tensorproto` files and the inference does not need to be repeated, convert each file the same way into another directory, then pass that directory as `--ref`.

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

`--ref` is the CPU reference directory and must contain `.npy` or `.bin` files (`dump_cpuep` writes `.npy` directly; `.tensorproto` files left by step 2 must be converted as described in that section). `--test` is the GPU directory from section 4. `--model` supplies the topological order: graph inputs first, then `graph.node` order, with every output of one node sharing that node's index. Without a usable model the script falls back to sorting by `l2`, and that CSV cannot be used to find the first bad node.

`main.py accuracy` runs the comparison with `helper/` as its working directory, so relative paths passed to it resolve against `helper/`. Omitting `--ref`, `--test`, and `--csv` compares `helper/dump_cpuep` with `helper/dump_hipep` and writes `helper/report.csv`.

The script pairs files by name and handles the `InsertedPrecisionFreeCast_` prefix and the `_out{N}` suffix of multi-output ops. The CSV and the printed table use the logical tensor name with those affixes removed, in topological order (`idx` starts at 0, `topo_idx` increases).

Read the CSV from top to bottom. Error propagates downstream, so the first row whose primary metric gets worse is the root cause, and its `op_type` is the node to fix. Whether to dump again before the next comparison follows section 6.

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

## 6. Deciding that an op is wrong

Every op's GPU result may differ numerically from the CPU reference. Assert that a layer is wrong only when the accuracy shows a large deviation. Whether the deviation is large enough is a judgment the AI makes from the model type (CNN: `l2`, LLM: `cos`), how the row changes relative to the healthy rows above it, and whether the error starts at that row and then propagates downstream.

### Locate mode

Name the bad layer in the chat (`topo_idx`, `op_type`, `orig_name`) and give the reason. Also ask whether to continue by reading the code, and proceed only after the user decides.

### Automatic mode

Do not ask. Run this loop directly:

1. Find the first abnormal op by the rule above. Stop when the whole model no longer has a large deviation.
2. Record that layer and the reason in the chat.
3. Fix the op in the code.
4. Compare again with section 5. Keep the CPU EP dump; regenerate it with section 2 only when files are missing or unreadable. For hip-ep, decide per round: regenerate with section 4 when this round's change affects that op's GPU result, or when the existing dump does not match the current build; when the existing dump still matches the current implementation, compare with it as it is.
5. Go back to step 1.

Stop when any one of these is true, and report the layers already fixed and the layer that is still wrong:

- The whole model's primary metric is back in range.
- Continuous work has reached the time limit given at the start.
- After the fix, the first abnormal point is still the same `orig_name`. Treat that fix as ineffective and stop on that layer, so the same spot is not edited again and again inside the time limit.

## 7. Limitations

- Ops nested in regions (e.g. `hip.if` branches) are not dumped.
- Pure views (e.g. `Reshape`) and folded constants have no dump file.
- When one ONNX node lowers to several HIP ops, only the last one is dumped.
