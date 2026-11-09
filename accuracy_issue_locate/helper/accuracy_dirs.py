#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较两个目录中同名 tensor dump 的精度（L2 范数）。

排序方式（``--order``）
    l2     按 L2 范数从大到小（默认；加 ``--asc`` 可改为升序）
    topo   按 ONNX 模型的拓扑顺序从前到后（需要 ``--model``；加 ``--asc`` 反转）

文件命名规律
    ONNX 张量名（如 ``/model/layers.0/foo``）中的所有非字母数字字符
    会被替换为 ``_``，形成文件名（如 ``_model_layers_0_foo``），
    扩展名通常为 ``.bin`` 或 ``.npy``。

``InsertedPrecisionFreeCast_`` 前缀
    某些算子（如 Sigmoid、Mul、Sqrt 等）在 CPU EP 不支持 fp16，ORT 会自动插入
    PrecisionFree Cast 节点，把 fp16 临时 cast 成 fp32 给该算子计算，再 cast 回来。
    这些 Cast 节点的输出 dump 时会带 ``InsertedPrecisionFreeCast_`` 前缀，
    且 dtype 为 fp32。本脚本会：
      * 配对：把 ``InsertedPrecisionFreeCast_xxx`` 与对侧 ``xxx`` 视为同一逻辑张量
        进行配对（一侧带前缀、另一侧不带的情况下）。
      * 加载：带前缀的 ``.bin`` 强制按 float32 读取。
      * 比较：把 fp32 数据先 cast 成 fp16 模拟精度截断，再与对侧比较。

``_out{N}`` 后缀（多输出算子）
    ROCm/HIP 后端在 dump 多输出算子时，会在张量文件名末尾追加 ``_out{N}``
    （N 为该输出在节点中的索引），而 baseline 侧通常按 ONNX 张量名命名、不带这个后缀。
    本脚本会把 ``xxx_out{N}`` 与对侧的 ``xxx`` 视为同一张量进行配对
    （同一逻辑名只配对一次，按 N 升序优先匹配，即 ``_out0`` 先匹配）。
    例：``_model_layers_0_linear_attn_LinearAttention_output_0_out0``
        与 ``_model_layers_0_linear_attn_LinearAttention_output_0`` 视为一对。

支持的格式
    .npy   numpy.load 直接获得 dtype/shape
    .bin   按对应 .npy 或 ONNX 模型推断的 dtype 加载

典型用法
    python accuracy_dirs.py \
        --ref  D:/gmz_rocm/dev/test/script/dump \
        --test D:/gmz_rocm/dev/test/models/qwen3.5-9b/dump_bin \
        --model D:/gmz_rocm/dev/test/models/qwen3.5-9b/prefill_p512m16384.onnx \
        --order topo
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

# Windows 控制台默认 cp1252，不能输出中文；强制切到 utf-8
for _stream_name in ("stdout", "stderr"):
    _s = getattr(sys, _stream_name, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

__SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(__SCRIPT_DIR__, "dump_cpuep")
DEFAULT_TEST = os.path.join(__SCRIPT_DIR__, "dump_gpu")
DEFAULT_MODEL = os.path.join(__SCRIPT_DIR__, "decode_p512m16384.onnx")

INSERTED_CAST_PREFIX = "InsertedPrecisionFreeCast_"


class PairSpec(NamedTuple):
    """一对待比较的文件。

    logical_stem: 逻辑张量名（去掉 ``InsertedPrecisionFreeCast_`` 前缀），
                  用于在 ONNX 模型 / 拓扑表中查询。
    ref_path / test_path: 实际参与比较的文件路径。
    ref_is_cast / test_is_cast: 该侧文件是否是 ``InsertedPrecisionFreeCast_`` 输出。
                  True 时按 fp32 加载并在比较前 cast 到 fp16 模拟精度。
    """

    logical_stem: str
    ref_path: str
    test_path: str
    ref_is_cast: bool
    test_is_cast: bool


def tensor_name_to_filename(name: str) -> str:
    """ONNX 张量名 -> dump 文件名（去扩展名）。

    规则：所有非字母数字的字符都替换为 ``_``。
    例：``/model/layers.0/foo`` -> ``_model_layers_0_foo``
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def strip_inserted_cast_prefix(stem: str) -> Optional[str]:
    """如果 stem 以 ``InsertedPrecisionFreeCast_`` 开头，返回去掉前缀后的 stem；
    否则返回 None。

    例：``InsertedPrecisionFreeCast__model_layers_7_mlp_Mul_output_0``
        -> ``_model_layers_7_mlp_Mul_output_0``
    """
    if stem.startswith(INSERTED_CAST_PREFIX):
        return stem[len(INSERTED_CAST_PREFIX):]
    return None


ROCM_OUT_SUFFIX_RE = re.compile(r"^(.+)_out(\d+)$")


def strip_rocm_out_suffix(stem: str) -> Optional[Tuple[str, int]]:
    """如果 stem 以 ``_out{N}`` 结尾（N 为非负整数），返回 ``(去掉后缀的 stem, N)``；
    否则返回 None。

    多输出算子的 ROCm/HIP dump 文件名会在张量名末尾追加 ``_out{N}`` 后缀。
    例：``_model_layers_0_linear_attn_LinearAttention_output_0_out0``
        -> (``_model_layers_0_linear_attn_LinearAttention_output_0``, 0)
    """
    m = ROCM_OUT_SUFFIX_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return None


def build_pairs(
    ref_files: Dict[str, str],
    test_files: Dict[str, str],
) -> Tuple[List[PairSpec], List[str], List[str]]:
    """根据两个目录的 stem -> path 映射构建配对集合。

    优先级：
      1. 两侧 stem 完全相同 -> 普通配对（两侧若都带 cast 前缀则同时标 cast=both）
      2. 一侧带 ``InsertedPrecisionFreeCast_`` 前缀、另一侧不带 -> 去前缀后配对
      3. 一侧带 ``_out{N}`` 后缀（多输出算子的 ROCm 命名）、另一侧不带 -> 去后缀后配对
         同一逻辑名只配对一次，按 N 升序优先（即 ``_out0`` 先匹配）。

    返回 ``(pairs, only_ref_stems, only_test_stems)``。
    """
    pairs: List[PairSpec] = []
    matched_ref: set[str] = set()
    matched_test: set[str] = set()

    for stem in sorted(set(ref_files) & set(test_files)):
        is_cast = stem.startswith(INSERTED_CAST_PREFIX)
        logical = stem[len(INSERTED_CAST_PREFIX):] if is_cast else stem
        pairs.append(PairSpec(
            logical_stem=logical,
            ref_path=ref_files[stem],
            test_path=test_files[stem],
            ref_is_cast=is_cast,
            test_is_cast=is_cast,
        ))
        matched_ref.add(stem)
        matched_test.add(stem)

    for test_stem in sorted(test_files):
        if test_stem in matched_test:
            continue
        stripped = strip_inserted_cast_prefix(test_stem)
        if stripped is None:
            continue
        if stripped in ref_files and stripped not in matched_ref:
            pairs.append(PairSpec(
                logical_stem=stripped,
                ref_path=ref_files[stripped],
                test_path=test_files[test_stem],
                ref_is_cast=False,
                test_is_cast=True,
            ))
            matched_ref.add(stripped)
            matched_test.add(test_stem)

    for ref_stem in sorted(ref_files):
        if ref_stem in matched_ref:
            continue
        stripped = strip_inserted_cast_prefix(ref_stem)
        if stripped is None:
            continue
        if stripped in test_files and stripped not in matched_test:
            pairs.append(PairSpec(
                logical_stem=stripped,
                ref_path=ref_files[ref_stem],
                test_path=test_files[stripped],
                ref_is_cast=True,
                test_is_cast=False,
            ))
            matched_ref.add(ref_stem)
            matched_test.add(stripped)

    def _gather_out_candidates(
        files: Dict[str, str], already_matched: set,
    ) -> List[Tuple[int, str, str]]:
        """收集未匹配的 ``xxx_out{N}`` stem，返回 ``[(N, base_stem, full_stem), ...]``，
        按 N 升序排列。"""
        out: List[Tuple[int, str, str]] = []
        for stem in files:
            if stem in already_matched:
                continue
            info = strip_rocm_out_suffix(stem)
            if info is None:
                continue
            base, idx = info
            out.append((idx, base, stem))
        out.sort()
        return out

    for _idx, base, test_stem in _gather_out_candidates(test_files, matched_test):
        if base in ref_files and base not in matched_ref:
            is_cast = base.startswith(INSERTED_CAST_PREFIX)
            logical = base[len(INSERTED_CAST_PREFIX):] if is_cast else base
            pairs.append(PairSpec(
                logical_stem=logical,
                ref_path=ref_files[base],
                test_path=test_files[test_stem],
                ref_is_cast=is_cast,
                test_is_cast=is_cast,
            ))
            matched_ref.add(base)
            matched_test.add(test_stem)

    for _idx, base, ref_stem in _gather_out_candidates(ref_files, matched_ref):
        if base in test_files and base not in matched_test:
            is_cast = base.startswith(INSERTED_CAST_PREFIX)
            logical = base[len(INSERTED_CAST_PREFIX):] if is_cast else base
            pairs.append(PairSpec(
                logical_stem=logical,
                ref_path=ref_files[ref_stem],
                test_path=test_files[base],
                ref_is_cast=is_cast,
                test_is_cast=is_cast,
            ))
            matched_ref.add(ref_stem)
            matched_test.add(base)

    only_ref = sorted(set(ref_files) - matched_ref)
    only_test = sorted(set(test_files) - matched_test)
    return pairs, only_ref, only_test


def load_onnx_info(
    model_path: str,
) -> Tuple[
    Dict[str, Tuple[str, np.dtype, Optional[Tuple[int, ...]]]],
    Dict[str, int],
    Dict[str, str],
]:
    """从 ONNX 模型构建三份映射：

    1. ``filename -> (orig_name, dtype, shape)`` —— 给 ``.bin`` 兜底 dtype/shape。
    2. ``filename -> topo_index`` —— ONNX 图的拓扑顺序，供 ``--order topo`` 使用。
       顺序约定：``graph.input`` 优先（按出现顺序），其后按 ``graph.node`` 的顺序，
       同一节点的多个输出共享该节点的索引。
    3. ``filename -> op_type`` —— 张量是哪个算子的输出；图输入记 ``"<input>"``。
    """
    try:
        import onnx
        from onnx import helper as onnx_helper
    except ImportError:
        print("[warn] 未安装 onnx，无法从模型推断 dtype/拓扑顺序；只能依赖对侧 .npy 文件提供 dtype。",
              file=sys.stderr)
        return {}, {}, {}

    print(f"[info] 解析 ONNX 模型: {model_path}")
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    types: Dict[str, Tuple[str, np.dtype, Optional[Tuple[int, ...]]]] = {}

    def _add(name: str, type_proto) -> None:
        if not type_proto.HasField("tensor_type"):
            return
        tt = type_proto.tensor_type
        try:
            np_dtype = np.dtype(onnx_helper.tensor_dtype_to_np_dtype(tt.elem_type))
        except Exception:
            return
        shape: list[int] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(int(d.dim_value))
            else:
                shape.append(-1)  # 动态维
        types[tensor_name_to_filename(name)] = (name, np_dtype, tuple(shape))

    for vi in graph.value_info:
        _add(vi.name, vi.type)
    for vi in graph.output:
        _add(vi.name, vi.type)
    for vi in graph.input:
        _add(vi.name, vi.type)

    topo: Dict[str, int] = {}
    op_types: Dict[str, str] = {}
    idx = 0
    for vi in graph.input:
        key = tensor_name_to_filename(vi.name)
        topo.setdefault(key, idx)
        op_types.setdefault(key, "<input>")
        idx += 1
    for node in graph.node:
        for o in node.output:
            if not o:
                continue
            key = tensor_name_to_filename(o)
            topo.setdefault(key, idx)
            op_types.setdefault(key, node.op_type)
        idx += 1

    return types, topo, op_types


def list_dir_files(dir_path: str) -> Dict[str, str]:
    """返回 ``stem -> full_path``，stem 不含扩展名。"""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    out: Dict[str, str] = {}
    for f in os.listdir(dir_path):
        full = os.path.join(dir_path, f)
        if not os.path.isfile(full):
            continue
        stem, _ext = os.path.splitext(f)
        out[stem] = full
    return out


def load_tensor(
    path: str,
    dtype_hint: Optional[np.dtype] = None,
    shape_hint: Optional[Tuple[int, ...]] = None,
    force_fp32: bool = False,
) -> np.ndarray:
    """按扩展名加载张量；.bin 必须给出 dtype。

    ``force_fp32``：当为 True 时，``.bin`` 文件强制按 ``float32`` 读取，覆盖 ``dtype_hint``。
    用于 ``InsertedPrecisionFreeCast_xxx.bin``（CPU EP 把 fp16 转为 fp32 计算后保存的）。
    对 ``.npy`` 文件不影响（其 dtype 由文件头自带）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".bin":
        actual_dtype = np.dtype(np.float32) if force_fp32 else dtype_hint
        if actual_dtype is None:
            raise ValueError(f".bin 文件需要 dtype 提示: {path}")
        data = np.fromfile(path, dtype=actual_dtype)
        if shape_hint and all(d > 0 for d in shape_hint):
            try:
                data = data.reshape(shape_hint)
            except Exception:
                pass
        return data
    raise ValueError(f"不支持的扩展名 {ext!r}: {path}")


def coerce_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把两侧统一到能逐元素相减的形态：
    - 把整型/布尔型也提升到 float64（避免 underflow/overflow）
    - 形状不一致时回退到一维比较；元素数也不一致时取较小的前缀以避免崩溃。
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        a_arr = a_arr.ravel()
        b_arr = b_arr.ravel()
    if a_arr.size != b_arr.size:
        n = min(a_arr.size, b_arr.size)
        a_arr = a_arr.ravel()[:n]
        b_arr = b_arr.ravel()[:n]
    return a_arr.astype(np.float64, copy=False), b_arr.astype(np.float64, copy=False)


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """返回 L2 范数、max_abs、mean_abs、cosine、相对 L2，以及 NaN/Inf 计数。"""
    af, bf = coerce_pair(a, b)
    a_bad = int(np.isnan(af).sum() + np.isinf(af).sum())
    b_bad = int(np.isnan(bf).sum() + np.isinf(bf).sum())
    a_safe = np.nan_to_num(af, nan=0.0, posinf=0.0, neginf=0.0)
    b_safe = np.nan_to_num(bf, nan=0.0, posinf=0.0, neginf=0.0)
    a_flat = a_safe.ravel()
    b_flat = b_safe.ravel()
    diff = a_flat - b_flat

    l2 = float(np.linalg.norm(diff))
    if diff.size:
        max_abs = float(np.abs(diff).max())
        mean_abs = float(np.abs(diff).mean())
        # 有符号最大差 (a-b).max()：保留符号，配合 a_mean/b_mean 判断系统性偏置方向
        max_diff = float(diff.max())
        a_mean = float(a_flat.mean())
        b_mean = float(b_flat.mean())
    else:
        max_abs = 0.0
        mean_abs = 0.0
        max_diff = 0.0
        a_mean = 0.0
        b_mean = 0.0

    a_norm = float(np.linalg.norm(a_flat))
    b_norm = float(np.linalg.norm(b_flat))
    rel_l2 = l2 / a_norm if a_norm > 0 else (l2 if l2 > 0 else 0.0)
    if a_norm > 0 and b_norm > 0:
        cos = float(np.dot(a_flat, b_flat) / (a_norm * b_norm))
    else:
        cos = 1.0 if l2 == 0.0 else 0.0
    return dict(
        l2=l2,
        rel_l2=rel_l2,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_diff=max_diff,
        a_mean=a_mean,
        b_mean=b_mean,
        cos=cos,
        a_norm=a_norm,
        b_norm=b_norm,
        a_bad=a_bad,
        b_bad=b_bad,
        numel=int(af.size),
    )


def status_label(m: Dict[str, float]) -> str:
    if m["a_bad"] or m["b_bad"]:
        return "NAN/INF"
    if math.isnan(m["l2"]) or math.isinf(m["l2"]):
        return "NAN/INF"
    if m["l2"] == 0.0:
        return "EXACT"
    if m["rel_l2"] < 1e-3 or m["max_abs"] < 1e-3:
        return "PASS"
    if m["rel_l2"] < 1e-2:
        return "WARN"
    return "FAIL"


def fmt_shape(shape) -> str:
    if shape is None:
        return "-"
    return "x".join(str(s) for s in shape)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="比较两个目录下同名 tensor dump 的精度；可按 L2 范数或 ONNX 拓扑序排序。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref", default=DEFAULT_REF, help="参考目录（CPU/baseline）")
    p.add_argument("--test", default=DEFAULT_TEST, help="被测目录（待评估实现）")
    p.add_argument("--model", default=DEFAULT_MODEL, help="ONNX 模型路径，用于推断 .bin 的 dtype/shape")
    p.add_argument("--top", type=int, default=0, help="只显示前 N 项；0 表示全部")
    p.add_argument("--order", choices=("l2", "topo"), default="topo",
                   help="排序方式：l2=按 L2 范数; topo=按 ONNX 模型拓扑序，需要 --model")
    p.add_argument("--asc", action="store_true",
                   help="升序排列（l2 模式默认降序；topo 模式默认从前到后，加 --asc 时反转）")
    p.add_argument("--threshold", type=float, default=None, help="只显示 L2 ≥ 阈值的项")
    p.add_argument("--csv", default=None, help="同时把完整报告写入 CSV 文件")
    p.add_argument("--no-model", action="store_true", help="禁用从 ONNX 模型推断 dtype")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ref_files = list_dir_files(args.ref)
    test_files = list_dir_files(args.test)

    pairs, only_ref, only_test = build_pairs(ref_files, test_files)
    cast_pair_count = sum(1 for p in pairs if p.ref_is_cast or p.test_is_cast)

    print(f"[info] ref  目录: {args.ref}  ({len(ref_files)} 个文件)")
    print(f"[info] test 目录: {args.test}  ({len(test_files)} 个文件)")
    print(f"[info] 可对应的文件: {len(pairs)} 对（其中 InsertedPrecisionFreeCast 配对 {cast_pair_count} 对）；"
          f"仅在 ref 中: {len(only_ref)}；仅在 test 中: {len(only_test)}")

    onnx_info: Dict[str, Tuple[str, np.dtype, Optional[Tuple[int, ...]]]] = {}
    topo_index: Dict[str, int] = {}
    op_types: Dict[str, str] = {}
    if not args.no_model and args.model and os.path.isfile(args.model):
        try:
            onnx_info, topo_index, op_types = load_onnx_info(args.model)
            print(f"[info] 从 ONNX 模型获得 {len(onnx_info)} 条类型信息，"
                  f"{len(topo_index)} 条拓扑顺序，{len(op_types)} 条 op_type")
        except Exception as exc:
            print(f"[warn] 加载 ONNX 信息失败: {exc}", file=sys.stderr)
    elif args.model and not os.path.isfile(args.model):
        print(f"[warn] ONNX 模型不存在，跳过类型/拓扑推断: {args.model}", file=sys.stderr)

    rows = []
    for pair in pairs:
        logical = pair.logical_stem
        ref_path = pair.ref_path
        test_path = pair.test_path
        ref_ext = os.path.splitext(ref_path)[1].lower()
        test_ext = os.path.splitext(test_path)[1].lower()

        dtype_hint: Optional[np.dtype] = None
        shape_hint: Optional[Tuple[int, ...]] = None
        orig_name = logical

        npy_path: Optional[str] = None
        if ref_ext == ".npy" and not pair.ref_is_cast:
            npy_path = ref_path
        elif test_ext == ".npy" and not pair.test_is_cast:
            npy_path = test_path
        elif ref_ext == ".npy":
            npy_path = ref_path
        elif test_ext == ".npy":
            npy_path = test_path

        if npy_path is not None:
            try:
                tmp = np.load(npy_path, mmap_mode="r", allow_pickle=False)
                dtype_hint = tmp.dtype
                shape_hint = tuple(tmp.shape)
                del tmp
            except Exception as exc:
                print(f"[warn] {logical}: 读取 .npy 失败: {exc}", file=sys.stderr)

        if dtype_hint is None and logical in onnx_info:
            orig_name, dtype_hint, shape_hint = onnx_info[logical]
        elif logical in onnx_info:
            orig_name = onnx_info[logical][0]

        try:
            ref_arr = load_tensor(ref_path, dtype_hint, shape_hint, force_fp32=pair.ref_is_cast)
            test_arr = load_tensor(test_path, dtype_hint, shape_hint, force_fp32=pair.test_is_cast)
        except Exception as exc:
            print(f"[error] {logical}: 加载失败: {exc}", file=sys.stderr)
            continue

        if pair.ref_is_cast:
            ref_arr = ref_arr.astype(np.float16)
        if pair.test_is_cast:
            test_arr = test_arr.astype(np.float16)

        if pair.ref_is_cast and pair.test_is_cast:
            cast_mode = "both"
        elif pair.ref_is_cast:
            cast_mode = "ref"
        elif pair.test_is_cast:
            cast_mode = "test"
        else:
            cast_mode = ""

        m = compute_metrics(ref_arr, test_arr)
        rows.append(dict(
            stem=logical,
            orig_name=orig_name,
            cast=cast_mode,
            dtype=str(ref_arr.dtype),
            shape=fmt_shape(getattr(ref_arr, "shape", None)),
            **m,
            status=status_label(m),
            a_path=ref_path,
            b_path=test_path,
        ))

    if args.threshold is not None:
        rows = [r for r in rows if r["l2"] >= args.threshold]

    for r in rows:
        r["topo_idx"] = topo_index.get(r["stem"])
        r["op_type"] = op_types.get(r["stem"], "")

    if args.order == "topo":
        if not topo_index:
            print("[warn] 未获得 ONNX 拓扑信息，--order topo 回退为按 L2 排序", file=sys.stderr)
            rows.sort(key=lambda r: r["l2"], reverse=not args.asc)
        else:
            unknown = 10**9
            rows.sort(
                key=lambda r: (r["topo_idx"] if r["topo_idx"] is not None else unknown,
                               r["stem"]),
                reverse=args.asc,
            )
    else:
        rows.sort(key=lambda r: r["l2"], reverse=not args.asc)

    if args.top and args.top > 0:
        shown = rows[: args.top]
    else:
        shown = rows

    title_order = "拓扑顺序" if args.order == "topo" else "L2 范数"
    op_w = 20
    name_w = 56
    cast_w = 4
    print()
    print(f"=== 按{title_order}排序的精度对比报告 ===")
    print(f"{'idx':>4}  {'topo':>6}  {'op_type':<{op_w}}  {'name':<{name_w}}  "
          f"{'cast':<{cast_w}}  {'dtype':<8} {'shape':<22} {'numel':>10}  "
          f"{'L2':>11}  {'rel_L2':>10}  {'max_abs':>11}  {'mean_abs':>11}  "
          f"{'max_diff':>11}  {'a_mean':>11}  {'b_mean':>11}  {'cos':>9}  status")
    print("-" * 265)
    for i, r in enumerate(shown):
        name_display = r["orig_name"] if r["orig_name"] != r["stem"] else r["stem"]
        if len(name_display) > name_w:
            name_display = "…" + name_display[-(name_w - 1):]
        topo_str = str(r["topo_idx"]) if r.get("topo_idx") is not None else "-"
        op_str = r.get("op_type") or "-"
        if len(op_str) > op_w:
            op_str = op_str[: op_w - 1] + "…"
        cast_str = r.get("cast") or "-"
        print(f"{i:>4}  {topo_str:>6}  {op_str:<{op_w}}  {name_display:<{name_w}}  "
              f"{cast_str:<{cast_w}}  {r['dtype']:<8} {r['shape']:<22} {r['numel']:>10}  "
              f"{r['l2']:>11.4e}  {r['rel_l2']:>10.3e}  {r['max_abs']:>11.4e}  {r['mean_abs']:>11.4e}  "
              f"{r['max_diff']:>11.4e}  {r['a_mean']:>11.4e}  {r['b_mean']:>11.4e}  "
              f"{r['cos']:>9.5f}  {r['status']}")

    fail = sum(1 for r in rows if r["status"] in ("FAIL", "NAN/INF"))
    warn = sum(1 for r in rows if r["status"] == "WARN")
    print()
    print(f"汇总: 共 {len(rows)} 项；FAIL/NAN={fail}, WARN={warn}, "
          f"PASS+EXACT={len(rows) - fail - warn}")

    if only_ref:
        print(f"\n[info] 仅在 ref 中存在 ({len(only_ref)}):")
        for n in only_ref[:20]:
            print(f"   - {n}")
        if len(only_ref) > 20:
            print(f"   ... 共 {len(only_ref)} 个")
    if only_test:
        print(f"\n[info] 仅在 test 中存在 ({len(only_test)}):")
        for n in only_test[:20]:
            print(f"   - {n}")
        if len(only_test) > 20:
            print(f"   ... 共 {len(only_test)} 个")

    if args.csv:
        fields = ["idx", "topo_idx", "op_type", "stem", "orig_name", "cast", "dtype", "shape", "numel",
                  "l2", "rel_l2", "max_abs", "mean_abs", "max_diff", "a_mean", "b_mean", "cos",
                  "a_norm", "b_norm", "a_bad", "b_bad", "status",
                  "a_path", "b_path"]
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, r in enumerate(rows):
                row = {"idx": i, **{k: r.get(k) for k in fields if k != "idx"}}
                w.writerow(row)
            raw = csv.writer(f)
            raw.writerow([])
        print(f"[info] 已写入 CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
