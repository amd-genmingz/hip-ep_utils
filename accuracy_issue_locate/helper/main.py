"""Per-node accuracy comparison driver: CPU EP vs hip-ep.

Commands:
  dump_cpuep   run the model on CPU EP with ORT node dump, convert to .npy
  dump_hipep   run the model on hip-ep with HIPDNN_EP_DUMP_DIR
  accuracy     compare the two dump directories with accuracy_dirs.py

Fill in the user settings below before running.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent

# ---- user settings ---------------------------------------------------------
# Model and its input list (one .npy per line, in model input order).
MODEL = r""
INPUT_LIST = r""

# test_onnx_runner executable. Empty: look in ./test_onnx_runner/build
# (Visual Studio generators add a RelWithDebInfo subdirectory).
RUNNER = r""

# Directories searched for shared libraries, ahead of the system ones.
# CPU EP: an onnxruntime built with onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON.
CPUEP_LIB_DIRS = [
    r"",
]
# hip-ep: onnxruntime + hip-ep package + TheRock runtime.
HIPEP_LIB_DIRS = [
    r"",
]
# hip-ep plugin library (hipgpu.dll / libhipgpu.so).
HIPEP_LIBRARY = r""

# Extra environment for the hip-ep run.
HIPEP_ENV = {
    "HIPDNN_EP_DUMP_START": "0",
    # "HIPDNN_EP_DUMP_END": "6666",
}

# Graph optimization level passed to test_onnx_runner (-l).
ORT_OPT_LEVEL = "1"
# ----------------------------------------------------------------------------

DEFAULT_CPUEP_DIR = HERE / "dump_cpuep"
DEFAULT_HIPEP_DIR = HERE / "dump_hipep"
ACCURACY_SCRIPT = HERE / "accuracy_dirs.py"


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def find_runner():
    if RUNNER:
        return Path(RUNNER)
    exe = "test_onnx_runner.exe" if os.name == "nt" else "test_onnx_runner"
    build = HERE / "test_onnx_runner" / "build"
    for cand in (build / "RelWithDebInfo" / exe, build / exe):
        if cand.is_file():
            return cand
    return build / exe


def make_env(lib_dirs, extra=None):
    """Copy of the current environment with lib_dirs searched first."""
    env = os.environ.copy()
    dirs = [str(Path(d)) for d in lib_dirs if d]
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[var] = os.pathsep.join(dirs + [env.get(var, "")])
    env.update(extra or {})
    return env


def run_process(cmd, env, log_path):
    """Run cmd, echo its output and save it to log_path. Returns exit code."""
    print(" ".join(str(c) for c in cmd))
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen([str(c) for c in cmd], env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    print(f"ret = {proc.returncode}, log: {log_path}")
    return proc.returncode


def prepare_dump_dir(dump_dir):
    """Create dump_dir; refuse a non-empty one so stale files never mix in."""
    path = Path(dump_dir).resolve()
    if path.exists():
        if not path.is_dir():
            fail(f"not a directory: {path}")
        if any(path.iterdir()):
            fail(f"directory is not empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def check_inputs(model, input_list):
    if not model or not Path(model).is_file():
        fail(f"model not found: {model!r} (set MODEL or pass --model)")
    if not input_list or not Path(input_list).is_file():
        fail(f"input list not found: {input_list!r} "
             f"(set INPUT_LIST or pass --inputs)")
    if not find_runner().is_file():
        fail(f"test_onnx_runner not found: {find_runner()} "
             f"(set RUNNER or build it with test_onnx_runner/build.*)")


def runner_cmd(model, input_list, cpuep):
    cmd = [find_runner()]
    if cpuep:
        cmd.append("-n")
    cmd += ["-l", ORT_OPT_LEVEL, "-f", input_list, model]
    return cmd


def tensorproto_to_npy(src, dst):
    proto = onnx.TensorProto()
    with open(src, "rb") as f:
        proto.ParseFromString(f.read())
    with open(dst, "wb") as f:
        np.save(f, numpy_helper.to_array(proto), allow_pickle=False)


def dump_cpuep(model, input_list, dump_dir):
    """ORT node dump writes <name>.tensorproto; convert each to <name>.npy so
    the file names pair with the hip-ep dump."""
    check_inputs(model, input_list)
    out_dir = prepare_dump_dir(dump_dir)
    raw_dir = Path(tempfile.mkdtemp(prefix="dump_cpuep_raw_", dir=HERE))
    try:
        env = make_env(CPUEP_LIB_DIRS, {
            "ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA": "1",
            "ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA": "1",
            "ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION": "files",
            "ORT_DEBUG_NODE_IO_OUTPUT_DIR": str(raw_dir),
            "ORT_DEBUG_NODE_IO_DUMPING_DATA_TO_FILES_FOR_ALL_NODES_IS_OK": "1",
            "ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT": "0",
            "ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA": "0",
        })
        ret = run_process(runner_cmd(model, input_list, cpuep=True), env,
                          HERE / "lastrun_dump_cpuep.log")
        files = sorted(raw_dir.glob("*.tensorproto"))
        if not files:
            fail(f"CPU EP dump produced no files (ret={ret}). Is onnxruntime "
                 f"built with onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON?")
        for src in files:
            tensorproto_to_npy(src, out_dir / (src.stem + ".npy"))
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"dump_cpuep done: {len(files)} .npy -> {out_dir}")
    return ret


def dump_hipep(model, input_list, dump_dir):
    """hip-ep writes <name>.npy plus index.jsonl for the first inference.
    Dump ops are inserted at compile time, so a cached model compiled without
    dumps writes nothing."""
    check_inputs(model, input_list)
    if not HIPEP_LIBRARY or not Path(HIPEP_LIBRARY).is_file():
        fail(f"hip-ep library not found: {HIPEP_LIBRARY!r} "
             f"(set HIPEP_LIBRARY)")
    out_dir = prepare_dump_dir(dump_dir)
    env = make_env(HIPEP_LIB_DIRS, {
        **HIPEP_ENV,
        "MORPHIZEN_EP_LIBRARY": str(Path(HIPEP_LIBRARY)),
        "HIPDNN_EP_DUMP_DIR": str(out_dir),
    })
    ret = run_process(runner_cmd(model, input_list, cpuep=False), env,
                      HERE / "lastrun_dump_hipep.log")
    n = len(list(out_dir.glob("*.npy")))
    if n == 0:
        print("WARNING: no .npy produced. Check that hip-ep includes the "
              "tensor dump and that no stale model cache was hit.")
    else:
        print(f"dump_hipep done: {n} .npy (+ index.jsonl) -> {out_dir}")
    return ret


def accuracy(model, ref_dir, test_dir, csv_path):
    cmd = [sys.executable, ACCURACY_SCRIPT,
           "--ref", ref_dir, "--test", test_dir,
           "--model", model, "--order", "topo", "--csv", csv_path]
    print(" ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=str(HERE)).returncode


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=MODEL, help="ONNX model path")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, default_dir in (("dump_cpuep", DEFAULT_CPUEP_DIR),
                              ("dump_hipep", DEFAULT_HIPEP_DIR)):
        sp = sub.add_parser(name, parents=[common])
        sp.add_argument("--inputs", default=INPUT_LIST,
                        help="input list file (one .npy per line)")
        sp.add_argument("--dir", default=str(default_dir),
                        help="empty output directory")

    sp = sub.add_parser("accuracy", parents=[common])
    sp.add_argument("--ref", default=str(DEFAULT_CPUEP_DIR),
                    help="reference dump directory (CPU EP)")
    sp.add_argument("--test", default=str(DEFAULT_HIPEP_DIR),
                    help="dump directory under test (hip-ep)")
    sp.add_argument("--csv", default=str(HERE / "report.csv"),
                    help="full report CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "dump_cpuep":
        sys.exit(dump_cpuep(args.model, args.inputs, args.dir))
    elif args.cmd == "dump_hipep":
        sys.exit(dump_hipep(args.model, args.inputs, args.dir))
    else:
        sys.exit(accuracy(args.model, args.ref, args.test, args.csv))
