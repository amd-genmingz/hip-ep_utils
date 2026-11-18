# Build test_onnx_runner (RelWithDebInfo) into .\build.

# ---- user settings ---------------------------------------------------------
# ONNX Runtime install root, containing include\ and lib\.
$OnnxRuntimeRoot = ""
# CMake generator. "Ninja" requires a VS Developer PowerShell.
$Generator = "Visual Studio 17 2022"
# ----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

if (-not $OnnxRuntimeRoot) {
    Write-Error "set `$OnnxRuntimeRoot at the top of $($MyInvocation.MyCommand.Name)"
    exit 1
}

$SrcDir = $PSScriptRoot
$BuildDir = Join-Path $SrcDir "build"

$ConfigureArgs = @("-S", $SrcDir, "-B", $BuildDir, "-G", $Generator)
if ($Generator -like "Visual Studio*") {
    $ConfigureArgs += @("-A", "x64")
}
$ConfigureArgs += @(
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    "-DONNXRUNTIME_ROOT=$OnnxRuntimeRoot"
)

cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config RelWithDebInfo --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
