@echo off
setlocal
rem Build test_onnx_runner (RelWithDebInfo) into .\build.

rem ---- user settings --------------------------------------------------------
rem ONNX Runtime install root, containing include\ and lib\.
set "ONNXRUNTIME_ROOT="
rem CMake generator. "Ninja" requires a VS Developer Command Prompt.
set "GENERATOR=Visual Studio 17 2022"
rem ---------------------------------------------------------------------------

if "%ONNXRUNTIME_ROOT%"=="" (
  echo ERROR: set ONNXRUNTIME_ROOT at the top of %~nx0
  exit /b 1
)

set "SRC_DIR=%~dp0"
if "%SRC_DIR:~-1%"=="\" set "SRC_DIR=%SRC_DIR:~0,-1%"
set "BUILD_DIR=%SRC_DIR%\build"

set "ARCH_ARGS="
echo %GENERATOR% | findstr /b /c:"Visual Studio" >nul && set "ARCH_ARGS=-A x64"

cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "%GENERATOR%" %ARCH_ARGS% ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config RelWithDebInfo --parallel
if errorlevel 1 exit /b 1
endlocal
