// Minimal ONNX Runtime model runner for accuracy debugging.
//
// Depends only on ONNX Runtime and the C++17 standard library.
//
// Environment variables:
//   MORPHIZEN_EP_LIBRARY           plugin EP library to register (EP mode)
//   EP_KREGISTERATIONNAME          EP registration name
//                                  (default: MorphiZenExecutionProvider)
//   MORPHIZEN_EP_JSON_CONFIG       passed to the EP as provider option
//                                  "config_file"
//   XLNX_EXTERNAL_PROVIDER_OPTION  same format as -i
//   XLNX_EXTERNAL_SESSION_CONFIG   same format as -C
//   XLNX_ENABLE_DUMP=1             write onnx_input_<i>.bin,
//                                  onnx_output_<i>.bin and io.json to the cwd
//   ORT_LOG_LEVEL                  verbose|info|warning|error|fatal
//                                  (default: error)
//   DEBUG_TEST_ONNX_RUNNER=<n>     runner log verbosity (default: 0)
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::ostringstream check_os_;                                            \
      check_os_ << "CHECK failed: " #cond ": " << msg;                         \
      throw std::runtime_error(check_os_.str());                               \
    }                                                                          \
  } while (0)

static std::string env_str(const char* name, const std::string& def = "") {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : def;
}

static int env_int(const char* name, int def) {
  std::string v = env_str(name);
  return v.empty() ? def : std::stoi(v);
}

static const int g_debug_level = env_int("DEBUG_TEST_ONNX_RUNNER", 0);
#define LOG_V(n)                                                               \
  if (g_debug_level < (n)) {                                                   \
  } else                                                                       \
    std::cout

template <typename T>
static std::string vector_to_string(const std::vector<T>& v) {
  std::ostringstream ss;
  for (const auto& item : v)
    ss << item << " ";
  return ss.str();
}

static std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> result;
  std::string temp;
  for (char ch : str) {
    if (ch == delimiter) {
      if (!temp.empty()) {
        result.push_back(temp);
        temp.clear();
      }
    } else {
      temp += ch;
    }
  }
  if (!temp.empty())
    result.push_back(temp);
  return result;
}

static std::string trim(std::string s) {
  s.erase(0, s.find_first_not_of(" \t\r\n"));
  s.erase(s.find_last_not_of(" \t\r\n") + 1);
  return s;
}

static void usage() {
  std::cout
      << "usage: test_onnx_runner [options] -f <list> <onnx model>\n"
         "  -f <list>      required: file listing the .npy inputs, one per\n"
         "                 line, in model input order\n"
         "  -n             disable the EP (run on CPU EP)\n"
         "  -l <level>     graph optimization level (0, 1, 2, 99)\n"
         "  -p             enable ONNX Runtime profiling\n"
         "  -i <opts>      provider options: 'k1|v1 k2|v2'\n"
         "  -C <cfgs>      session config entries: 'k1|v1,k2|v2'\n"
         "  -c <lib>       custom op library (repeatable)\n"
         "The list must name exactly one .npy file per model input.\n";
}

static size_t get_data_type_size(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return 1;
  default:
    CHECK(false, "unsupported data type " << type);
  }
  return 0;
}

// dt is a numpy descr such as "<f4" or "|u1". Big-endian is rejected.
static ONNXTensorElementDataType npy_dtype_to_onnx(const std::string& dt) {
  CHECK(dt.size() >= 3 && dt[0] != '>', "unsupported npy dtype: " << dt);
  const std::string k = dt.substr(1);
  if (k == "f4") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (k == "f8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  if (k == "f2") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (k == "i1") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  if (k == "i2") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  if (k == "i4") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  if (k == "i8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  if (k == "u1") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  if (k == "u2") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  if (k == "u4") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  if (k == "u8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  if (k == "b1") return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  CHECK(false, "unsupported npy dtype: " << dt);
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

// Header example:
// "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 224, 224), }"
static void parse_npy_header(const std::string& header, std::string& dtype,
                             std::vector<int64_t>& shape, bool& fortran_order) {
  auto dp = header.find("'descr'");
  CHECK(dp != std::string::npos, "npy: missing 'descr'");
  auto v1 = header.find('\'', header.find(':', dp));
  auto v2 = header.find('\'', v1 + 1);
  CHECK(v1 != std::string::npos && v2 != std::string::npos,
        "npy: malformed 'descr'");
  dtype = header.substr(v1 + 1, v2 - v1 - 1);

  auto fp = header.find("'fortran_order'");
  CHECK(fp != std::string::npos, "npy: missing 'fortran_order'");
  auto fc = header.find(':', fp);
  auto ft = header.find("True", fc);
  auto ff = header.find("False", fc);
  fortran_order =
      ft != std::string::npos && (ff == std::string::npos || ft < ff);

  auto sp = header.find("'shape'");
  CHECK(sp != std::string::npos, "npy: missing 'shape'");
  auto lp = header.find('(', sp);
  auto rp = header.find(')', lp);
  CHECK(lp != std::string::npos && rp != std::string::npos,
        "npy: malformed 'shape'");
  shape.clear();
  std::string num;
  for (char c : header.substr(lp + 1, rp - lp - 1)) {
    if ((c >= '0' && c <= '9') || c == '-') {
      num += c;
    } else if (!num.empty()) {
      shape.push_back(std::stoll(num));
      num.clear();
    }
  }
  if (!num.empty())
    shape.push_back(std::stoll(num));
}

// The npy shape is the source of truth for the tensor shape: the model may
// report dynamic (-1) or stale static dims. A real mismatch is rejected by
// Session::Run with a precise error.
static std::vector<char> read_npy_file(const std::string& path,
                                       ONNXTensorElementDataType expected_type,
                                       const std::vector<int64_t>& model_shape,
                                       std::vector<int64_t>& shape) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.good(), "failed to open npy file: " << path);
  char magic[6];
  f.read(magic, 6);
  CHECK(f.good() && magic[0] == '\x93' && std::string(magic + 1, 5) == "NUMPY",
        "not a valid npy file: " << path);
  uint8_t ver[2] = {0, 0};
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t header_len = 0;
  if (ver[0] == 1) {
    uint16_t hl16 = 0;
    f.read(reinterpret_cast<char*>(&hl16), 2);
    header_len = hl16;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  std::string dtype;
  bool fortran_order = false;
  parse_npy_header(header, dtype, shape, fortran_order);
  CHECK(!fortran_order, "fortran_order npy not supported: " << path);
  auto npy_type = npy_dtype_to_onnx(dtype);
  CHECK(npy_type == expected_type, "npy dtype mismatch for "
                                       << path << ": npy=" << dtype << " (onnx="
                                       << npy_type << "), expected onnx="
                                       << expected_type);

  for (size_t d = 0; d < model_shape.size() && d < shape.size(); ++d) {
    if (model_shape[d] != -1 && model_shape[d] != shape[d]) {
      std::cout << "WARNING: npy static-dim mismatch for " << path << " axis "
                << d << ": npy=" << shape[d] << ", model=" << model_shape[d]
                << ". Using the npy shape." << std::endl;
    }
  }

  int64_t count = 1;
  for (auto d : shape)
    count *= d;
  std::vector<char> buf(static_cast<size_t>(count) *
                        get_data_type_size(expected_type));
  CHECK(f.read(buf.data(), buf.size()).good(),
        "failed to read npy payload from " << path);
  return buf;
}

static void write_file(const std::string& path, const void* data,
                       size_t size) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  CHECK(f.write(static_cast<const char*>(data), size).good(),
        "failed to write " << path);
}

static void save_input_output_json(const std::vector<std::string>& in,
                                   const std::vector<std::string>& out) {
  if (in.empty() && out.empty())
    return;
  auto join = [](const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i)
      s += (i ? ", \"" : "\"") + v[i] + "\"";
    return s;
  };
  std::ofstream("io.json") << "{\n\t\"input\": [" << join(in)
                           << "],\n\t\"output\": [" << join(out) << "]\n}\n";
}

static int log_level_to_int(const std::string& level) {
  if (level == "verbose") return ORT_LOGGING_LEVEL_VERBOSE;
  if (level == "info") return ORT_LOGGING_LEVEL_INFO;
  if (level == "warning") return ORT_LOGGING_LEVEL_WARNING;
  if (level == "fatal") return ORT_LOGGING_LEVEL_FATAL;
  return ORT_LOGGING_LEVEL_ERROR;
}

struct Options {
  bool enable_ep = true;
  bool enable_profiler = false;
  int opt_level = -1;
  std::string provider_options;
  std::string session_configs;
  std::vector<std::string> custom_ops;
  std::string input_file_list;
  std::string model;
};

// getopt-style parsing: options come first, then the model.
static Options parse_args(int argc, char* argv[]) {
  Options o;
  const std::string with_value = "licCf";
  int i = 1;
  for (; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.size() < 2 || arg[0] != '-')
      break;
    char opt = arg[1];
    std::string value;
    if (with_value.find(opt) != std::string::npos) {
      if (arg.size() > 2) {
        value = arg.substr(2);
      } else {
        CHECK(i + 1 < argc, "option -" << opt << " requires a value");
        value = argv[++i];
      }
    } else {
      CHECK(arg.size() == 2, "unknown option " << arg);
    }
    switch (opt) {
    case 'n': o.enable_ep = false; break;
    case 'p': o.enable_profiler = true; break;
    case 'l': o.opt_level = std::stoi(value); break;
    case 'i': o.provider_options = value; break;
    case 'C': o.session_configs = value; break;
    case 'c': o.custom_ops.push_back(value); break;
    case 'f': o.input_file_list = value; break;
    default: CHECK(false, "unknown option " << arg);
    }
  }
  CHECK(!o.input_file_list.empty(), "missing -f <list>");
  CHECK(i + 1 == argc, "expected exactly one <onnx model> after the options");
  o.model = argv[i];
  return o;
}

static std::vector<std::string> read_file_list(const std::string& list_path) {
  std::ifstream f(list_path);
  CHECK(f.good(), "failed to open input file list: " << list_path);
  std::vector<std::string> files;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty())
      continue;
    CHECK(fs::path(line).extension() == ".npy",
          "only .npy inputs are supported: " << line);
    files.push_back(line);
  }
  std::cout << "Loaded " << files.size() << " input files from " << list_path
            << std::endl;
  return files;
}

static int run(int argc, char* argv[]) {
  Options opt = parse_args(argc, argv);

  if (!env_str("XLNX_EXTERNAL_SESSION_CONFIG").empty())
    opt.session_configs = env_str("XLNX_EXTERNAL_SESSION_CONFIG");
  if (!env_str("XLNX_EXTERNAL_PROVIDER_OPTION").empty())
    opt.provider_options = env_str("XLNX_EXTERNAL_PROVIDER_OPTION");
  const bool enable_dump = env_int("XLNX_ENABLE_DUMP", 0) != 0;

  const std::vector<std::string> input_files =
      read_file_list(opt.input_file_list);

  const std::string log_level = env_str("ORT_LOG_LEVEL", "error");
  Ort::Env env(static_cast<OrtLoggingLevel>(log_level_to_int(log_level)),
               "test_onnx_runner");

  const std::string ep_name =
      env_str("EP_KREGISTERATIONNAME", "MorphiZenExecutionProvider");
  const std::string ep_library = env_str("MORPHIZEN_EP_LIBRARY");
  std::vector<Ort::ConstEpDevice> selected_devices;
  if (opt.enable_ep) {
    CHECK(!ep_library.empty(), "MORPHIZEN_EP_LIBRARY is not set (use -n to "
                               "run on the CPU EP)");
    fs::path library_path = fs::u8path(ep_library);
    CHECK(fs::exists(library_path),
          "EP library not found: " << library_path.string());
    std::cout << "RegisterExecutionProviderLibrary: " << ep_library
              << std::endl;
    env.RegisterExecutionProviderLibrary(ep_name.c_str(), library_path.c_str());
    for (const auto& device : env.GetEpDevices()) {
      if (device.EpName() == ep_name) {
        std::cout << "Selected EP device: " << device.EpName()
                  << " from vendor: " << device.EpVendor() << std::endl;
        selected_devices.emplace_back(device);
      }
    }
    CHECK(!selected_devices.empty(), "no devices found for EP: " << ep_name);
  }

  Ort::SessionOptions session_options;
  session_options.SetLogSeverityLevel(log_level_to_int(log_level));
  for (const auto& entry : split(opt.session_configs, ',')) {
    auto kv = split(entry, '|');
    CHECK(kv.size() == 2, "session config entry format error: " << entry);
    session_options.AddConfigEntry(kv[0].c_str(), kv[1].c_str());
    std::cout << "session config " << kv[0] << " = " << kv[1] << std::endl;
  }
  if (opt.opt_level != -1) {
    std::cout << "graph_optimization_level = " << opt.opt_level << std::endl;
    session_options.SetGraphOptimizationLevel(
        static_cast<GraphOptimizationLevel>(opt.opt_level));
  }
  for (const auto& lib : opt.custom_ops)
    session_options.RegisterCustomOpsLibrary(fs::u8path(lib).c_str());
  if (opt.enable_profiler)
    session_options.EnableProfiling(
        fs::path("profile_test_onnx_runner").c_str());

  if (opt.enable_ep) {
    std::unordered_map<std::string, std::string> ep_options;
    for (const auto& entry : split(opt.provider_options, ' ')) {
      auto kv = split(entry, '|');
      CHECK(kv.size() == 2, "provider option format error: " << entry);
      ep_options[kv[0]] = kv[1];
      std::cout << "provider option " << kv[0] << " = " << kv[1] << std::endl;
    }
    const std::string config_file = env_str("MORPHIZEN_EP_JSON_CONFIG");
    if (!config_file.empty())
      ep_options["config_file"] = config_file;
    session_options.AppendExecutionProvider_V2(env, selected_devices,
                                               ep_options);
  }

  {
    auto t0 = std::chrono::steady_clock::now();
    const fs::path model_path = fs::u8path(opt.model);
    Ort::Session session(env, model_path.c_str(), session_options);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << " ONNXRuntime session create :  "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                     .count()
              << " us" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session.GetInputCount();
    const size_t output_count = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> name_holders;
    std::vector<const char*> input_names, output_names;
    for (size_t i = 0; i < input_count; ++i) {
      name_holders.push_back(session.GetInputNameAllocated(i, allocator));
      input_names.push_back(name_holders.back().get());
    }
    for (size_t i = 0; i < output_count; ++i) {
      name_holders.push_back(session.GetOutputNameAllocated(i, allocator));
      output_names.push_back(name_holders.back().get());
    }

    CHECK(input_files.size() == input_count,
          "input file count (" << input_files.size()
                               << ") does not match model input count ("
                               << input_count << ")");

    auto cpu_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::vector<char>> input_data(input_count);
    std::vector<Ort::Value> input_tensors;
    std::vector<std::string> dumped_inputs, dumped_outputs;
    for (size_t i = 0; i < input_count; ++i) {
      auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto type = info.GetElementType();
      std::vector<int64_t> shape;
      input_data[i] =
          read_npy_file(input_files[i], type, info.GetShape(), shape);
      LOG_V(1) << "input." << i << " " << input_names[i]
               << " shape = " << vector_to_string(shape) << std::endl;
      input_tensors.push_back(Ort::Value::CreateTensor(
          cpu_info, input_data[i].data(), input_data[i].size(), shape.data(),
          shape.size(), type));

      if (enable_dump) {
        auto name = "onnx_input_" + std::to_string(i) + ".bin";
        write_file(name, input_data[i].data(), input_data[i].size());
        dumped_inputs.push_back(name);
      }
    }

    std::cout << "Running model..." << std::endl;
    t0 = std::chrono::steady_clock::now();
    auto outputs =
        session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);
    t1 = std::chrono::steady_clock::now();
    std::cout << " ONNXRuntime session run :  "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                     .count()
              << std::endl;
    std::cout << "done" << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
      CHECK(outputs[i].IsTensor(), "output " << output_names[i]
                                             << " is not a tensor");
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      LOG_V(1) << "output." << i << " " << output_names[i]
               << " shape = " << vector_to_string(info.GetShape())
               << std::endl;
      if (enable_dump) {
        auto name = "onnx_output_" + std::to_string(i) + ".bin";
        write_file(name, outputs[i].GetTensorMutableData<uint8_t>(),
                   info.GetElementCount() *
                       get_data_type_size(info.GetElementType()));
        std::cout << "output tensor: " << output_names[i]
                  << ", file name:" << name << std::endl;
        dumped_outputs.push_back(name);
      }
    }
    save_input_output_json(dumped_inputs, dumped_outputs);
  }

  if (opt.enable_ep)
    env.UnregisterExecutionProviderLibrary(ep_name.c_str());
  return 0;
}

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const Ort::Exception& e) {
    std::cerr << "ORT exception: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    usage();
  }
  return 1;
}
