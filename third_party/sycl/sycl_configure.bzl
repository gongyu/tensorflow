# -*- Python -*-
"""SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * TF_NEED_OPENCL_SYCL: boolean value representing whether to use SYCL.
  * COMPUTECPP_TOOLKIT_PATH: The path to the ComputeCpp toolkit.
  * TRISYCL_INCLUDE_DIR: The path to the include directory of triSYCL.
                         (if using triSYCL instead of ComputeCpp)
  * PYTHON_LIB_PATH: The path to the python lib
  * TF_SYCL_BITCODE_TARGET: The SYCL bitcode target
  * TF_SYCL_OFFLINE_COMPILER: Optional OpenCL offline compiler
  * TF_SYCL_OFFLINE_COMPILER_ARGS: Optional OpenCL offline compiler arguments
  * TF_SYCL_CROSS_TOOLCHAIN: The path to the toolchain (only if cross-compiling)
  * TF_SYCL_CROSS_TOOLCHAIN_NAME: The name of the toolchain (only if cross-compiling)
  * TF_SYCL_USE_HALF: Whether to support half type or not
  * TF_SYCL_USE_DOUBLE: Whether to support double type or not
  * TF_SYCL_USE_LOCAL_MEM: Whether to assume if the device has local memory or not
  * TF_SYCL_USE_SERIAL_MEMOP: Whether to replace memcpy intrinsics by serial operations in kernels
  * TF_SYCL_PLATFORM: Enable platform specific optimizations
  * TF_SYCL_USE_TENSOROPT: Whether to use enable TensorOpt module
  * TF_SYCL_IMGDNN_DIR: Path to IMGDNN root, set IMGDNN as the TensorOpt backend if enabled
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"
_HOST_C_COMPILER= "HOST_C_COMPILER"
_TF_NEED_OPENCL_SYCL= "TF_NEED_OPENCL_SYCL"
_COMPUTECPP_TOOLKIT_PATH = "COMPUTECPP_TOOLKIT_PATH"
_TRISYCL_INCLUDE_DIR = "TRISYCL_INCLUDE_DIR"
_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"
_TF_SYCL_BITCODE_TARGET = "TF_SYCL_BITCODE_TARGET"
_TF_SYCL_OFFLINE_COMPILER = "TF_SYCL_OFFLINE_COMPILER"
_TF_SYCL_OFFLINE_COMPILER_ARGS = "TF_SYCL_OFFLINE_COMPILER_ARGS"
_TF_SYCL_CROSS_TOOLCHAIN = "TF_SYCL_CROSS_TOOLCHAIN"
_TF_SYCL_CROSS_TOOLCHAIN_NAME = "TF_SYCL_CROSS_TOOLCHAIN_NAME"
_TF_SYCL_SYSROOT = "TF_SYCL_SYSROOT"
_TF_SYCL_USE_HALF = "TF_SYCL_USE_HALF"
_TF_SYCL_USE_DOUBLE = "TF_SYCL_USE_DOUBLE"
_TF_SYCL_USE_LOCAL_MEM = "TF_SYCL_USE_LOCAL_MEM"
_TF_SYCL_USE_SERIAL_MEMOP = "TF_SYCL_USE_SERIAL_MEMOP"
_TF_SYCL_PLATFORM = "TF_SYCL_PLATFORM"
_TF_SYCL_SNN_BUILD_DIR = "TF_SYCL_SNN_BUILD_DIR"
_TF_SYCL_TOPT_BUILD_DIR = "TF_SYCL_TOPT_BUILD_DIR"
_TF_SYCL_USE_TENSOROPT = "TF_SYCL_USE_TENSOROPT"
_TF_SYCL_IMGDNN_DIR = "TF_SYCL_IMGDNN_DIR"

_COMPUTECPP_MIN_VERSION = "1.2.0"

# Target variables are not available during the configure step as the target
# is provided in the bazel command line.
_CPU_TARGET = "armeabi"

def _optional_get_env(repository_ctx, name, default=None):
  """Return the environment variable's value if present, default otherwise"""
  if name in repository_ctx.os.environ:
    return repository_ctx.os.environ[name]
  return default

def _enable_sycl(repository_ctx):
  """Return true if SYCL is enabled"""
  if _TF_NEED_OPENCL_SYCL in repository_ctx.os.environ:
    enable_sycl = repository_ctx.os.environ[_TF_NEED_OPENCL_SYCL].strip()
    return enable_sycl == "1"
  return False

def _enable_compute_cpp(repository_ctx):
  """Return true if ComputeCpp is enabled"""
  return _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ

def _crosscompile(repository_ctx):
  """Return true if cross-compiling for generic targets"""
  return (_TF_SYCL_CROSS_TOOLCHAIN in repository_ctx.os.environ and
         _TF_SYCL_CROSS_TOOLCHAIN_NAME in repository_ctx.os.environ)


def _auto_configure_fail(msg):
  """Output failure message when auto configuration fails"""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def _find_c(repository_ctx):
  """Find host C compiler"""
  c_name = "gcc"
  if _HOST_C_COMPILER in repository_ctx.os.environ:
    c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
  if c_name.startswith("/"):
    return c_name
  c = repository_ctx.which(c_name)
  if c == None:
    fail("Cannot find C compiler, please correct your path.")
  return c

def _find_cc(repository_ctx):
  """Find host C++ compiler"""
  cc_name = "g++"
  if _HOST_CXX_COMPILER in repository_ctx.os.environ:
    cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
  if cc_name.startswith("/"):
    return cc_name
  cc = repository_ctx.which(cc_name)
  if cc == None:
    fail("Cannot find C++ compiler, please correct your path.")
  return cc

def _to_tuple(version):
  """Converts a version with dot deparated values to a tuple of ints"""
  return tuple([int(x) for x in version.split('.')])

def _check_computecpp_version(repository_ctx, computecpp_path):
  """Check ComputeCpp version"""
  computecpp_info_cmd = "{}/bin/computecpp_info --dump-version".format(computecpp_path)
  result = repository_ctx.execute(computecpp_info_cmd.split(' '), quiet=True)
  if result.return_code != 0:
    fail("Failed to execute '{}', ".format(computecpp_info_cmd) +
         "check that the path to computecpp_info is correct and " +
         "that the file is executable on this architecture.")
  current_version = result.stdout.split(' ')[1].strip('\n')

  if _to_tuple(current_version) < _to_tuple(_COMPUTECPP_MIN_VERSION):
    fail("Found ComputeCpp version {} but expected at least {}".format(
        current_version, _COMPUTECPP_MIN_VERSION))

def _find_computecpp_root(repository_ctx):
  """Find ComputeCpp compiler"""
  computecpp_path = ""
  if _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ:
    computecpp_path = repository_ctx.os.environ[_COMPUTECPP_TOOLKIT_PATH].strip()
  if computecpp_path.startswith("/"):
    _check_computecpp_version(repository_ctx, computecpp_path)
    return computecpp_path
  fail("Cannot find SYCL compiler, please correct your path")

def _find_trisycl_include_dir(repository_ctx):
  """Find triSYCL include directory"""
  if _TRISYCL_INCLUDE_DIR in repository_ctx.os.environ:
    sycl_name = repository_ctx.os.environ[_TRISYCL_INCLUDE_DIR].strip()
    if sycl_name.startswith("/"):
      return sycl_name
  fail( "Cannot find triSYCL include directory, please correct your path")

def _find_python_lib(repository_ctx):
  """Returns python path"""
  if _PYTHON_LIB_PATH in repository_ctx.os.environ:
    return repository_ctx.os.environ[_PYTHON_LIB_PATH].strip()
  fail("Environment variable PYTHON_LIB_PATH was not specified re-run ./configure")

def _check_lib(repository_ctx, lib_path):
  """Checks if lib exists under sycl_toolkit_path or fail if it doesn't"""
  if not repository_ctx.path(lib_path).exists:
    _auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not"""
  if not repository_ctx.path(directory).exists:
    _auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dst_dir):
  """Symlinks all the files in a directory"""
  files = repository_ctx.path(src_dir).readdir()
  for src_file in files:
    repository_ctx.symlink(src_file, dst_dir + "/" + src_file.basename)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  """Instantiate a template file with substitutions

  This function is used to copy a file to TF build directory
  """
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/sycl/%s.tpl" % tpl),
      substitutions)

def _file(repository_ctx, label):
  """Copy a file to TF build directory"""
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/sycl/%s" % label),
      {})

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_sycl_disabled():
  fail("ERROR: Building with --config=sycl but TensorFlow is not configured " +
       "to build with SYCL support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with SYCL support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_sycl_disabled.bzl", "error_sycl_disabled")

error_sycl_disabled()
"""

def _create_dummy_repository(repository_ctx):
  """
  Some files need to be copied to the build directory when SYCL is disabled

  This is needed in particular for tensorflow/sycl/platform/default/build_config:sycl.
  Also add a dummy rule to intercept calls to --config=sycl when SYCL hasn't
  been configured.
  """
  _tpl(repository_ctx, "sycl:build_defs.bzl")
  _tpl(repository_ctx, "sycl:BUILD")
  _file(repository_ctx, "sycl:LICENSE.text")
  _file(repository_ctx, "sycl:include/vptr/virtual_ptr.hpp")

  repository_ctx.file("sycl/include/sycl.hpp", "")
  repository_ctx.file("sycl/lib/libComputeCpp.so", "")

  repository_ctx.file("crosstool/error_sycl_disabled.bzl",
                      _DUMMY_CROSSTOOL_BZL_FILE)
  repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _get_cross_variables(repository_ctx):
  """Return a tuple of variables used for cross-compilation"""
  gcc_toolchain_path = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN]
  gcc_toolchain_name = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN_NAME]
  sysroot = _optional_get_env(repository_ctx, _TF_SYCL_SYSROOT,
                              "{}/{}/libc".format(gcc_toolchain_path, gcc_toolchain_name))
  return (gcc_toolchain_path, gcc_toolchain_name, sysroot)

def cmake_options_to_str(cmake_options):
  """Convert CMake options as a dictionary of list to a string"""
  res = ""
  for key in cmake_options:
    if key.startswith("-"):
      res += key + " "
      continue
    values = cmake_options[key]
    if values:
      res += "-D{}=\"".format(key)
      if type(values) == type([]):
        res += " ".join(values)
      else:
        res += values
      res += "\" "
  return res

def _get_dependencies_substitutions(repository_ctx):
  """Get substitutions for the BUILD.tpl file"""

  # CMake options are maps of the form:
  #   flag_name -> string or list of strings
  snn_exports = []
  topt_exports = []
  cl_headers = "../../opencl_headers/opencl22/"
  snn_cmake_options = {
    "--no-warn-unused-cli": "",
    "OpenCL_INCLUDE_DIR": cl_headers,
    "COMPUTECPP_USER_FLAGS": [],
    "CMAKE_CXX_FLAGS_RELEASE": ["-O3"],
    "CMAKE_EXE_LINKER_FLAGS": ["-Wl,--enable-new-dtags"],
  }
  topt_cmake_options = {
    "CMAKE_BUILD_TYPE": "Release",
    "BUILD_SHARED_LIBS": "ON",
    "OpenCL_INCLUDE_DIR": cl_headers,
    "COMPUTECPP_USER_FLAGS": [],
    "CMAKE_CXX_FLAGS_RELEASE": ["-O3"],
    "CMAKE_EXE_LINKER_FLAGS": [],
  }
  use_computecpp = _enable_compute_cpp(repository_ctx)
  computecpp_root = _find_computecpp_root(repository_ctx) if use_computecpp else ""
  snn_build_dir = _optional_get_env(repository_ctx, _TF_SYCL_SNN_BUILD_DIR, "")
  topt_build_dir = _optional_get_env(repository_ctx, _TF_SYCL_TOPT_BUILD_DIR, "")
  if _crosscompile(repository_ctx):
    (gcc_toolchain_path, gcc_toolchain_name, sysroot) = _get_cross_variables(repository_ctx)
    platform_name = gcc_toolchain_name[0:gcc_toolchain_name.find('-')]
    for triple in [(snn_exports, snn_cmake_options, "SNN"), (topt_exports, topt_cmake_options, "TENSOROPT")]:
      exports = triple[0]
      cmake_options = triple[1]
      prefix = triple[2]
      exports.append("export {}_TOOLCHAIN_DIR={};".format(prefix, gcc_toolchain_path))
      exports.append("export {}_SYSROOT_DIR={};".format(prefix, sysroot))
      exports.append("export {}_TARGET_TRIPLE={};".format(prefix, gcc_toolchain_name))
      cmake_options["CMAKE_TOOLCHAIN_FILE"] = "../cmake/toolchains/gcc-generic.cmake"
      cmake_options["CMAKE_SYSTEM_PROCESSOR"] = platform_name
      if use_computecpp:
        cmake_options["ComputeCpp_HOST_DIR"] = computecpp_root

  if use_computecpp:
    # TensorOpt only needs the path to ComputeCpp but won't use the other options
    topt_cmake_options["ComputeCpp_DIR"] = computecpp_root
    snn_cmake_options["ComputeCpp_DIR"] = computecpp_root

    bitcode_target = repository_ctx.os.environ[_TF_SYCL_BITCODE_TARGET]
    snn_cmake_options["COMPUTECPP_BITCODE"] = bitcode_target

    use_serial_memop = _optional_get_env(repository_ctx, _TF_SYCL_USE_SERIAL_MEMOP)
    serial_memop = "ON" if use_serial_memop == "1" else "OFF"
    snn_cmake_options["SNN_COMPUTECPP_USE_SERIAL_MEMOP"] = serial_memop

    computecpp_user_flags = []
    platform = _optional_get_env(repository_ctx, _TF_SYCL_PLATFORM)
    if platform:
      computecpp_user_flags.append("-D{}=1".format(platform))

    offline_compiler = _optional_get_env(repository_ctx, _TF_SYCL_OFFLINE_COMPILER)
    if offline_compiler:
      computecpp_user_flags.append("--sycl-custom-tool={}".format(offline_compiler))
      offline_compiler_args = _optional_get_env(repository_ctx, _TF_SYCL_OFFLINE_COMPILER_ARGS)
      if offline_compiler_args:
        computecpp_user_flags.append("-sycl-custom-args '{}'".format(offline_compiler_args))
    if computecpp_user_flags:
      snn_cmake_options["COMPUTECPP_USER_FLAGS"].extend(computecpp_user_flags)
  else:
    snn_cmake_options["SNN_TRISYCL"] = "ON"
  snn_cmake_options["SNN_BUILD_TESTS"] = "OFF"
  snn_cmake_options["SNN_BUILD_BENCHMARKS"] = "OFF"
  snn_cmake_options["SNN_BUILD_SAMPLES"] = "OFF"
  snn_cmake_options["SNN_BUILD_DOCUMENTATION"] = "OFF"
  snn_cmake_options["SNN_CONV2D_DIRECT_STATIC_KERNELS"] = "ON"
  snn_cmake_options["SNN_DOWNLOAD_SYCLBLAS"] = "OFF"
  snn_cmake_options["SyclBLAS_DIR"] = "../../sycl_blas_external"

  use_half = "ON" if _optional_get_env(repository_ctx, _TF_SYCL_USE_HALF) != "0" else "OFF"
  snn_cmake_options["SNN_ENABLE_HALF"] = use_half
  use_double = "ON" if _optional_get_env(repository_ctx, _TF_SYCL_USE_DOUBLE) != "0" else "OFF"
  snn_cmake_options["SNN_ENABLE_DOUBLE"] = use_double

  use_local_mem = _optional_get_env(repository_ctx, _TF_SYCL_USE_LOCAL_MEM)
  local_mem = "ON" if use_local_mem == "1" else "OFF"
  no_local_mem = "ON" if use_local_mem == "0" else "OFF"
  snn_cmake_options["SNN_EIGEN_LOCAL_MEM"] = local_mem
  snn_cmake_options["SNN_EIGEN_NO_LOCAL_MEM"] = no_local_mem

  topt_backend_src = "[]"
  use_tensoropt = _optional_get_env(repository_ctx, _TF_SYCL_USE_TENSOROPT)
  imgdnn_dir = _optional_get_env(repository_ctx, _TF_SYCL_IMGDNN_DIR)
  if imgdnn_dir and use_tensoropt == "1":
    _check_dir(repository_ctx, imgdnn_dir)
    _check_lib(repository_ctx, imgdnn_dir + "/lib/libIMGDNN.so")
    _symlink_dir(repository_ctx, imgdnn_dir, "sycl/imgdnn")
    topt_cmake_options["TENSOROPT_BACKEND"] = "IMGDNN"
    topt_cmake_options["IMGDNN_DIR"] = imgdnn_dir
    topt_backend_src = "glob([\"imgdnn/**/*.h\"]) + [\"imgdnn/lib/libIMGDNN.so\"]"

  return {
    "%{SNN_BUILD_DIR}%" : snn_build_dir,
    "%{SNN_EXPORTS}%" : ' '.join(snn_exports),
    "%{SNN_CMAKE_OPTIONS}%" : cmake_options_to_str(snn_cmake_options),
    "%{TOPT_BUILD_DIR}%" : topt_build_dir,
    "%{TOPT_EXPORTS}%" : ' '.join(topt_exports),
    "%{TOPT_CMAKE_OPTIONS}%" : cmake_options_to_str(topt_cmake_options),
    "%{TOPT_BACKEND_SRC}%" : topt_backend_src,
  }

def _sycl_autoconf_impl(repository_ctx):
  """Implementation of the sycl_autoconf rule."""
  # ARM toolchain bits
  cross_compiler_path = ""
  cross_target = ""
  cpu_target = ""
  cross_sysroot = ""
  extra_flags = ""
  if _crosscompile(repository_ctx):
    (cross_compiler_path, cross_target, cross_sysroot) = _get_cross_variables(repository_ctx)
    _check_dir(repository_ctx, cross_compiler_path)
    cpu_target = _CPU_TARGET

  # SYCL toolchain bits
  if not _enable_sycl(repository_ctx):
    _create_dummy_repository(repository_ctx)
  else:
    # copy template files
    _tpl(repository_ctx, "sycl:build_defs.bzl")
    _tpl(repository_ctx, "sycl:BUILD", _get_dependencies_substitutions(repository_ctx))
    _tpl(repository_ctx, "crosstool:BUILD")
    _file(repository_ctx, "sycl:LICENSE.text")
    _file(repository_ctx, "sycl:include/vptr/virtual_ptr.hpp")

    if _enable_compute_cpp(repository_ctx):
      computecpp_root = _find_computecpp_root(repository_ctx)
      _check_dir(repository_ctx, computecpp_root)

      bitcode_target = repository_ctx.os.environ[_TF_SYCL_BITCODE_TARGET]
      _tpl(repository_ctx, "crosstool:CROSSTOOL", {
        "%{COMPUTECPP_ROOT_DIR}%" : computecpp_root,
        "%{BITCODE_FORMAT}%" : bitcode_target,
        "%{CROSS_COMPILER_PATH}%" : cross_compiler_path,
        "%{CROSS_TARGET}%" : cross_target,
        "%{CROSS_SYSROOT}%" : cross_sysroot,
        "%{CPU_TARGET}%" : cpu_target,
        "%{EXTRA_FLAGS}%" : extra_flags,
      })

      _check_lib(repository_ctx, computecpp_root + "/lib/libComputeCpp.so")
      _symlink_dir(repository_ctx, computecpp_root + "/lib", "sycl/lib")
      _symlink_dir(repository_ctx, computecpp_root + "/include", "sycl/include")
      _symlink_dir(repository_ctx, computecpp_root + "/bin", "sycl/bin")
    else:
      trisycl_include_dir = _find_trisycl_include_dir(repository_ctx)
      _check_dir(repository_ctx, trisycl_include_dir)

      _tpl(repository_ctx, "crosstool:trisycl",
      {
        "%{host_cxx_compiler}" : _find_cc(repository_ctx),
        "%{host_c_compiler}" : _find_c(repository_ctx),
        "%{trisycl_include_dir}" : trisycl_include_dir
      })

      _tpl(repository_ctx, "crosstool:CROSSTOOL",
      {
        "%{sycl_include_dir}" : trisycl_include_dir,
        "%{sycl_impl}" : "trisycl",
        "%{c++_std}" : "-std=c++1y",
        "%{python_lib_path}" : _find_python_lib(repository_ctx),
      })

      _symlink_dir(repository_ctx, trisycl_include_dir, "sycl/include")


sycl_configure = repository_rule(
  implementation = _sycl_autoconf_impl,
  local = True,
)
"""Detects and configures the SYCL toolchain.

Add the following to your WORKSPACE FILE:

```python
sycl_configure(name = "local_config_sycl")
```

Args:
  name: A unique name for this workspace rule.
"""
