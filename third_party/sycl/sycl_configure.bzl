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
_TF_SYCL_USE_HALF = "TF_SYCL_USE_HALF"
_TF_SYCL_USE_DOUBLE = "TF_SYCL_USE_DOUBLE"
_TF_SYCL_USE_LOCAL_MEM = "TF_SYCL_USE_LOCAL_MEM"
_TF_SYCL_USE_SERIAL_MEMOP = "TF_SYCL_USE_SERIAL_MEMOP"
_TF_SYCL_PLATFORM = "TF_SYCL_PLATFORM"

_COMPUTECPP_MIN_VERSION = "1.2.0"

def _optional_get_env(repository_ctx, name, default=None):
  if name in repository_ctx.os.environ:
    return repository_ctx.os.environ[name]
  return default

def _enable_sycl(repository_ctx):
  if _TF_NEED_OPENCL_SYCL in repository_ctx.os.environ:
    enable_sycl = repository_ctx.os.environ[_TF_NEED_OPENCL_SYCL].strip()
    return enable_sycl == "1"
  return False

def _enable_compute_cpp(repository_ctx):
  return _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ

def _crosscompile(repository_ctx):
  return _TF_SYCL_CROSS_TOOLCHAIN in repository_ctx.os.environ

def _auto_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def _find_c(repository_ctx):
  """Find host C compiler."""
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
  """Find host C++ compiler."""
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
  ''' Converts a version with dot deparated values to a tuple of ints '''
  return tuple([int(x) for x in version.split('.')])

def _check_computecpp_version(repository_ctx, computecpp_path):
  '''
  Checks if the version of computecpp at computecpp_path is
  more recent than _COMPUTECPP_MIN_VERSION
  '''
  computecpp_info_cmd = "{}/bin/computecpp_info --dump-version".format(computecpp_path)
  result = repository_ctx.execute(computecpp_info_cmd.split(' '), quiet=True)
  current_version = result.stdout.split(' ')[1].strip('\n')

  if _to_tuple(current_version) < _to_tuple(_COMPUTECPP_MIN_VERSION):
    fail("Found ComputeCpp version {} but expected at least {}".format(
        current_version, _COMPUTECPP_MIN_VERSION))

def _find_computecpp_root(repository_ctx):
  """Find ComputeCpp compiler."""
  computecpp_path = ""
  if _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ:
    computecpp_path = repository_ctx.os.environ[_COMPUTECPP_TOOLKIT_PATH].strip()
  if computecpp_path.startswith("/"):
    _check_computecpp_version(repository_ctx, computecpp_path)
    return computecpp_path
  fail("Cannot find SYCL compiler, please correct your path")

def _find_trisycl_include_dir(repository_ctx):
  """Find triSYCL include directory. """
  if _TRISYCL_INCLUDE_DIR in repository_ctx.os.environ:
    sycl_name = repository_ctx.os.environ[_TRISYCL_INCLUDE_DIR].strip()
    if sycl_name.startswith("/"):
      return sycl_name
  fail( "Cannot find triSYCL include directory, please correct your path")

def _find_python_lib(repository_ctx):
  """Returns python path."""
  if _PYTHON_LIB_PATH in repository_ctx.os.environ:
    return repository_ctx.os.environ[_PYTHON_LIB_PATH].strip()
  fail("Environment variable PYTHON_LIB_PATH was not specified re-run ./configure")

def _check_lib(repository_ctx, lib_path):
  """Checks if lib exists under sycl_toolkit_path or fail if it doesn't.

  Args:
    repository_ctx: The repository context.
    toolkit_path: The toolkit directory containing the libraries.
    ib: The library to look for under toolkit_path.
  """
  if not repository_ctx.path(lib_path).exists:
    _auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not.

  Args:
    repository_ctx: The repository context.
    directory: The directory to check the existence of.
  """
  if not repository_ctx.path(directory).exists:
    _auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
  """Symlinks all the files in a directory.

  Args:
    repository_ctx: The repository context.
    src_dir: The source directory.
    dest_dir: The destination directory to create the symlinks in.
  """
  files = repository_ctx.path(src_dir).readdir()
  for src_file in files:
    repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/sycl/%s.tpl" % tpl),
      substitutions)

def _file(repository_ctx, label):
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
  # Set up BUILD file for sycl/.
  _tpl(repository_ctx, "sycl:build_defs.bzl")
  _tpl(repository_ctx, "sycl:BUILD")
  _file(repository_ctx, "sycl:LICENSE.text")
  _file(repository_ctx, "sycl:include/vptr/virtual_ptr.hpp")

  # Create dummy files for the SYCL toolkit since they are still required by
  # tensorflow/sycl/platform/default/build_config:sycl.
  repository_ctx.file("sycl/include/sycl.hpp", "")
  repository_ctx.file("sycl/lib/libComputeCpp.so", "")

  # If sycl_configure is not configured to build with SYCL support, and the user
  # attempts to build with --config=sycl, add a dummy build rule to intercept
  # this and fail with an actionable error message.
  repository_ctx.file("crosstool/error_sycl_disabled.bzl",
                      _DUMMY_CROSSTOOL_BZL_FILE)
  repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _get_dependencies_substitutions(repository_ctx):
  snn_exports = []
  snn_cmake_options = ["-DOpenCL_INCLUDE_DIR=../../opencl_headers/opencl22/"]
  use_computecpp = _enable_compute_cpp(repository_ctx)
  computecpp_root = _find_computecpp_root(repository_ctx) if use_computecpp else ""
  if _crosscompile(repository_ctx):
    gcc_toolchain_path = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN]
    gcc_toolchain_name = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN_NAME]
    platform_name = gcc_toolchain_name[0:gcc_toolchain_name.find('-')]
    snn_exports.append("export SNN_TOOLCHAIN_DIR={};".format(gcc_toolchain_path))
    snn_exports.append("export SNN_SYSROOT_DIR={}/{}/libc;".format(gcc_toolchain_path, gcc_toolchain_name))
    snn_exports.append("export SNN_TARGET_TRIPLE={};".format(gcc_toolchain_name))
    snn_cmake_options.append("-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc-generic.cmake")
    snn_cmake_options.append("-DCMAKE_SYSTEM_PROCESSOR={}".format(platform_name))
    if use_computecpp:
      snn_cmake_options.append("-DComputeCpp_HOST_DIR={}".format(computecpp_root))

  if use_computecpp:
    snn_cmake_options.append("-DComputeCpp_DIR={}".format(computecpp_root))
    computecpp_user_flags = ""
    offline_compiler = _optional_get_env(repository_ctx, _TF_SYCL_OFFLINE_COMPILER)
    if offline_compiler:
      computecpp_user_flags = "{} --sycl-custom-tool={}".format(
          computecpp_user_flags, offline_compiler)
      offline_compiler_args = _optional_get_env(repository_ctx, _TF_SYCL_OFFLINE_COMPILER_ARGS)
      if offline_compiler_args:
        computecpp_user_flags = "{} -sycl-custom-args '{}'".format(computecpp_user_flags, offline_compiler_args)
    if computecpp_user_flags:
      snn_cmake_options.append("-DCOMPUTECPP_USER_FLAGS=\"{}\"".format(computecpp_user_flags))
  else:
    snn_cmake_options.append("-DSNN_TRISYCL=ON")
  snn_cmake_options.append("-DSNN_BUILD_TESTS=OFF")
  snn_cmake_options.append("-DSNN_BUILD_BENCHMARKS=OFF")
  snn_cmake_options.append("-DSNN_BUILD_SAMPLES=OFF")
  snn_cmake_options.append("-DSNN_BUILD_DOCUMENTATION=OFF")
  snn_cmake_options.append("-DSNN_CONV2D_DIRECT_STATIC_KERNELS=ON")
  snn_cmake_options.append("-DCMAKE_EXE_LINKER_FLAGS=-Wl,--enable-new-dtags")
  snn_cmake_options.append("-DCMAKE_CXX_FLAGS_RELEASE=-O3")

  bitcode_target = repository_ctx.os.environ[_TF_SYCL_BITCODE_TARGET]
  snn_cmake_options.append("-DCOMPUTECPP_BITCODE={}".format(bitcode_target))

  use_half = "ON" if _optional_get_env(repository_ctx, _TF_SYCL_USE_HALF) != "0" else "OFF"
  snn_cmake_options.append("-DSNN_ENABLE_HALF={}".format(use_half))
  use_double = "ON" if _optional_get_env(repository_ctx, _TF_SYCL_USE_DOUBLE) != "0" else "OFF"
  snn_cmake_options.append("-DSNN_ENABLE_DOUBLE={}".format(use_double))

  use_local_mem = _optional_get_env(repository_ctx, _TF_SYCL_USE_LOCAL_MEM)
  local_mem = "ON" if use_local_mem == "1" else "OFF"
  no_local_mem = "ON" if use_local_mem == "0" else "OFF"
  snn_cmake_options.append("-DSNN_EIGEN_LOCAL_MEM={}".format(local_mem))
  snn_cmake_options.append("-DSNN_EIGEN_NO_LOCAL_MEM={}".format(no_local_mem))

  use_serial_memop = _optional_get_env(repository_ctx, _TF_SYCL_USE_SERIAL_MEMOP)
  serial_memop = "ON" if use_serial_memop == "1" else "OFF"
  snn_cmake_options.append("-DSNN_COMPUTECPP_USE_SERIAL_MEMOP={}".format(serial_memop))

  platform = _optional_get_env(repository_ctx, _TF_SYCL_PLATFORM)
  if platform:
    snn_cmake_options.append("-DCMAKE_CXX_FLAGS=-D{}=1".format(platform))

  snn_cmake_options.append("-DSNN_DOWNLOAD_SYCLBLAS=OFF")
  snn_cmake_options.append("-DSyclBLAS_DIR=../../sycl_blas_external")

  return {
    "%{SNN_EXPORTS}%" : ' '.join(snn_exports),
    "%{SNN_CMAKE_OPTIONS}%" : ' '.join(snn_cmake_options)
  }

def _sycl_autoconf_impl(repository_ctx):
  """Implementation of the sycl_autoconf rule."""
  # ARM toolchain bits
  if _crosscompile(repository_ctx):
    gcc_toolchain_path = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN]
    gcc_toolchain_name = repository_ctx.os.environ[_TF_SYCL_CROSS_TOOLCHAIN_NAME]
    _check_dir(repository_ctx, gcc_toolchain_path)
  else:
    gcc_toolchain_path = ""
    gcc_toolchain_name = ""

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
      _tpl(repository_ctx, "crosstool:CROSSTOOL",
      {
        "%{CROSS_COMPILER_PATH}%" : gcc_toolchain_path,
        "%{CROSS_TARGET}%" : gcc_toolchain_name,
        "%{COMPUTECPP_ROOT_DIR}%"  : computecpp_root,
        "%{BITCODE_FORMAT}%" : bitcode_target
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
