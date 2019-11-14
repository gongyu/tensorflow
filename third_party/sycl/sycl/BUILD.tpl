licenses(["notice"])  # Apache 2.0

load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.text"])

config_setting(
    name = "using_sycl_ccpp",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "false",
    },
)

config_setting(
    name = "using_sycl_trisycl",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "true",
    },
)

config_setting(
    name = "using_tensoropt",
    define_values = {
        "using_tensoropt": "true",
    },
)

cc_library(
    name = "sycl",
    hdrs = glob([
        "**/*.h",
        "**/*.hpp",
    ]) + ["@opencl_headers//:OpenCL-Headers"],
    includes = ["include"],
    deps = ["@opencl_headers//:OpenCL-Headers"],
)

filegroup(
    name = "sycl_runtime_fg",
    srcs = %{SYCL_RUNTIME_SRCS}%,
)

# SYCL-DNN

genrule(
    name = "snn_genrule",
    srcs = [
      ":sycl_runtime_fg",
      "@sycl_dnn_archive//:snn_repo",
      "@sycl_blas_archive//:sycl_blas_headers",
      "@opencl_headers//:OpenCL-Headers",
    ],
    outs = [
      "libsycldnn_static.a",
      "include/sycldnn/export.h",
    ],
    # Below $_ holds the last argument of the previous command,
    # the extra $ is needed for bazel shell cmd.
    # The build directory depends on TARGET_CPU as the host and sycl
    # toolchains are both building SYCL-DNN in parallel.
    # An empty archive is enough for the host.
    # $(@D) is a bazel variable substitued by the output directory
    cmd = """
          cd external/sycl_dnn_archive &&
          mkdir -p build_`echo $(TARGET_CPU)` && cd $$_ &&
          if [[ \"$(@D)\" =~ \"host\" ]]; then
            cmake %{SNN_HOST_CMAKE_OPTIONS}% .. > cmake_log
            ar rcs libsycldnn_static.a;
          else
            rm -f CMakeCache.txt &&
            %{SNN_EXPORTS}% cmake %{SNN_HOST_CMAKE_OPTIONS}% %{SNN_CMAKE_OPTIONS}% .. > cmake_log &&
            make sycl_dnn_static > make_log;
          fi &&
          cp -f libsycldnn_static.a ../../../$(@D)/ &&
          mkdir -p ../../../$(@D)/include &&
          cp -rf sycldnn ../../../$(@D)/include/
    """,
)

cc_library(
    name = "sycl_dnn",
    srcs = [":snn_genrule"],
    hdrs = ["include/sycldnn/export.h"],
    includes = ["include"],
    deps = [
      "@sycl_dnn_archive//:snn_headers",
      "@sycl_blas_archive//:sycl_blas_headers",
    ],
    linkstatic = 1,
)

# SYCL-BLAS

cc_library(
    name = "sycl_blas",
    deps = [
      "@sycl_blas_archive//:sycl_blas_headers",
    ],
)

# TensorOpt

filegroup(
    name = "topt_backend",
    srcs = %{TOPT_BACKEND_SRC}%,
)

genrule(
    name = "topt_genrule",
    srcs = [
      ":sycl_runtime_fg",
      ":topt_backend",
      "@tensoropt_archive//:topt_repo",
    ],
    outs = ["libtensoropt.so"],
    # Below $_ holds the last argument of the previous command,
    # the extra $ is needed for bazel shell cmd.
    # The build directory depends on TARGET_CPU as the host and sycl
    # toolchains are both building the project in parallel.
    # An empty archive is enough for the host.
    # TensorOpt will fail to build if no backend is provided but the rule pass
    # anyway. If no backend is provided TensorOpt is disabled and
    # libtensoropt.so won't be used
    cmd = """
          cd external/tensoropt_archive &&
          mkdir -p build_`echo $(TARGET_CPU)` && cd $$_ &&
          echo "int tensoropt_empty_so = 0;" | gcc -x c++ -shared -o libtensoropt.so - &&
          if [[ ! \"$@\" =~ \"host\" ]]; then
            find ! -name 'libtensoropt.so' -type f -exec rm -f {} + &&
            %{TOPT_EXPORTS}% cmake %{TOPT_CMAKE_OPTIONS}% .. > cmake_log &&
            make tensoropt > make_log || true;
          fi &&
          cp -f libtensoropt.so `dirname ../../../$@`
    """,
)

cc_library(
    name = "tensoropt",
    srcs = ["libtensoropt.so"],
    deps = [
      "@tensoropt_archive//:tensoropt_headers",
    ],
)
