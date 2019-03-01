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


cc_library(
    name = "sycl",
    hdrs = glob([
        "**/*.h",
        "**/*.hpp",
    ]) + ["@opencl_headers//:OpenCL-Headers"],
    includes = ["include"],
    deps = ["@opencl_headers//:OpenCL-Headers"],
)

# SYCL-DNN

genrule(
    name = "snn_genrule",
    srcs = ["@sycl_dnn_archive//:snn_repo"],
    outs = ["libsycldnn_static.a"],
    # Below $_ holds the last argument of the previous command,
    # the extra $ is needed for bazel shell cmd.
    # The build directory depends on TARGET_CPU as the host and sycl
    # toolchains are both building SYCL-DNN in parallel.
    # An empty archive is enough for the host.
    cmd = """
          cd external/sycl_dnn_archive &&
          mkdir -p build_`echo $(TARGET_CPU)` && cd $$_ &&
          if [[ \"$@\" =~ \"host\" ]]; then
            ar rcs libsycldnn_static.a;
          else
            rm -rf * &&
            %{SNN_EXPORTS}% cmake %{SNN_CMAKE_OPTIONS}% .. > cmake_log &&
            make sycl_dnn_static > make_log;
          fi &&
          cp -f libsycldnn_static.a `dirname ../../../$@`
    """,
)

cc_library(
    name = "sycl_dnn",
    srcs = ["libsycldnn_static.a"],
    deps = [
      "@sycl_dnn_archive//:snn_headers",
    ],
    linkstatic = 1,
)

# SYCL-BLAS

cc_library(
    name = "sycl_blas",
    hdrs = ["include/vptr/virtual_ptr.hpp"],
    includes = ["include"],
    deps = [
      "@sycl_blas_archive//:sycl_blas_headers",
    ],
)
