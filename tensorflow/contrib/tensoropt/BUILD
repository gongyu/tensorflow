# Description:
#   Wrap TensorOpt with tensorflow

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_cc_test",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

cc_library(
    name = "topt_api",
    hdrs = [
        "api/runtime_api.h",
        "api/topt_lib_api.h",
    ],
)

tf_custom_op_library(
    name = "python/ops/_topt_engine_op.so",
    srcs = [
        "ops/topt_engine_op.cc",
    ],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ],
)

cc_library(
    name = "topt_engine_op_kernel",
    srcs = [
        "kernels/topt_engine_op.cc",
        "kernels/convert_nodes.cc",
    ],
    hdrs = [
        "kernels/topt_engine_op.h",
        "kernels/convert_nodes.h",
        "kernels/hash_shapes.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":topt_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib_proto_parsing",
        "@local_config_sycl//sycl:tensoropt",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = [
        "topt_engine_op",
    ],
)

cc_binary(
    name = "libtopt_op_cc.so",
    srcs = [
        "//tensorflow:libtensorflow_cc.so",
    ],
    deps = [
        ":topt_engine_op_op_lib",
        ":topt_engine_op_kernel",
    ],
    linkshared = 1,
)

tf_gen_op_wrapper_py(
    name = "topt_engine_op",
    gen_locally = True,
    deps = [
        ":topt_engine_op_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "topt_engine_op_loader",
    srcs = ["python/ops/topt_engine_op.py"],
    dso = [
        ":python/ops/_topt_engine_op.so",
        "@local_config_sycl//sycl:tensoropt",
    ],
    kernels = [
        ":topt_engine_op_kernel",
        ":topt_engine_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
    ],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":topt_convert_py",
        ":topt_ops_py",
        "//tensorflow/python:errors",
    ],
)

py_library(
    name = "topt_ops_py",
    srcs_version = "PY2AND3",
    deps = [
        ":topt_engine_op",
        ":topt_engine_op_loader",
    ],
)

py_library(
    name = "topt_convert_py",
    srcs = ["python/topt_convert.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":wrap_conversion",
        "//tensorflow/python:tf_optimizer",
    ],
)

tf_py_wrap_cc(
    name = "wrap_conversion",
    srcs = ["topt_conversion.i"],
    copts = tf_copts(),
    deps = [
        ":topt_conversion",
        ":topt_engine_op_kernel",
        "//tensorflow/core:framework_lite",
        "//third_party/python_runtime:headers",
    ],
)

# Library for the node-level conversion portion of TensorOpt operation creation
cc_library(
    name = "topt_conversion",
    srcs = [
        "convert/convert_graph.cc",
        "convert/topt_optimization_pass.cc",
    ],
    hdrs = [
        "convert/convert_graph.h",
        "convert/topt_optimization_pass.h",
    ],
    deps = [
        ":segment",
        ":topt_api",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@local_config_sycl//sycl:tensoropt",
    ],
)

# Library for the segmenting portion of TensorOpt operation creation
cc_library(
    name = "segment",
    srcs = ["segment/segment.cc"],
    hdrs = [
        "segment/segment.h",
        "segment/union_find.h",
    ],
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "tensoropt_test_cc",
    size = "small",
    srcs = ["tensoropt_test.cc"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        ":topt_api",
        "@local_config_sycl//sycl:sycl",
        "@local_config_sycl//sycl:tensoropt",
    ],
)

tf_cc_test(
    name = "segment_test",
    size = "small",
    srcs = ["segment/segment_test.cc"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":segment",
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

py_test(
    name = "tf_topt_integration_test",
    srcs = ["test/tf_topt_integration_test.py"],
    main = "test/tf_topt_integration_test.py",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":init_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)
