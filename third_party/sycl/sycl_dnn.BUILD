exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "snn_headers",
    hdrs = glob(["include/*"]),
    includes = ["include"],
)

filegroup(
    name = "snn_repo",
    srcs = glob(["**/*"]),
)
