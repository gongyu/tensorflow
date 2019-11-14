exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensoropt_headers",
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
)

filegroup(
    name = "topt_repo",
    srcs = glob(["**/*"]),
)
