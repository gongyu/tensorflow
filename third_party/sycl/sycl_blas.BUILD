exports_files(["LICENSE.md"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sycl_blas_headers",
    hdrs = glob([
      "include/**/*.h",
      "include/**/*.hpp",
      "src/**/*.hpp",
    ]),
    includes = ["include", "src"],
)
