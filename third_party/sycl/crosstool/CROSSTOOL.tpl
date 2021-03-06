major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "local_linux"
}

default_toolchain {
  cpu: "arm"
  toolchain_identifier: "local_linux"
}

default_toolchain {
  cpu: "%{CPU_TARGET}%"
  toolchain_identifier: "%{CROSS_TARGET}%"
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  supports_gold_linker: true
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "local"
  target_cpu: "local"
  target_system_name: "local"
  toolchain_identifier: "local_linux"

  tool_path { name: "ar" path: "/usr/bin/ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  tool_path { name: "gcc" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute" }
  tool_path { name: "g++" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  tool_path { name: "gcov" path: "/usr/bin/gcov" }
  tool_path { name: "ld" path: "/usr/bin/ld" }
  tool_path { name: "nm" path: "/usr/bin/nm" }
  tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
  tool_path { name: "objdump" path: "/usr/bin/objdump" }
  tool_path { name: "strip" path: "/usr/bin/strip" }

  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/lib"
  cxx_builtin_include_directory: "/usr/lib64"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"
  cxx_builtin_include_directory: "%{COMPUTECPP_ROOT_DIR}%"

  cxx_flag: "-std=c++11"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-target"
  cxx_flag: "%{BITCODE_FORMAT}%"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"
  cxx_flag: "-Wno-unused-variable"
  cxx_flag: "-Wno-unused-const-variable"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-no-canonical-prefixes"

  compiler_flag: "-ffunction-sections"
  compiler_flag: "-fdata-sections"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-Wall"

# ComputeCpp must be linked for the target configuration only
  linker_flag: "-lComputeCpp"
  linker_flag: "-L%{COMPUTECPP_ROOT_DIR}%/lib/"
  linker_flag: "-lstdc++"
  linker_flag: "-B/usr/bin/"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"
  linker_flag: "-Wl,--allow-shlib-undefined"

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O0"
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
    compiler_flag: "-O0"
    compiler_flag: "-fexceptions"
    compiler_flag: "-DEIGEN_EXCEPTIONS"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O2"
    compiler_flag: "-fno-exceptions"
    compiler_flag: "-DNDEBUG"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-mllvm"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-inline-threshold=10000"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-O3"
    linker_flag: "-Wl,--gc-sections"
    linker_flag: "-Wl,--strip-all"
  }

  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "%{CROSS_TARGET}%"
  abi_libc_version: "%{CROSS_TARGET}%"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "%{CROSS_TARGET}%"
  needsPic: true
  supports_gold_linker: true
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "%{CROSS_TARGET}%"
  target_cpu: "armeabi"
  target_system_name: "%{CROSS_TARGET}%"
  toolchain_identifier: "%{CROSS_TARGET}%"

  tool_path { name: "ar" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-ar" }
  tool_path { name: "compat-ld" path: "/bin/false" }
  tool_path { name: "cpp" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-cpp" }
  tool_path { name: "dwp" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-dwp" }
  tool_path { name: "gcc" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute" }
  tool_path { name: "g++" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  tool_path { name: "gcov" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-gcov" }
  tool_path { name: "ld" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-ld" }
  tool_path { name: "nm" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-nm" }
  tool_path { name: "objcopy" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-objcopy" }
  tool_path { name: "objdump" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-objdump" }
  tool_path { name: "strip" path: "%{CROSS_COMPILER_PATH}%/bin/%{CROSS_TARGET}%-strip" }

  cxx_builtin_include_directory: "%{CROSS_COMPILER_PATH}%"
  cxx_builtin_include_directory: "%{COMPUTECPP_ROOT_DIR}%"

  compiler_flag: "-target"
  compiler_flag: "%{CROSS_TARGET}%"
  compiler_flag: "--gcc-toolchain=%{CROSS_COMPILER_PATH}%"
  compiler_flag: "--sysroot=%{CROSS_SYSROOT}%"

  cxx_flag: "-std=c++11"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-target"
  cxx_flag: "%{BITCODE_FORMAT}%"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"
  cxx_flag: "-Wno-unused-variable"
  cxx_flag: "-Wno-unused-const-variable"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-no-canonical-prefixes"

  compiler_flag: "-ffunction-sections"
  compiler_flag: "-fdata-sections"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-Wall"

# ComputeCpp must be linked for the target configuration only
  linker_flag: "-lComputeCpp"
  linker_flag: "-L%{COMPUTECPP_ROOT_DIR}%/lib/"
  linker_flag: "-target"
  linker_flag: "%{CROSS_TARGET}%"
  linker_flag: "--gcc-toolchain=%{CROSS_COMPILER_PATH}%"
  linker_flag: "--sysroot=%{CROSS_SYSROOT}%"
  linker_flag: "-lstdc++"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"
  linker_flag: "-Wl,--allow-shlib-undefined"

  %{EXTRA_FLAGS}%

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O0"
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
    compiler_flag: "-O0"
    compiler_flag: "-fexceptions"
    compiler_flag: "-DEIGEN_EXCEPTIONS"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O2"
    compiler_flag: "-fno-exceptions"
    compiler_flag: "-DNDEBUG"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-mllvm"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-inline-threshold=10000"
    cxx_flag: "-Xsycl-device"
    cxx_flag: "-O3"
    linker_flag: "-Wl,--gc-sections"
    linker_flag: "-Wl,--strip-all"
  }

  linking_mode_flags { mode: DYNAMIC }
}
