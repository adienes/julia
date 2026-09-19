# [Package Images](@id pkgimages)

Julia package images provide object (native code) caches for Julia packages.
They are similar to Julia's [system image](@ref dev-sysimg) and support many of the same features.
In fact the underlying serialization format is the same, and the system image is the base image that the package images are built against.

## High-level overview

Each package-image cache consists of a `.ji` file containing the serialized heap
and a companion shared library containing native code. The heap contains global
data as well as the metadata describing the package's methods and types. The
shared library caches the final output of Julia's LLVM-based compiler.

The command line option `--pkgimages=no` turns off native object caching for
the session. A compatible `.ji` produced with package images enabled can still
be loaded; Julia ignores the companion shared library and JIT-compiles methods
as needed. A redundant semantic-only cache therefore does not need to be
generated solely because native package images are disabled.
See [`JULIA_MAX_NUM_PRECOMPILE_FILES`](@ref JULIA_MAX_NUM_PRECOMPILE_FILES) for the upper limit of variants Julia caches per default.

!!! note
    While the package images present themselves as native shared libraries, they are only an approximation thereof. You will not be able to link against them from a native program and they must be loaded from Julia.


## Linking

Since the package images contain native code, we must run a linker over them before we can use them. You can set the environment variable [`JULIA_VERBOSE_LINKING`](@ref JULIA_VERBOSE_LINKING) to `true` to make the package image linking process verbose.

Furthermore, we cannot assume that the user has a working system linker installed. Therefore, Julia ships with LLD, the LLVM linker, to provide a working out of the box experience. In `base/linking.jl`, we implement a limited interface to be able to link package images on all supported platforms.

### Quirks
Despite LLD being a multi-platform linker, it does not provide a consistent interface across platforms. Furthermore, it is meant to be used from `clang` or
another compiler driver, we therefore reimplement some of the logic from `llvm-project/clang/lib/Driver/ToolChains`. Thankfully one can use `lld -flavor` to set lld to the right platform

#### Windows
To avoid having to deal with `link.exe` we use `-flavor gnu`, effectively turning `lld` into a cross-linker from a mingw32 environment. Windows DLLs are required to contain a `_DllMainCRTStartup` function and to minimize our dependence on mingw32 libraries, we inject a stub definition ourselves.

#### MacOS
Dynamic libraries on macOS need to link against `-lSystem`. On recent macOS versions, `-lSystem` is only available for linking when Xcode is available.
To that effect we link with `-undefined dynamic_lookup`.

## [Package images optimized for multiple microarchitectures](@id pkgimgs-multi-versioning)

Similar to [multi-versioning](@ref sysimg-multi-versioning) for system images, package images support multi-versioning. This allows creating package caches that can run efficiently on different CPU architectures within the same environment.

See the [`JULIA_CPU_TARGET`](@ref JULIA_CPU_TARGET) environment variable for more information on how to set the CPU target for package images.

## Flags that impact package image creation and selection

These are the Julia command line flags that impact cache selection. Which of them
have to match depends on whether the session will load the cache's native image,
which is decided by `--pkgimages`:

- `--pkgimages=yes` requires a cache that has a native image, and compares every
  flag below.
- `--pkgimages=existing` compares every flag below against a cache that has a
  native image, and only the heap flags against one that does not.
- `--pkgimages=no` never loads a native image, so it compares only the heap flags.

Flags recorded in the heap, which have to match whenever the `.ji` is loaded:

- `--inline`: Exact match required, since inference stores the optimized IR it
  produced and inlining is applied before the results are cached.

Flags that only describe native code, which have to match only when that native
code is loaded. A session that loads the heap alone regenerates machine code with
its own settings, so these do not restrict it:

- `-g`, `--debug-info`: Exact match required since it changes code generation.
- `--check-bounds`: Exact match required since it changes code generation.
- `-O`, `--optimize`: Reject package images generated for a lower optimization level,
  but allow for higher optimization levels to be loaded.
