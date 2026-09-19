# Reuse semantic package caches with `--pkgimages=no`

## Summary

This change allows a Julia process started with `--pkgimages=no` to reuse the
semantic heap stored in a package cache that was originally produced with
native package images enabled. Julia loads the existing `.ji` directly,
deliberately ignores the companion native library, and JIT-compiles code as it
is needed at runtime.

Previously, the loader rejected such a cache as soon as it saw that the cache
had clone targets. It then precompiled the package again to produce a second,
semantic-only `.ji`. That repeated inference, lowering, serialization, and disk
I/O even though the original cache already contained the same reusable Julia
heap state.

The change addresses the central behavior requested by JuliaLang/julia#51412.
It is enabled by JuliaLang/julia#61649, which moved the serialized heap into the
`.ji` and left native code in the companion shared library.

## Why `--pkgimages=no` should not mean “discard all precompilation”

`--pkgimages=no` is a native-code policy. It says that the process should not
load or produce package native code. It does not necessarily mean that the
process wants to repeat the semantic work performed during package
precompilation.

A package cache contains two conceptually different products:

1. Semantic state, including restored modules, methods, inferred code, method
   roots, backedges, and serialized heap objects.
2. Native state, including machine code, native global-variable mappings,
   function pointers, clone targets, and CPU-specific dispatch data.

Before JuliaLang/julia#61649, that distinction was difficult to enforce at load
time because the serialized heap itself lived in the native library. After
JuliaLang/julia#61649, the heap lives in the `.ji`, so a process can reuse the
first product without accepting the second.

This matters for CI and deployment workflows that restore a normal package
cache but intentionally run some processes with native pkgimages disabled. The
old behavior converted a native cache into a redundant semantic-only cache on
the first such invocation. The new behavior makes that conversion unnecessary.

## User-visible behavior

Given a valid cache pair such as:

```text
Example.ji       # cache header and serialized heap
Example.dylib    # native code on macOS
```

the old behavior under `--pkgimages=no` was:

```text
reject Example.ji because it describes a pkgimage
rerun package precompilation without native code
write a replacement or additional semantic-only Example.ji
load the newly generated cache
```

The new behavior is:

```text
validate Example.ji as semantic cache data
do not require or open Example.dylib
restore the heap directly from Example.ji
ignore all native pointers and native target metadata
JIT-compile methods when execution requires them
do not generate a replacement cache
```

The shared library may be absent, corrupt, or incompatible with the current CPU
without preventing semantic reuse. It is not consulted when
`--pkgimages=no` is active.

The behavior of `--pkgimages=yes`, `--pkgimages=existing`, standalone `.ji`
caches, and `--compiled-modules=no` is unchanged.

## Implementation

### Loader policy

[`base/loading.jl`](base/loading.jl) now derives `ignore_native` from
`JLOptions().use_pkgimages == 0`.

When a selected cache was produced as a package image:

- normal pkgimage loading still passes the shared-library path to
  `jl_restore_package_image_from_file`;
- `--pkgimages=no` passes the `.ji` path and requests native-code-free
  restoration.

The stale-cache checks retain all semantic validation, including:

- cache format and Julia version compatibility;
- syntax version;
- package and dependency build IDs;
- source-file identity, size, timestamps, and content hashes;
- package preferences;
- the `.ji` file checksum;
- compatibility with already loaded dependencies.

Only checks whose result can affect native code are skipped:

- clone-target compatibility;
- existence of the companion shared library;
- the shared library's checksum.

Those checks cannot protect anything used by this load path, because the native
artifact is never opened.

### Cache-flag policy

[`src/staticdata_utils.c`](src/staticdata_utils.c) allows a native-produced cache
to match an explicitly requested configuration that does not use native package
images. Only the package-image bit is ignored. Optimization, debug,
bounds-checking, and inlining flags retain their existing exact or compatible
matching rules.

The matcher bases this policy on its `requested_flags` argument rather than the
current process's global options. This is necessary because precompilation can
check several explicit `CacheFlags` configurations from one process. Clone-target
compatibility, object-cache existence, and object-cache checksums are likewise
validated according to the requested configuration rather than the parent
process. They remain enforced whenever Julia will actually load native code.

### Direct `.ji` restoration

[`src/staticdata.c`](src/staticdata.c) adds a native-free restoration path for
package images.

The path:

1. Opens the `.ji` without loading the companion library.
2. Validates the image header and obtains the heap boundaries and checksum.
3. Restores uncompressed input directly from the file stream into permanent
   image storage, avoiding a temporary full-file copy.
4. For compressed input only, reads the file into memory, decompresses the heap
   into page-backed storage, and restores from that buffer.
5. Uses an empty `jl_image_t`, so no native function pointers, global-variable
   mappings, or CPU dispatch data are installed.

If `--permalloc-pkgimg=yes` requests a permanent copy of a decompressed image,
the temporary page-backed decompression buffer is released after restoration.
Otherwise that buffer remains the restored heap's backing storage. Thus the
restored objects never outlive their underlying memory.

A corrupt compressed payload is converted to an exception result after its
temporary buffers are released, matching the loader's other recoverable cache
validation failures. The loader can reject that cache and recompile instead of
turning cache corruption into a hard `require` failure.

Compression is now an explicit part of the `.ji` format. The writer records
`JI_FLAG_COMPRESSED_ZSTD` whenever the heap payload is compressed, and the
native-free reader selects decompression from that flag rather than inferring a
codec from payload bytes. `JI_FORMAT_VERSION` is bumped from 15 to 16 so a
reader cannot silently apply the new interpretation to an older cache.

### Native-code tainting

Loading one package without its native code has implications for later package
images. A downstream native image may contain a direct call edge into native
code belonging to the earlier package. It would be unsafe to load that
downstream code after the earlier package's native functions were omitted.

The existing `ignore_native` design handles this conservatively with
`IMAGE_NATIVE_CODE_TAINTED`. The direct `.ji` path preserves that rule:

- the process is marked native-code-tainted before the heap is restored;
- native function and global mappings are absent for this image;
- subsequent image restoration also ignores native code.

This is intentionally process-wide. A more selective dependency-aware native
linkage model could relax it in the future, but doing so is not required to
separate semantic cache reuse safely.

## Why this is correct

The correctness argument rests on four boundaries.

### 1. Semantic invalidation is unchanged

The loader still rejects a `.ji` when its serialized Julia state is stale or
incompatible. Source changes, dependency changes, preference changes, build-ID
changes, syntax changes, cache corruption, and incompatible Julia versions all
take the same rejection paths as before.

This change does not make a stale semantic cache loadable. It removes only
rejections caused by native state that the process will not use.

### 2. Native state is neither trusted nor referenced

The native shared library is not opened. CPU clone selection is not performed.
The `JL_IMAGE_KIND_JI` buffer does not provide native image pointers, and the
process-wide taint rule prevents downstream images from reintroducing unsafe
native call edges.

Consequently, skipping native target and native artifact checks is safe: there
is no execution path from this load operation into the skipped artifact.

### 3. Heap restoration uses the existing deserializer

The change does not introduce a second serialization format or a new semantic
deserializer. Once the `.ji` is available as an uncompressed image buffer, it
uses the same header validation, dependency verification, relocation, method
activation, backedge insertion, root copying, and module registration as the
established package-image loader.

This keeps the new behavior inside existing, well-tested restoration
invariants.

### 4. Cache-flag relaxation is narrowly scoped

Only the package-image bit is ignored when the explicitly requested cache
configuration does not use native package images. All other cache flags retain
their previous exact or compatible comparisons. Native-consuming configurations
retain their previous flag rules and clone-target validation.

## Why this is the right approach

### It uses the architectural boundary created by JuliaLang/julia#61649

The serialized heap is now a real file-level artifact separate from machine
code. Loading that file directly makes the implementation match the architecture
instead of continuing to treat the `.ji` and shared library as inseparable.

### It avoids `dlopen` entirely

An intermediate implementation could open the shared library only to call its
heap-unpacking hook and then discard its native pointers. That reuses semantic
state, but it still requires the native artifact to exist and be loadable. It
also retains CPU/ABI, dynamic-loader, code-signing, and deployment constraints
for an artifact the user explicitly disabled.

Direct `.ji` restoration removes those constraints and proves that semantic
reuse is independent of the native artifact.

### It reuses the cache instead of rewriting it

Producing a semantic-only copy would preserve the old precompilation and I/O
costs, consume additional depot space, complicate cache eviction, and create two
files representing the same semantic state. Reusing the existing `.ji` is both
simpler and cheaper.

### It makes compression an explicit format contract

Inferring compression from the heap payload would work for the current writer,
but it would leave an important storage property implicit. An explicit
`JI_FLAG_COMPRESSED_ZSTD` makes the reader/writer contract inspectable, rejects
mislabelled or corrupt data through the normal decompressor, and leaves room for
future codecs to receive distinct format values.

Package-image headers already require the exact Julia git branch and commit, and
package caches are disposable across Julia builds. Preserving caches emitted by
an older implementation is therefore not a meaningful production compatibility
goal. Bumping `JI_FORMAT_VERSION` also gives dirty-tree developer builds an
unambiguous invalidation boundary.

### It keeps normal native loading unchanged

The established shared-library path remains intact for processes that allow
pkgimages. The new path is selected only by the existing `ignore_native`
parameter under explicit `--pkgimages=no`, which limits the behavioral and
review surface.

## Tests

[`test/loading.jl`](test/loading.jl) covers the complete policy boundary:

1. It creates a package with an explicitly precompiled method.
2. It produces and directly loads an uncompressed native cache.
3. It directly exercises `_tryrequire_from_serialized` without an object-cache
   path, covering cache loads returned by the precompilation driver.
4. It then produces and loads a compressed native cache as well.
5. It corrupts a compressed payload while preserving the outer file checksum
   and verifies that decompression returns a rejectable cache error.
6. Before each load, it moves the native shared library out of the way.
7. It starts Julia with `--pkgimages=no` and loads the package.
8. It calls the package method, proving that execution succeeds through JIT
   compilation rather than a hidden native pointer.
9. It checks the loader log to verify direct native-free cache loading.
10. It verifies that no cache generation occurred.
11. It verifies byte-for-byte that each original `.ji` was not replaced.

The `CacheFlags` tests also run the matcher under `--pkgimages=no` with several
explicit requested configurations. They verify that only the package-image bit
is relaxed and that matching does not depend on the parent process's global
package-image policy. The loading tests check the same boundary for native
artifact staleness from both `--pkgimages=yes` and `--pkgimages=no` processes.

[`test/cmdlineargs.jl`](test/cmdlineargs.jl) independently checks that
`JI_FLAG_COMPRESSED_ZSTD` agrees with the actual payload for compressed and
uncompressed package caches, both with and without native output.

Local validation completed successfully:

- Julia rebuilt with `make -j`;
- `JULIA_TEST_FAILFAST=1 make test-revise-loading` passed all 151,210 checks;
- `JULIA_TEST_FAILFAST=1 make test-revise-cmdlineargs` succeeded with 577
  passing and 5 expected-broken checks;
- Clang static analysis, clang-tidy, the safety checker, and the GC-rooting
  checker passed for `staticdata.c`;
- `make fix-whitespace` and `git diff --check` passed.

## Local performance result

A local A/B benchmark used an unmodified nightly that contained
JuliaLang/julia#61649 but not this change, and compared it with the patched
build. Each trial began with a pristine depot containing native caches for
Revise and its dependency graph, then timed the first `--pkgimages=no` load.

Across five single-worker trials:

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean wall time | 24.83 s | 1.48 s | 23.35 s faster (94%) |
| Mean user CPU | 23.63 s | 1.38 s | 22.25 CPU-seconds fewer |
| New redundant `.ji` files | 26 | 0 | 26 files avoided |
| Additional cache bytes | 32.2 MB | 0 | 32.2 MB avoided |

The exact saving depends on the package graph and CI topology. Workflows that
do not mix native-cache production with `--pkgimages=no`, or that already have
a semantic-only cache, should see little or no change.

## Non-goals and follow-up work

This change makes semantic reuse work, but it does not complete every possible
cache-layout improvement. In particular, it does not yet:

- remove native-code-related inputs from cache filename generation;
- deduplicate semantic `.ji` files across compatible codegen targets;
- allow selectively re-enabling safe native images after a semantic-only image
  has tainted the process;
- measure savings across the full Julia package ecosystem or hosted CI fleets.

Those are separable follow-ups. The present change establishes the necessary
policy and load path: a valid semantic heap can be reused independently of its
native artifact, without weakening semantic cache validation.
