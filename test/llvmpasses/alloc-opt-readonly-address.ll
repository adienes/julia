; This file is a part of Julia. License is MIT: https://julialang.org/license

; RUN: opt --load-pass-plugin=libjulia-codegen%{shlibext} -passes='function(AllocOpt)' -S %s | FileCheck %s

@tag = external addrspace(10) global {}

; CHECK-LABEL: @roots_bundle(
; CHECK-NOT: @julia.gc_alloc_obj
; CHECK: call {{.*}}@jl_apply_generic_spans({{.*}}) [ "jl_roots"(ptr addrspace(10) %a, ptr addrspace(10) %b) ]
; CHECK-LABEL: }{{$}}
define void @roots_bundle(ptr addrspace(10) %f, ptr addrspace(10) %a, ptr addrspace(10) %b, ptr %va, i64 %nva) {
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls = call ptr @julia.ptls_states()
  %v = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 24, ptr addrspace(10) @tag) #0
  %v_derived = addrspacecast ptr addrspace(10) %v to ptr addrspace(11)
  store i64 2, ptr addrspace(11) %v_derived, align 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 8
  store atomic ptr addrspace(10) %a, ptr addrspace(11) %slot1 release, align 8
  %slot2 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 16
  store atomic ptr addrspace(10) %b, ptr addrspace(11) %slot2 release, align 8
  %raw = call ptr @julia.pointer_from_objref(ptr addrspace(11) %v_derived)
  %data = getelementptr inbounds i8, ptr %raw, i64 8
  %r = call nonnull ptr addrspace(10) (ptr addrspace(10), i64, ...) @jl_apply_generic_spans(ptr addrspace(10) %f, i64 2, ptr noalias nocapture noundef readonly %data, i64 2, ptr noalias nocapture noundef readonly %va, i64 %nva) [ "jl_roots"(ptr addrspace(10) %v) ]
  ret void
}

; CHECK-LABEL: @roots_preserve(
; CHECK-NOT: @julia.gc_alloc_obj
; CHECK: call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %a, ptr addrspace(10) %b)
; CHECK: call {{.*}}@jl_apply_generic_spans({{.*}}) [ "jl_roots"(ptr addrspace(10) %a, ptr addrspace(10) %b) ]
; CHECK-LABEL: }{{$}}
define void @roots_preserve(ptr addrspace(10) %f, ptr addrspace(10) %a, ptr addrspace(10) %b, ptr %va, i64 %nva) {
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls = call ptr @julia.ptls_states()
  %v = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 24, ptr addrspace(10) @tag) #0
  %v_derived = addrspacecast ptr addrspace(10) %v to ptr addrspace(11)
  store i64 2, ptr addrspace(11) %v_derived, align 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 8
  store atomic ptr addrspace(10) %a, ptr addrspace(11) %slot1 release, align 8
  %slot2 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 16
  store atomic ptr addrspace(10) %b, ptr addrspace(11) %slot2 release, align 8
  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %v)
  %raw = call ptr @julia.pointer_from_objref(ptr addrspace(11) %v_derived)
  %data = getelementptr inbounds i8, ptr %raw, i64 8
  %r = call nonnull ptr addrspace(10) (ptr addrspace(10), i64, ...) @jl_apply_generic_spans(ptr addrspace(10) %f, i64 2, ptr noalias nocapture noundef readonly %data, i64 2, ptr noalias nocapture noundef readonly %va, i64 %nva) [ "jl_roots"(ptr addrspace(10) %v) ]
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

; CHECK-LABEL: @readonly_call_without_roots(
; CHECK: call {{.*}}@julia.gc_alloc_obj
; CHECK-LABEL: }{{$}}
define void @readonly_call_without_roots(ptr addrspace(10) %a) {
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls = call ptr @julia.ptls_states()
  %v = call noalias nonnull align 8 dereferenceable(16) ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 16, ptr addrspace(10) @tag) #0
  %v_derived = addrspacecast ptr addrspace(10) %v to ptr addrspace(11)
  store i64 1, ptr addrspace(11) %v_derived, align 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 8
  store atomic ptr addrspace(10) %a, ptr addrspace(11) %slot1 release, align 8
  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %v)
  %raw = call ptr @julia.pointer_from_objref(ptr addrspace(11) %v_derived)
  call void @read_buffer(ptr nocapture readonly %raw)
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

; CHECK-LABEL: @writable_call(
; CHECK: call {{.*}}@julia.gc_alloc_obj
; CHECK-LABEL: }{{$}}
define void @writable_call(ptr addrspace(10) %f, ptr addrspace(10) %a) {
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls = call ptr @julia.ptls_states()
  %v = call noalias nonnull align 8 dereferenceable(16) ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 16, ptr addrspace(10) @tag) #0
  %v_derived = addrspacecast ptr addrspace(10) %v to ptr addrspace(11)
  store i64 1, ptr addrspace(11) %v_derived, align 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 8
  store atomic ptr addrspace(10) %a, ptr addrspace(11) %slot1 release, align 8
  %raw = call ptr @julia.pointer_from_objref(ptr addrspace(11) %v_derived)
  %data = getelementptr inbounds i8, ptr %raw, i64 8
  %r = call nonnull ptr addrspace(10) (ptr addrspace(10), i64, ...) @jl_apply_generic_spans(ptr addrspace(10) %f, i64 1, ptr nocapture noundef %data, i64 1) [ "jl_roots"(ptr addrspace(10) %v) ]
  ret void
}

; CHECK-LABEL: @store_in_other_block(
; CHECK: call {{.*}}@julia.gc_alloc_obj
; CHECK-LABEL: }{{$}}
define void @store_in_other_block(ptr addrspace(10) %f, ptr addrspace(10) %a, i1 %c) {
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls = call ptr @julia.ptls_states()
  %v = call noalias nonnull align 8 dereferenceable(16) ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 16, ptr addrspace(10) @tag) #0
  %v_derived = addrspacecast ptr addrspace(10) %v to ptr addrspace(11)
  store i64 1, ptr addrspace(11) %v_derived, align 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(11) %v_derived, i64 8
  br i1 %c, label %fill, label %call

fill:
  store atomic ptr addrspace(10) %a, ptr addrspace(11) %slot1 release, align 8
  br label %call

call:
  %raw = call ptr @julia.pointer_from_objref(ptr addrspace(11) %v_derived)
  %data = getelementptr inbounds i8, ptr %raw, i64 8
  %r = call nonnull ptr addrspace(10) (ptr addrspace(10), i64, ...) @jl_apply_generic_spans(ptr addrspace(10) %f, i64 1, ptr noalias nocapture noundef readonly %data, i64 1) [ "jl_roots"(ptr addrspace(10) %v) ]
  ret void
}

declare ptr @julia.get_pgcstack()
declare ptr @julia.ptls_states()
declare ptr @julia.pointer_from_objref(ptr addrspace(11))
declare token @llvm.julia.gc_preserve_begin(...)
declare void @llvm.julia.gc_preserve_end(token)
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) #0
declare nonnull ptr addrspace(10) @jl_apply_generic_spans(ptr addrspace(10), i64, ...)
declare void @read_buffer(ptr)

attributes #0 = { mustprogress nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: read, inaccessiblemem: readwrite) }
