// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "julia_internal.h"
#include "julia_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    jl_array_t *opaque;
    arraylist_t active;
    jl_array_t *memo;
    jl_array_t *lengths;
    jl_array_t *implied;
    jl_array_t *bounds;
    int cut;
    int unsupported;
    int work;
} mctx_t;

#define BUDGET 2000
#define EXHAUSTED(ctx) ((ctx)->work > BUDGET)

enum { R_B, R_SUBS, R_LOCALS, R_CHECKS, R_SLOTS, R_MEETS, R_LIFTS, R_CLOSED, R_N };
enum { B_VAR, B_LB, B_UB, B_EQ, B_OWNER, B_DIAG, B_COV, B_INV, B_MIN, B_OFFTO, B_OFF, B_INH, B_N };
enum { REL_MEET, REL_SUB, REL_EQ };

typedef struct {
    jl_svec_t *root;
    int changed;
    int depth;
    int version;
    int met;
    int split;
    int cver;
    mctx_t *ctx;
} menv_t;

#define ENV_ARR(e, i) env_arr((e), (i))
#define ENV_W(e, i) env_arr_w((e), (i))
#define ENV_B(e) ENV_ARR(e, R_B)
#define NB(e) jl_array_nrows(ENV_B(e))
#define BND(e, i) ((jl_svec_t*)jl_array_ptr_ref(ENV_B(e), (i)))
#define LEN(a) jl_array_nrows(a)
#define AREF(a, i) jl_array_ptr_ref((a), (i))
#define ANY ((jl_value_t*)jl_any_type)
#define BOT ((jl_value_t*)jl_bottom_type)
#define TRY(x) do { if (x) return 1; } while (0)

static jl_value_t *meet_or_nothing(mctx_t *ctx, jl_value_t *A, jl_value_t *B);

static int is_ua(jl_value_t *t) JL_NOTSAFEPOINT { return jl_is_unionall(t); }
static int is_un(jl_value_t *t) JL_NOTSAFEPOINT { return jl_is_uniontype(t); }
static int istypeeq(jl_value_t *t) JL_NOTSAFEPOINT { return jl_is_typeeq(t); }
static int isegal(jl_value_t *t) JL_NOTSAFEPOINT { return jl_is_typeegal(t); }
static int is_tuple_type(jl_value_t *t) JL_NOTSAFEPOINT { return jl_is_tuple_type(t); }
static int is_type(jl_value_t *t) JL_NOTSAFEPOINT { return t != NULL && jl_is_type(t); }

static jl_value_t *va_T(jl_value_t *v) JL_NOTSAFEPOINT { return jl_unwrap_vararg(v); }
static jl_value_t *va_N(jl_value_t *v) JL_NOTSAFEPOINT { return jl_unwrap_vararg_num(v); }

static int has_vararg(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (!jl_is_tuple_type(t))
        return 0;
    size_t n = jl_nparams(t);
    return n > 0 && jl_is_vararg(jl_tparam(t, n - 1));
}

static size_t nfixed(jl_value_t *t) JL_NOTSAFEPOINT
{
    size_t n = jl_nparams(t);
    return has_vararg(t) ? n - 1 : n;
}

static jl_value_t *tuple_va(jl_value_t *t) JL_NOTSAFEPOINT
{
    return has_vararg(t) ? jl_tparam(t, jl_nparams(t) - 1) : NULL;
}

static jl_value_t *make_va(jl_value_t *T, jl_value_t *N)
{
    return (jl_value_t*)jl_wrap_vararg(T, N, 1, 1);
}

static int in_array(jl_array_t *a, jl_value_t *x) JL_NOTSAFEPOINT
{
    size_t i, n = LEN(a);
    for (i = 0; i < n; i++)
        if (AREF(a, i) == x)
            return 1;
    return 0;
}

static int isopaque(mctx_t *ctx, jl_value_t *t) JL_NOTSAFEPOINT
{
    return jl_is_typevar(t) && in_array(ctx->opaque, t);
}

static int issub(jl_value_t *a, jl_value_t *b)
{
    if (a == b)
        return 1;
    if (a == NULL || b == NULL)
        return 0;
    if (!(jl_is_type(a) || jl_is_typevar(a)) || !(jl_is_type(b) || jl_is_typevar(b)))
        return 0;
    return jl_subtype(a, b);
}

static int jeq(jl_value_t *a, jl_value_t *b)
{
    if (a == b)
        return 1;
    if (a == NULL || b == NULL)
        return 0;
    if (jl_is_datatype(a) && jl_is_datatype(b) && ((jl_datatype_t*)a)->name != ((jl_datatype_t*)b)->name)
        return 0;
    if (jl_is_type(a) && jl_is_type(b))
        return jl_types_equal(a, b);
    return jl_egal(a, b);
}

typedef struct scope_t {
    jl_tvar_t *var;
    struct scope_t *prev;
} scope_t;

static int in_scope(scope_t *s, jl_value_t *v) JL_NOTSAFEPOINT
{
    for (; s; s = s->prev)
        if ((jl_value_t*)s->var == v)
            return 1;
    return 0;
}

static void free_vars_(mctx_t *ctx, arraylist_t *acc, jl_value_t *t, scope_t *scope) JL_NOTSAFEPOINT
{
    if (jl_is_typevar(t)) {
        if (in_scope(scope, t) || isopaque(ctx, t))
            return;
        size_t i;
        for (i = 0; i < acc->len; i++)
            if (acc->items[i] == t)
                return;
        arraylist_push(acc, t);
    }
    else if (is_un(t)) {
        free_vars_(ctx, acc, ((jl_uniontype_t*)t)->a, scope);
        free_vars_(ctx, acc, ((jl_uniontype_t*)t)->b, scope);
    }
    else if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        free_vars_(ctx, acc, u->var->lb, scope);
        free_vars_(ctx, acc, u->var->ub, scope);
        scope_t s = { u->var, scope };
        free_vars_(ctx, acc, u->body, &s);
    }
    else if (jl_is_some_Type(t)) {
        free_vars_(ctx, acc, jl_some_Type_T(t), scope);
    }
    else if (jl_is_vararg(t)) {
        free_vars_(ctx, acc, va_T(t), scope);
        if (va_N(t))
            free_vars_(ctx, acc, va_N(t), scope);
    }
    else if (jl_is_datatype(t)) {
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            free_vars_(ctx, acc, jl_tparam(t, i), scope);
    }
}

static void free_vars(mctx_t *ctx, arraylist_t *acc, jl_value_t *t) JL_NOTSAFEPOINT
{
    free_vars_(ctx, acc, t, NULL);
}

static int has_unknowns(mctx_t *ctx, jl_value_t *t, scope_t *scope) JL_NOTSAFEPOINT
{
    if (jl_is_typevar(t))
        return !in_scope(scope, t) && !isopaque(ctx, t);
    if (is_un(t))
        return has_unknowns(ctx, ((jl_uniontype_t*)t)->a, scope) || has_unknowns(ctx, ((jl_uniontype_t*)t)->b, scope);
    if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        if (has_unknowns(ctx, u->var->lb, scope) || has_unknowns(ctx, u->var->ub, scope))
            return 1;
        scope_t s = { u->var, scope };
        return has_unknowns(ctx, u->body, &s);
    }
    if (jl_is_some_Type(t))
        return has_unknowns(ctx, jl_some_Type_T(t), scope);
    if (jl_is_vararg(t))
        return has_unknowns(ctx, va_T(t), scope) || (va_N(t) && has_unknowns(ctx, va_N(t), scope));
    if (jl_is_datatype(t)) {
        if (!jl_has_free_typevars(t))
            return 0;
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            if (has_unknowns(ctx, jl_tparam(t, i), scope))
                return 1;
    }
    return 0;
}

static int isclosed(mctx_t *ctx, jl_value_t *t) JL_NOTSAFEPOINT
{
    if (!jl_has_free_typevars(t))
        return 1;
    return LEN(ctx->opaque) > 0 && !has_unknowns(ctx, t, NULL);
}

static jl_value_t *inst(jl_value_t *x, jl_tvar_t *v, jl_value_t *val)
{
    if (x == (jl_value_t*)v)
        return val;
    if (!jl_has_typevar(x, v))
        return x;
    jl_value_t *r = NULL;
    jl_typeenv_t env = { v, val, NULL };
    JL_TRY {
        r = jl_instantiate_with_typeenv(x, &env, 1);
    }
    JL_CATCH {
        r = NULL;
    }
    return r;
}

static void occurrences_(jl_value_t *t, jl_tvar_t *v, int inv, int *acc) JL_NOTSAFEPOINT
{
    if (t == (jl_value_t*)v) {
        acc[inv ? 1 : 0]++;
    }
    else if (is_un(t)) {
        occurrences_(((jl_uniontype_t*)t)->a, v, inv, acc);
        occurrences_(((jl_uniontype_t*)t)->b, v, inv, acc);
    }
    else if (istypeeq(t)) {
        occurrences_(jl_typeeq_T(t), v, 1, acc);
    }
    else if (jl_is_vararg(t)) {
        occurrences_(va_T(t), v, inv, acc);
        occurrences_(va_T(t), v, inv, acc);
        if (va_N(t) == (jl_value_t*)v)
            acc[1]++;
    }
    else if (jl_is_datatype(t)) {
        int tuple = jl_is_tuple_type(t);
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            occurrences_(jl_tparam(t, i), v, inv || !tuple, acc);
    }
    else if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        occurrences_(u->var->lb, v, 1, acc);
        occurrences_(u->var->ub, v, 1, acc);
        if (u->var != v)
            occurrences_(u->body, v, inv, acc);
    }
}

static void occurrences(jl_value_t *t, jl_tvar_t *v, int *acc) JL_NOTSAFEPOINT
{
    acc[0] = acc[1] = 0;
    occurrences_(t, v, 0, acc);
}

static void occurrences_in(jl_array_t *vars, jl_value_t *t, jl_tvar_t *v, int *acc) JL_NOTSAFEPOINT
{
    occurrences(t, v, acc);
    size_t i, n = LEN(vars);
    for (i = 0; i < n; i++) {
        jl_tvar_t *w = (jl_tvar_t*)AREF(vars, i);
        if (w == v)
            continue;
        occurrences_(w->lb, v, 1, acc);
        occurrences_(w->ub, v, 1, acc);
    }
}

static int is_diagonal(jl_array_t *vars, jl_value_t *t, jl_tvar_t *v) JL_NOTSAFEPOINT
{
    int acc[2];
    occurrences_in(vars, t, v, acc);
    return acc[0] > 1 && acc[1] == 0;
}

static int is_concrete_leaf(mctx_t *ctx, jl_value_t *t)
{
    if (isopaque(ctx, t))
        return 1;
    if (!jl_is_datatype(t) || !jl_is_concrete_type(t))
        return 0;
    if (jl_subtype(t, (jl_value_t*)jl_type_type))
        return 0;
    if (jl_is_tuple_type(t)) {
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            if (!is_concrete_leaf(ctx, jl_tparam(t, i)))
                return 0;
    }
    return 1;
}

static jl_value_t *head(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (istypeeq(t))
        return (jl_value_t*)jl_typeeq_type;
    if (jl_is_datatype(t))
        return (jl_value_t*)((jl_datatype_t*)t)->name;
    return NULL;
}

static jl_datatype_t *common_head(jl_datatype_t *lo, jl_datatype_t *hi) JL_NOTSAFEPOINT
{
    jl_datatype_t *s = lo;
    while (s != jl_any_type && s->name != hi->name)
        s = s->super;
    return (s == jl_any_type && hi != jl_any_type) ? NULL : s;
}

static int isclosed(mctx_t *ctx, jl_value_t *t) JL_NOTSAFEPOINT;

static int type_range(mctx_t *ctx, jl_value_t *t, jl_value_t **lb, jl_value_t **ub) JL_NOTSAFEPOINT
{
    if (!isclosed(ctx, t))
        return 0;
    if (istypeeq(t)) {
        *lb = *ub = jl_typeeq_T(t);
        return 1;
    }
    if (!is_ua(t))
        return 0;
    jl_unionall_t *u = (jl_unionall_t*)t;
    if (!istypeeq(u->body) || jl_typeeq_T(u->body) != (jl_value_t*)u->var)
        return 0;
    *lb = u->var->lb;
    *ub = u->var->ub;
    return 1;
}

static jl_value_t *upper(jl_value_t *t) JL_NOTSAFEPOINT
{
    while (jl_is_typevar(t))
        t = ((jl_tvar_t*)t)->ub;
    return t;
}

static int nominally_disjoint(jl_value_t *a, jl_value_t *b) JL_NOTSAFEPOINT
{
    a = jl_unwrap_unionall(a);
    b = jl_unwrap_unionall(b);
    if (is_un(a))
        return nominally_disjoint(((jl_uniontype_t*)a)->a, b) && nominally_disjoint(((jl_uniontype_t*)a)->b, b);
    if (is_un(b))
        return nominally_disjoint(a, ((jl_uniontype_t*)b)->a) && nominally_disjoint(a, ((jl_uniontype_t*)b)->b);
    if (!jl_is_datatype(a) || !jl_is_datatype(b) || jl_is_tuple_type(a) || jl_is_tuple_type(b))
        return 0;
    return common_head((jl_datatype_t*)a, (jl_datatype_t*)b) == NULL &&
           common_head((jl_datatype_t*)b, (jl_datatype_t*)a) == NULL;
}

static void init_env(menv_t *e, mctx_t *ctx)
{
    e->root = jl_alloc_svec(R_N);
    e->changed = 0;
    e->depth = 0;
    e->version = 0;
    e->cver = -1;
    e->met = 0;
    e->split = 1;
    e->ctx = ctx;
}

static jl_array_t *env_arr(menv_t *e, int i) JL_NOTSAFEPOINT
{
    jl_value_t *a = jl_svecref(e->root, i);
    return a != NULL ? (jl_array_t*)a : (jl_array_t*)jl_an_empty_vec_any;
}

static jl_array_t *env_arr_w(menv_t *e, int i)
{
    jl_value_t *a = jl_svecref(e->root, i);
    if (a == NULL) {
        a = (jl_value_t*)jl_alloc_vec_any(0);
        jl_svecset(e->root, i, a);
    }
    return (jl_array_t*)a;
}

static ssize_t bidx(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    if (!jl_is_typevar(v))
        return -1;
    jl_array_t *b = ENV_B(e);
    size_t i, n = LEN(b);
    for (i = 0; i < n; i++)
        if (jl_svecref(AREF(b, i), B_VAR) == v)
            return i;
    return -1;
}

static int bound(menv_t *e, jl_value_t *t) JL_NOTSAFEPOINT
{
    return bidx(e, t) >= 0;
}

static jl_svec_t *binding(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    ssize_t i = bidx(e, v);
    return i < 0 ? NULL : BND(e, i);
}

#define BLB(s) jl_svecref((s), B_LB)
#define BUB(s) jl_svecref((s), B_UB)
#define BEQ(s) jl_svecref((s), B_EQ)

static int diag_of(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    jl_svec_t *s = binding(e, v);
    return s != NULL && jl_svecref(s, B_DIAG) == jl_true;
}

static jl_value_t *owner_of(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    jl_svec_t *s = binding(e, v);
    return s == NULL ? NULL : (jl_svecref(s, B_OWNER) == jl_nothing ? NULL : jl_svecref(s, B_OWNER));
}

static void uses_of(menv_t *e, jl_value_t *v, int *cov, int *inv) JL_NOTSAFEPOINT
{
    jl_svec_t *s = binding(e, v);
    *cov = s ? jl_unbox_long(jl_svecref(s, B_COV)) : 0;
    *inv = s ? jl_unbox_long(jl_svecref(s, B_INV)) : 0;
}

static void add_binding(menv_t *e, jl_tvar_t *v, jl_value_t *lb, jl_value_t *ub, jl_value_t *owner, int diag, int cov, int inv)
{
    jl_svec_t *s = jl_alloc_svec(B_N);
    JL_GC_PUSH1(&s);
    jl_svecset(s, B_VAR, v);
    jl_svecset(s, B_LB, lb);
    jl_svecset(s, B_UB, ub);
    jl_svecset(s, B_EQ, jl_nothing);
    jl_svecset(s, B_OWNER, owner ? owner : jl_nothing);
    jl_svecset(s, B_DIAG, diag ? jl_true : jl_false);
    jl_value_t *c = jl_box_long(cov);
    jl_svecset(s, B_COV, c);
    c = jl_box_long(inv);
    jl_svecset(s, B_INV, c);
    jl_svecset(s, B_MIN, jl_box_long(0));
    jl_svecset(s, B_OFFTO, jl_nothing);
    jl_svecset(s, B_OFF, jl_box_long(0));
    jl_svecset(s, B_INH, jl_false);
    jl_array_ptr_1d_push(ENV_W(e, R_B), (jl_value_t*)s);
    e->cver = -1;
    JL_GC_POP();
}

static void local_env(menv_t *sub, menv_t *e)
{
    sub->root = jl_alloc_svec(R_N);
    size_t i, n = NB(e);
    for (i = 0; i < n; i++) {
        jl_svec_t *s = jl_svec_copy(BND(e, i));
        jl_array_ptr_1d_push(ENV_W(sub, R_B), (jl_value_t*)s);
    }
    int k, copied[3] = { R_SUBS, R_CHECKS, R_LIFTS };
    for (k = 0; k < 3; k++) {
        if (LEN(ENV_ARR(e, copied[k])) > 0) {
            jl_array_t *a = jl_array_copy(ENV_ARR(e, copied[k]));
            jl_svecset(sub->root, copied[k], a);
        }
    }
    sub->changed = 0;
    sub->depth = e->depth + 1;
    sub->version = 0;
    sub->cver = -1;
    sub->met = 0;
    sub->split = e->split;
    sub->ctx = e->ctx;
}

static jl_value_t *resolve(menv_t *e, jl_value_t *t) JL_NOTSAFEPOINT
{
    int n = 0;
    while (n < 64) {
        jl_svec_t *s = binding(e, t);
        if (s == NULL || BEQ(s) == jl_nothing)
            break;
        t = BEQ(s);
        n++;
    }
    return t;
}

static jl_value_t *rebuild_datatype(jl_datatype_t *t, jl_value_t **ps, size_t n)
{
    jl_value_t *r = NULL;
    JL_TRY {
        if (jl_is_tuple_type(t))
            r = jl_apply_tuple_type_v(ps, n);
        else
            r = jl_apply_type((jl_value_t*)t->name->wrapper, ps, n);
    }
    JL_CATCH {
        r = NULL;
    }
    return r;
}

static jl_value_t *apply_nothrow(jl_datatype_t *t, jl_value_t **ps, size_t n)
{
    if (jl_is_tuple_type(t) || n == 0)
        return rebuild_datatype(t, ps, n);
    jl_value_t *body = t->name->wrapper;
    jl_typeenv_t *envs = (jl_typeenv_t*)alloca(n * sizeof(jl_typeenv_t));
    size_t i;
    for (i = 0; i < n; i++) {
        if (!jl_is_unionall(body))
            return rebuild_datatype(t, ps, n);
        envs[i].var = ((jl_unionall_t*)body)->var;
        envs[i].val = ps[i];
        envs[i].prev = i ? &envs[i - 1] : NULL;
        body = ((jl_unionall_t*)body)->body;
    }
    jl_value_t *r = NULL;
    JL_TRY {
        r = jl_instantiate_with_typeenv(body, &envs[n - 1], 1);
    }
    JL_CATCH {
        r = NULL;
    }
    return r;
}

JL_DLLEXPORT jl_value_t *jl_type_meet_inst(jl_value_t *x, jl_tvar_t *v, jl_value_t *val)
{
    return inst(x, v, val);
}

JL_DLLEXPORT jl_value_t *jl_type_meet_apply(jl_value_t *t, jl_svec_t *params)
{
    return apply_nothrow((jl_datatype_t*)t, jl_svec_data(params), jl_svec_len(params));
}

static jl_value_t *subst_(menv_t *e, jl_value_t *t, int inv)
{
    if (isclosed(e->ctx, t))
        return t;
    if (jl_is_typevar(t)) {
        jl_value_t *r = resolve(e, t);
        if (r == t)
            return t;
        return subst_(e, r, inv);
    }
    jl_value_t *r = NULL;
    if (is_un(t)) {
        int i, n = jl_count_union_components(t);
        jl_value_t **ms;
        JL_GC_PUSHARGS(ms, n);
        for (i = 0; i < n; i++) {
            ms[i] = subst_(e, jl_nth_union_component(t, i), inv);
            if (ms[i] != NULL && !(jl_is_type(ms[i]) || jl_is_typevar(ms[i])))
                ms[i] = NULL;
            if (ms[i] == NULL) {
                if (inv) {
                    JL_GC_POP();
                    return NULL;
                }
                ms[i] = BOT;
            }
        }
        r = jl_type_union(ms, n);
        JL_GC_POP();
        return r;
    }
    if (istypeeq(t)) {
        r = subst_(e, jl_typeeq_T(t), 1);
        if (r == NULL)
            return NULL;
        JL_GC_PUSH1(&r);
        r = (jl_value_t*)jl_wrap_Type(r);
        JL_GC_POP();
        return r;
    }
    if (jl_is_vararg(t)) {
        jl_value_t *T = NULL, *N = NULL;
        JL_GC_PUSH2(&T, &N);
        T = subst_(e, va_T(t), inv);
        if (T != NULL && va_N(t) != NULL)
            N = subst_(e, va_N(t), 1);
        if (T == NULL || (va_N(t) != NULL && N == NULL))
            r = NULL;
        else
            r = make_va(T, N);
        JL_GC_POP();
        return r;
    }
    if (jl_is_datatype(t)) {
        size_t i, n = jl_nparams(t);
        int pinv = inv || !jl_is_tuple_type(t);
        jl_value_t **ps;
        JL_GC_PUSHARGS(ps, n);
        int same = 1;
        for (i = 0; i < n; i++) {
            ps[i] = subst_(e, jl_tparam(t, i), pinv);
            if (ps[i] == NULL) {
                JL_GC_POP();
                return NULL;
            }
            same &= ps[i] == jl_tparam(t, i);
        }
        r = same ? t : apply_nothrow((jl_datatype_t*)t, ps, n);
        JL_GC_POP();
        return r;
    }
    if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        jl_value_t *lb = NULL, *ub = NULL, *body = NULL;
        jl_tvar_t *nv = NULL;
        JL_GC_PUSH4(&lb, &ub, &body, &nv);
        lb = subst_(e, u->var->lb, inv);
        if (lb != NULL)
            ub = subst_(e, u->var->ub, inv);
        if (lb == NULL || ub == NULL) {
            JL_GC_POP();
            return NULL;
        }
        if (lb == u->var->lb && ub == u->var->ub) {
            body = subst_(e, u->body, inv);
            r = body == NULL ? NULL : jl_type_unionall(u->var, body);
        }
        else {
            nv = jl_new_typevar(u->var->name, lb, ub);
            body = inst(u->body, u->var, (jl_value_t*)nv);
            if (body != NULL)
                body = subst_(e, body, inv);
            r = body == NULL ? NULL : jl_type_unionall(nv, body);
        }
        JL_GC_POP();
        return r;
    }
    return t;
}

static jl_value_t *subst(menv_t *e, jl_value_t *t)
{
    return subst_(e, t, 0);
}

static int occurs(jl_tvar_t *v, jl_value_t *t) JL_NOTSAFEPOINT
{
    return t != (jl_value_t*)v && jl_has_typevar(t, v);
}

static void touched(menv_t *e) JL_NOTSAFEPOINT
{
    e->changed = 1;
    e->version++;
}

static void push2(jl_array_t *a, jl_value_t *x, jl_value_t *y)
{
    jl_array_ptr_1d_push(a, x);
    jl_array_ptr_1d_push(a, y);
}

static void fact(menv_t *e, jl_value_t *lo, jl_value_t *hi)
{
    jl_array_t *subs = ENV_ARR(e, R_SUBS);
    size_t i, n = LEN(subs);
    for (i = 0; i < n; i += 2)
        if (jeq(AREF(subs, i), lo) && jeq(AREF(subs, i + 1), hi))
            return;
    push2(ENV_W(e, R_SUBS), lo, hi);
    e->changed = 1;
}

static int rank(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    if (in_array(ENV_ARR(e, R_LOCALS), v))
        return 0;
    return owner_of(e, v) != NULL ? 1 : 2;
}

static void shared(menv_t *e, jl_value_t *t, int *inv, int *tot) JL_NOTSAFEPOINT
{
    arraylist_t vs;
    arraylist_new(&vs, 0);
    free_vars(e->ctx, &vs, t);
    *inv = *tot = 0;
    size_t i;
    for (i = 0; i < vs.len; i++) {
        int c, iv;
        uses_of(e, (jl_value_t*)vs.items[i], &c, &iv);
        *inv += iv;
        *tot += c + iv;
    }
    arraylist_free(&vs);
}

static int bare_vars(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (is_un(t)) {
        int s = 0, i, n = jl_count_union_components(t);
        for (i = 0; i < n; i++) {
            jl_value_t *m = jl_nth_union_component(t, i);
            s += jl_is_typevar(m) ? 1 : bare_vars(m);
        }
        return s;
    }
    if (jl_is_datatype(t)) {
        int s = 0;
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            s += bare_vars(jl_tparam(t, i));
        return s;
    }
    return 0;
}

static int type_size(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (is_un(t))
        return 1 + type_size(((jl_uniontype_t*)t)->a) + type_size(((jl_uniontype_t*)t)->b);
    if (is_ua(t))
        return 1 + type_size(((jl_unionall_t*)t)->var->ub) + type_size(((jl_unionall_t*)t)->body);
    if (jl_is_datatype(t)) {
        int s = 1;
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            s += type_size(jl_tparam(t, i));
        return s;
    }
    if (jl_is_some_Type(t))
        return 1 + type_size(jl_some_Type_T(t));
    if (jl_is_vararg(t))
        return 1 + type_size(va_T(t));
    return 1;
}

static int cmp_int(long a, long b) JL_NOTSAFEPOINT
{
    return a < b ? -1 : a > b ? 1 : 0;
}

static int cmp_sym(jl_sym_t *a, jl_sym_t *b) JL_NOTSAFEPOINT
{
    int c = strcmp(jl_symbol_name(a), jl_symbol_name(b));
    return c < 0 ? -1 : c > 0 ? 1 : 0;
}

static void module_path(jl_module_t *m, ios_t *s) JL_NOTSAFEPOINT
{
    int top = m == jl_main_module || m == jl_base_module || m == jl_core_module;
    if (!top && m->parent != m && m->parent != NULL) {
        module_path(m->parent, s);
        ios_putc('.', s);
    }
    ios_puts(jl_symbol_name(m->name), s);
}

static int cmp_module(jl_module_t *a, jl_module_t *b) JL_NOTSAFEPOINT
{
    if (a == b)
        return 0;
    ios_t sa, sb;
    ios_mem(&sa, 0);
    ios_mem(&sb, 0);
    module_path(a, &sa);
    module_path(b, &sb);
    ios_putc(0, &sa);
    ios_putc(0, &sb);
    int c = strcmp(sa.buf, sb.buf);
    ios_close(&sa);
    ios_close(&sb);
    return c < 0 ? -1 : c > 0 ? 1 : 0;
}

static int order_rank(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (jl_is_typevar(t))
        return 0;
    if (jl_is_datatype(t))
        return 1;
    if (is_un(t))
        return 2;
    if (is_ua(t))
        return 3;
    if (jl_is_vararg(t))
        return 4;
    return 5;
}

static int type_order(jl_value_t *x, jl_value_t *y)
{
    if (x == y)
        return 0;
    int rx = order_rank(x), ry = order_rank(y);
    if (rx != ry)
        return cmp_int(rx, ry);
    int c;
    if (jl_is_typevar(x)) {
        jl_tvar_t *a = (jl_tvar_t*)x, *b = (jl_tvar_t*)y;
        c = cmp_sym(a->name, b->name);
        if (c != 0)
            return c;
        c = type_order(a->lb, b->lb);
        return c != 0 ? c : type_order(a->ub, b->ub);
    }
    if (jl_is_datatype(x)) {
        jl_datatype_t *a = (jl_datatype_t*)x, *b = (jl_datatype_t*)y;
        c = cmp_module(a->name->module, b->name->module);
        if (c != 0)
            return c;
        c = cmp_sym(a->name->name, b->name->name);
        if (c != 0)
            return c;
        size_t na = jl_nparams(a), nb = jl_nparams(b);
        c = cmp_int(na, nb);
        if (c != 0)
            return c;
        size_t i;
        for (i = 0; i < na; i++) {
            c = type_order(jl_tparam(a, i), jl_tparam(b, i));
            if (c != 0)
                return c;
        }
        return 0;
    }
    if (is_un(x)) {
        c = type_order(((jl_uniontype_t*)x)->a, ((jl_uniontype_t*)y)->a);
        return c != 0 ? c : type_order(((jl_uniontype_t*)x)->b, ((jl_uniontype_t*)y)->b);
    }
    if (is_ua(x)) {
        c = type_order((jl_value_t*)((jl_unionall_t*)x)->var, (jl_value_t*)((jl_unionall_t*)y)->var);
        return c != 0 ? c : type_order(((jl_unionall_t*)x)->body, ((jl_unionall_t*)y)->body);
    }
    if (jl_is_vararg(x)) {
        c = type_order(va_T(x), va_T(y));
        if (c != 0)
            return c;
        jl_value_t *nx = va_N(x), *ny = va_N(y);
        return type_order(nx ? nx : jl_box_long(-1), ny ? ny : jl_box_long(-1));
    }
    if (jeq(x, y))
        return 0;
    if (jl_is_long(x) && jl_is_long(y)) {
        char bx[32], by[32];
        snprintf(bx, sizeof(bx), "%ld", (long)jl_unbox_long(x));
        snprintf(by, sizeof(by), "%ld", (long)jl_unbox_long(y));
        int c = strcmp(bx, by);
        return c < 0 ? -1 : c > 0 ? 1 : 0;
    }
    if (jl_is_symbol(x) && jl_is_symbol(y))
        return cmp_sym((jl_sym_t*)x, (jl_sym_t*)y);
    return cmp_int((long)jl_object_id(x), (long)jl_object_id(y));
}

static jl_value_t *canonical(menv_t *e, jl_value_t *x, jl_value_t *y)
{
    if (isclosed(e->ctx, x))
        return x;
    if (isclosed(e->ctx, y))
        return y;
    int ix, ox, iy, oy;
    shared(e, x, &ix, &ox);
    shared(e, y, &iy, &oy);
    int kx[4] = { bare_vars(x), -ix, -ox, type_size(x) };
    int ky[4] = { bare_vars(y), -iy, -oy, type_size(y) };
    int i;
    for (i = 0; i < 4; i++)
        if (kx[i] != ky[i])
            return kx[i] < ky[i] ? x : y;
    return type_order(x, y) <= 0 ? x : y;
}

static void add_param_bounds(menv_t *e, jl_value_t *t);

static int at_least(menv_t *e, jl_value_t *v, ssize_t n);

static int in_union(jl_value_t *u, jl_value_t *x) JL_NOTSAFEPOINT
{
    int i, n = jl_count_union_components(u);
    for (i = 0; i < n; i++)
        if (jl_nth_union_component(u, i) == x)
            return 1;
    return 0;
}

static int bind_eq(menv_t *e, jl_value_t *v, jl_value_t *t)
{
    t = subst(e, t);
    if (t == NULL)
        return 1;
    if (is_un(t) && in_union(t, v)) {
        JL_GC_PUSH1(&t);
        int i, n = jl_count_union_components(t), k = 0;
        jl_value_t **ms;
        JL_GC_PUSHARGS(ms, n);
        for (i = 0; i < n; i++) {
            jl_value_t *m = jl_nth_union_component(t, i);
            if (m != v)
                ms[k++] = m;
        }
        jl_value_t *rest = jl_type_union(ms, k);
        ms[0] = rest;
        fact(e, rest, v);
        JL_GC_POP();
        JL_GC_POP();
        return 0;
    }
    if (occurs((jl_tvar_t*)v, t))
        return 1;
    JL_GC_PUSH1(&t);
    jl_svec_t *s = binding(e, v);
    ssize_t mn = jl_unbox_long(jl_svecref(s, B_MIN));
    if (mn > 0 && (jl_is_long(t) || jl_is_typevar(t)) && at_least(e, t, mn)) {
        JL_GC_POP();
        return 1;
    }
    jl_svecset(s, B_EQ, t);
    touched(e);
    fact(e, BLB(s), t);
    fact(e, t, BUB(s));
    add_param_bounds(e, t);
    JL_GC_POP();
    return 0;
}

static jl_value_t *closedize(menv_t *e, jl_value_t *t, int upper, int depth);

static jl_value_t *meet_closed(mctx_t *ctx, jl_value_t *a, jl_value_t *b)
{
    if (a == NULL || a == ANY)
        return b == NULL ? ANY : b;
    if (b == NULL || b == ANY || a == b)
        return a;
    jl_value_t *r = meet_or_nothing(ctx, a, b);
    return r == NULL ? a : r;
}

static int implied_by(mctx_t *ctx, jl_value_t *ub, jl_value_t *cub, jl_value_t *t)
{
    if (cub == NULL)
        return 0;
    if (ctx->implied == NULL)
        ctx->implied = jl_alloc_vec_any(0);
    size_t i, n = LEN(ctx->implied);
    for (i = 0; i < n; i += 4) {
        if (AREF(ctx->implied, i) == ub && AREF(ctx->implied, i + 1) == t &&
            jl_types_struct_equiv(AREF(ctx->implied, i + 2), cub))
            return AREF(ctx->implied, i + 3) == jl_true;
    }
    int r = issub(cub, t);
    if (n < 64) {
        jl_array_ptr_1d_push(ctx->implied, ub);
        jl_array_ptr_1d_push(ctx->implied, t);
        jl_array_ptr_1d_push(ctx->implied, cub);
        jl_array_ptr_1d_push(ctx->implied, r ? jl_true : jl_false);
    }
    return r;
}

static int set_ub(menv_t *e, jl_value_t *v, jl_value_t *t)
{
    jl_svec_t *s = binding(e, v);
    jl_value_t *ub = BUB(s);
    jl_value_t *nu = NULL;
    JL_GC_PUSH1(&nu);
    if (isclosed(e->ctx, ub)) {
        nu = meet_closed(e->ctx, ub, t);
    }
    else {
        nu = closedize(e, ub, 1, 0);
        nu = implied_by(e->ctx, ub, nu, t) ? BUB(s) : (t == NULL ? ANY : t);
    }
    if (nu == BOT && jl_svecref(s, B_INH) == jl_true) {
        JL_GC_POP();
        return 1;
    }
    if (!jeq(nu, BUB(s))) {
        jl_svecset(s, B_UB, nu);
        touched(e);
    }
    JL_GC_POP();
    return 0;
}

static void set_lb(menv_t *e, jl_value_t *v, jl_value_t *t)
{
    if (t == NULL)
        t = BOT;
    jl_svec_t *s = binding(e, v);
    jl_value_t *nl = NULL;
    JL_GC_PUSH1(&nl);
    if (BLB(s) == BOT) {
        nl = t;
    }
    else {
        jl_value_t *ts[2] = { BLB(s), t };
        nl = jl_type_union(ts, 2);
    }
    if (!jeq(nl, BLB(s))) {
        jl_svecset(s, B_LB, nl);
        touched(e);
    }
    JL_GC_POP();
}

static jl_value_t *rename_unionall(jl_value_t *u)
{
    if (!is_ua(u))
        return u;
    jl_unionall_t *ua = (jl_unionall_t*)u;
    jl_value_t *body = NULL;
    jl_tvar_t *nv = NULL;
    JL_GC_PUSH2(&body, &nv);
    body = rename_unionall(ua->body);
    body = jl_type_unionall(ua->var, body);
    nv = jl_new_typevar(ua->var->name, ua->var->lb, ua->var->ub);
    if (is_ua(body) && ((jl_unionall_t*)body)->var == ua->var)
        body = jl_instantiate_unionall((jl_unionall_t*)body, (jl_value_t*)nv);
    body = jl_type_unionall(nv, body);
    JL_GC_POP();
    return body;
}

static jl_value_t *lift(menv_t *e, jl_value_t *u)
{
    u = rename_unionall(u);
    JL_GC_PUSH1(&u);
    jl_tvar_t *v = ((jl_unionall_t*)u)->var;
    int acc[2];
    occurrences(((jl_unionall_t*)u)->body, v, acc);
    add_binding(e, v, v->lb, v->ub, NULL, acc[0] > 1 && acc[1] == 0, 0, 0);
    if (v->lb != BOT)
        fact(e, v->lb, (jl_value_t*)v);
    if (v->ub != ANY)
        fact(e, (jl_value_t*)v, v->ub);
    jl_value_t *body = ((jl_unionall_t*)u)->body;
    JL_GC_POP();
    return body;
}

static jl_value_t *lifted(menv_t *e, int rel, jl_value_t *t, jl_value_t *u)
{
    jl_array_t *lifts = ENV_ARR(e, R_LIFTS);
    size_t i, n = LEN(lifts);
    for (i = 0; i < n; i += 4) {
        if (jl_unbox_long(AREF(lifts, i)) == rel && jeq(AREF(lifts, i + 1), t) && jeq(AREF(lifts, i + 2), u))
            return AREF(lifts, i + 3);
    }
    jl_value_t *body = lift(e, u);
    JL_GC_PUSH1(&body);
    jl_value_t *r = jl_box_long(rel);
    jl_array_ptr_1d_push(ENV_W(e, R_LIFTS), r);
    push2(ENV_W(e, R_LIFTS), t, u);
    jl_array_ptr_1d_push(ENV_W(e, R_LIFTS), body);
    JL_GC_POP();
    return body;
}

static void add_param_bounds(menv_t *e, jl_value_t *t)
{
    if (!jl_is_datatype(t) || isclosed(e->ctx, t))
        return;
    size_t i, n = jl_nparams(t);
    if (!jl_is_tuple_type(t)) {
        jl_datatype_t *w = (jl_datatype_t*)jl_unwrap_unionall(((jl_datatype_t*)t)->name->wrapper);
        size_t nq = jl_nparams(w);
        for (i = 0; i < n && i < nq; i++) {
            jl_value_t *q = jl_tparam(w, i);
            if (!jl_is_typevar(q))
                continue;
            jl_tvar_t *qv = (jl_tvar_t*)q;
            int ubc = isclosed(e->ctx, qv->ub), lbc = isclosed(e->ctx, qv->lb);
            if ((ubc && qv->ub != ANY) || (lbc && qv->lb != BOT)) {
                jl_array_t *checks = ENV_W(e, R_CHECKS);
                jl_array_ptr_1d_push(checks, lbc ? qv->lb : BOT);
                jl_array_ptr_1d_push(checks, jl_tparam(t, i));
                jl_array_ptr_1d_push(checks, ubc ? qv->ub : ANY);
            }
        }
    }
    for (i = 0; i < n; i++)
        add_param_bounds(e, jl_tparam(t, i));
}

static jl_value_t *rewrap_closed(menv_t *e, jl_value_t *t, int depth)
{
    jl_value_t *r = subst(e, t);
    if (r == NULL)
        return BOT;
    jl_value_t *lb = NULL, *ub = NULL;
    jl_tvar_t *nv = NULL;
    JL_GC_PUSH4(&r, &lb, &ub, &nv);
    arraylist_t vs;
    arraylist_new(&vs, 0);
    free_vars(e->ctx, &vs, r);
    size_t i;
    for (i = 0; i < vs.len; i++) {
        jl_tvar_t *w = (jl_tvar_t*)vs.items[i];
        int acc[2];
        occurrences(r, w, acc);
        if (acc[0] > 1 && acc[1] == 0 && !diag_of(e, (jl_value_t*)w)) {
            r = NULL;
            break;
        }
        int b = bound(e, (jl_value_t*)w);
        lb = closedize(e, b ? (jl_value_t*)w : w->lb, 0, depth + 1);
        ub = closedize(e, b ? (jl_value_t*)w : w->ub, 1, depth + 1);
        if (lb == NULL)
            lb = BOT;
        if (ub == NULL)
            ub = ANY;
        if (!issub(lb, ub)) {
            r = BOT;
            break;
        }
        nv = jl_new_typevar(w->name, lb, ub);
        jl_value_t *body = inst(r, w, (jl_value_t*)nv);
        if (body == NULL) {
            r = BOT;
            break;
        }
        r = body;
        r = jl_type_unionall(nv, r);
    }
    arraylist_free(&vs);
    JL_GC_POP();
    return r;
}

static jl_value_t *closedize_(menv_t *e, jl_value_t *t, int upper, int depth);

static jl_value_t *closedize(menv_t *e, jl_value_t *t, int upper, int depth)
{
    if (depth != 0 || t == NULL || e->version < 0)
        return closedize_(e, t, upper, depth);
    jl_array_t *c = ENV_ARR(e, R_CLOSED);
    if (e->cver != e->version) {
        if (LEN(c) > 0)
            jl_array_del_end(c, LEN(c));
        e->cver = e->version;
    }
    jl_value_t *flag = upper ? jl_true : jl_false;
    size_t i, n = LEN(c);
    for (i = 0; i < n; i += 3)
        if (AREF(c, i) == t && AREF(c, i + 1) == flag)
            return AREF(c, i + 2) == jl_nothing ? NULL : AREF(c, i + 2);
    jl_value_t *r = closedize_(e, t, upper, 0);
    if (r != t) {
        JL_GC_PUSH1(&r);
        c = ENV_W(e, R_CLOSED);
        push2(c, t, flag);
        jl_array_ptr_1d_push(c, r == NULL ? jl_nothing : r);
        JL_GC_POP();
    }
    return r;
}

static jl_value_t *closedize_(menv_t *e, jl_value_t *t, int upper, int depth)
{
    if (t == NULL)
        return NULL;
    if (isclosed(e->ctx, t))
        return t;
    if (depth > 32)
        return upper ? ANY : BOT;
    if (jl_is_typevar(t)) {
        jl_value_t *r = resolve(e, t);
        if (r != t)
            return closedize(e, r, upper, depth + 1);
        jl_svec_t *s = binding(e, t);
        if (s == NULL)
            return closedize(e, upper ? ((jl_tvar_t*)t)->ub : ((jl_tvar_t*)t)->lb, upper, depth + 1);
        return closedize(e, upper ? BUB(s) : BLB(s), upper, depth + 1);
    }
    jl_value_t *r = NULL;
    if (is_un(t)) {
        int i, n = jl_count_union_components(t);
        jl_value_t **ms;
        JL_GC_PUSHARGS(ms, n);
        int k = 0;
        for (i = 0; i < n; i++) {
            ms[k] = closedize(e, jl_nth_union_component(t, i), upper, depth + 1);
            if (ms[k] == NULL) {
                JL_GC_POP();
                return NULL;
            }
            if (jl_is_type(ms[k]) || jl_is_typevar(ms[k]))
                k++;
        }
        r = jl_type_union(ms, k);
        JL_GC_POP();
        return r;
    }
    if (istypeeq(t) || (jl_is_datatype(t) && !jl_is_tuple_type(t)))
        return upper ? rewrap_closed(e, t, depth) : BOT;
    if (has_vararg(t)) {
        size_t i, nf = nfixed(t);
        jl_value_t *va = tuple_va(t);
        jl_value_t **ps;
        JL_GC_PUSHARGS(ps, nf + 1);
        for (i = 0; i < nf; i++) {
            ps[i] = closedize(e, jl_tparam(t, i), upper, depth + 1);
            if (ps[i] == NULL || ps[i] == BOT || !(jl_is_type(ps[i]) || jl_is_typevar(ps[i]))) {
                JL_GC_POP();
                return ps[i] == BOT ? BOT : NULL;
            }
        }
        jl_value_t *n = va_N(va);
        if (n != NULL && jl_is_typevar(n))
            n = resolve(e, n);
        if (n != NULL && jl_is_long(n) && jl_unbox_long(n) < 0) {
            JL_GC_POP();
            return BOT;
        }
        jl_value_t *T = closedize(e, va_T(va), upper, depth + 1);
        if (T == NULL) {
            JL_GC_POP();
            return NULL;
        }
        ps[nf] = T;
        if (upper) {
            ps[nf] = make_va(T, (n != NULL && jl_is_long(n)) ? n : NULL);
            r = ps[nf] == NULL ? NULL : rebuild_datatype(jl_anytuple_type, ps, nf + 1);
        }
        else if (n == NULL) {
            r = rebuild_datatype(jl_anytuple_type, ps, nf);
        }
        else if (jl_is_long(n)) {
            if (jl_unbox_long(n) > 0 && T == BOT) {
                r = BOT;
            }
            else {
                ps[nf] = make_va(T, n);
                r = ps[nf] == NULL ? BOT : rebuild_datatype(jl_anytuple_type, ps, nf + 1);
            }
        }
        else {
            r = BOT;
        }
        JL_GC_POP();
        return r;
    }
    if (jl_is_tuple_type(t)) {
        size_t i, n = jl_nparams(t);
        jl_value_t **ps;
        JL_GC_PUSHARGS(ps, n);
        for (i = 0; i < n; i++) {
            ps[i] = closedize(e, jl_tparam(t, i), upper, depth + 1);
            if (ps[i] == NULL || ps[i] == BOT || !(jl_is_type(ps[i]) || jl_is_typevar(ps[i]))) {
                JL_GC_POP();
                return ps[i] == BOT ? BOT : NULL;
            }
        }
        r = rebuild_datatype(jl_anytuple_type, ps, n);
        JL_GC_POP();
        return r;
    }
    if (is_ua(t)) {
        if (!upper)
            return BOT;
        r = rewrap_closed(e, t, depth);
        return r == NULL ? closedize(e, ((jl_unionall_t*)t)->body, 1, depth + 1) : r;
    }
    return upper ? ANY : BOT;
}

static int disjoint(mctx_t *ctx, jl_value_t *a, jl_value_t *b)
{
    return meet_or_nothing(ctx, a, b) == BOT;
}

static int never_empty(mctx_t *ctx, jl_value_t *t)
{
    if (istypeeq(t) || isegal(t))
        return 1;
    if (jl_is_datatype(t)) {
        if (!jl_is_tuple_type(t))
            return 1;
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++) {
            jl_value_t *p = jl_tparam(t, i);
            if (!jl_is_vararg(p) && !never_empty(ctx, p))
                return 0;
        }
        return 1;
    }
    return isclosed(ctx, t) && t != BOT;
}

static jl_value_t *kind_of(mctx_t *ctx, jl_value_t *t) JL_NOTSAFEPOINT
{
    if (isopaque(ctx, t))
        return NULL;
    if (isclosed(ctx, t))
        return jl_typeof(t);
    if (istypeeq(t))
        return (jl_value_t*)jl_typeeq_type;
    if (isegal(t))
        return (jl_value_t*)jl_typeegal_type;
    if (jl_is_datatype(t))
        return (jl_value_t*)jl_datatype_type;
    if (is_un(t))
        return (jl_value_t*)jl_uniontype_type;
    if (is_ua(t))
        return (jl_value_t*)jl_unionall_type;
    return NULL;
}


static int in_kind(mctx_t *ctx, jl_value_t *s, jl_value_t *k)
{
    jl_value_t *X = jl_some_Type_T(s);
    if (isopaque(ctx, k))
        return 0;
    if (isopaque(ctx, X))
        return jl_subtype(k, (jl_value_t*)jl_type_type) && k != (jl_value_t*)jl_typeofbottom_type;
    return istypeeq(s) ? jl_typeeq_meets_kind(X, k) : jl_isa(X, k);
}

static jl_value_t *member_kinds(mctx_t *ctx, jl_value_t *t)
{
    jl_value_t *kinds[6] = { (jl_value_t*)jl_datatype_type, (jl_value_t*)jl_unionall_type,
                             (jl_value_t*)jl_uniontype_type, (jl_value_t*)jl_typeofbottom_type,
                             (jl_value_t*)jl_typeeq_type, (jl_value_t*)jl_typeegal_type };
    jl_value_t *ks[6];
    int i, j, nk = 0, n = jl_count_union_components(t);
    for (i = 0; i < n; i++) {
        jl_value_t *m = jl_nth_union_component(t, i);
        if (!(istypeeq(m) || isegal(m)))
            return NULL;
    }
    for (j = 0; j < 6; j++) {
        for (i = 0; i < n; i++) {
            if (in_kind(ctx, jl_nth_union_component(t, i), kinds[j])) {
                ks[nk++] = kinds[j];
                break;
            }
        }
    }
    return jl_type_union(ks, nk);
}

static int open_kind_meets(jl_value_t *X, jl_value_t *k)
{
    jl_datatype_t *ks[3];
    int i, n = 0;
    if (istypeeq(X)) {
        ks[n++] = jl_typeeq_type;
        ks[n++] = jl_datatype_type;
    }
    else if (jl_is_datatype(X)) {
        ks[n++] = jl_datatype_type;
        ks[n++] = jl_unionall_type;
        if (jl_is_tuple_type(X))
            ks[n++] = jl_uniontype_type;
    }
    else if (is_un(X) || is_ua(X)) {
        ks[n++] = jl_datatype_type;
        ks[n++] = jl_unionall_type;
        ks[n++] = jl_uniontype_type;
    }
    else {
        return 1;
    }
    for (i = 0; i < n; i++)
        if (jl_subtype((jl_value_t*)ks[i], k))
            return 1;
    return 0;
}

static int is_atom(mctx_t *ctx, jl_value_t *t) JL_NOTSAFEPOINT
{
    return istypeeq(t) || isegal(t) || (jl_is_datatype(t) && !jl_is_tuple_type(t)) || isopaque(ctx, t);
}

static int adoptable(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (jl_is_datatype(t) || is_ua(t))
        return 1;
    if (is_un(t)) {
        int i, n = jl_count_union_components(t);
        for (i = 0; i < n; i++) {
            jl_value_t *m = jl_nth_union_component(t, i);
            if (jl_is_typevar(m) || !adoptable(m))
                return 0;
        }
        return 1;
    }
    return 0;
}

static int meet(menv_t *e, jl_value_t *x, jl_value_t *y, int lift);
static int sub(menv_t *e, jl_value_t *lo, jl_value_t *hi);
static int equate(menv_t *e, jl_value_t *x, jl_value_t *y);
static int solve(menv_t *e);
static jl_value_t *build(menv_t *e, jl_value_t *x, jl_value_t *y);
static jl_value_t *wrap(menv_t *e, jl_value_t *t, ssize_t only);

static int has_pair(jl_array_t *a, jl_value_t *x, jl_value_t *y)
{
    size_t i, n = LEN(a);
    for (i = 0; i < n; i += 2)
        if (jeq(AREF(a, i), x) && jeq(AREF(a, i + 1), y))
            return 1;
    return 0;
}

typedef struct {
    size_t k;
    jl_value_t *n;
} tlen_t;

static jl_value_t *live_va(menv_t *e, jl_value_t *t) JL_NOTSAFEPOINT
{
    jl_value_t *v = tuple_va(t);
    if (v == NULL)
        return NULL;
    jl_value_t *n = va_N(v);
    if (n != NULL && jl_is_typevar(n))
        n = resolve(e, n);
    return (n != NULL && jl_is_long(n) && jl_unbox_long(n) == 0) ? NULL : v;
}

static jl_value_t *tuple_el(jl_value_t *t, jl_value_t *va, size_t i) JL_NOTSAFEPOINT
{
    return i < nfixed(t) ? jl_tparam(t, i) : va_T(va);
}

static int tuple_pairs(jl_value_t *x, jl_value_t *vx, jl_value_t *y, jl_value_t *vy, size_t *k) JL_NOTSAFEPOINT
{
    size_t nx = nfixed(x), ny = nfixed(y);
    if ((vx == NULL && ny > nx) || (vy == NULL && nx > ny))
        return 1;
    *k = vx == NULL ? nx : vy == NULL ? ny : (nx > ny ? nx : ny);
    return 0;
}

static tlen_t tuple_length(menv_t *e, jl_value_t *t, jl_value_t *va) JL_NOTSAFEPOINT
{
    tlen_t l;
    l.k = nfixed(t);
    if (va == NULL) {
        l.n = jl_box_long(0);
        return l;
    }
    jl_value_t *n = va_N(va);
    l.n = n == NULL ? NULL : (jl_is_typevar(n) ? resolve(e, n) : n);
    return l;
}

static jl_value_t *len_root(menv_t *e, jl_value_t *v, ssize_t *d)
{
    v = resolve(e, v);
    *d = 0;
    while (jl_is_typevar(v)) {
        jl_svec_t *s = binding(e, v);
        if (s == NULL || jl_svecref(s, B_OFFTO) == jl_nothing)
            break;
        *d += jl_unbox_long(jl_svecref(s, B_OFF));
        v = resolve(e, jl_svecref(s, B_OFFTO));
    }
    if (jl_is_long(v)) {
        ssize_t n = jl_unbox_long(v) + *d;
        *d = 0;
        return jl_box_long(n);
    }
    return v;
}

static int has_offset(menv_t *e, jl_value_t *v) JL_NOTSAFEPOINT
{
    jl_svec_t *s = binding(e, v);
    return s != NULL && jl_svecref(s, B_OFFTO) != jl_nothing;
}

static int at_least(menv_t *e, jl_value_t *v, ssize_t n)
{
    ssize_t d;
    jl_value_t *r = len_root(e, v, &d);
    if (jl_is_long(r))
        return jl_unbox_long(r) < n;
    jl_svec_t *s = binding(e, r);
    if (s == NULL || n - d <= jl_unbox_long(jl_svecref(s, B_MIN)))
        return 0;
    jl_value_t *b = jl_box_long(n - d);
    jl_svecset(s, B_MIN, b);
    e->changed = 1;
    e->cver = -1;
    return 0;
}

static int equate_len(menv_t *e, jl_value_t *v, ssize_t n)
{
    if (n < 0)
        return 1;
    jl_value_t *b = jl_box_long(n);
    JL_GC_PUSH1(&b);
    int r = equate(e, v, b);
    JL_GC_POP();
    return r;
}

static int relate(menv_t *e, jl_value_t *x, jl_value_t *y, ssize_t d)
{
    ssize_t dx, dy;
    jl_value_t *rx = NULL, *ry = NULL;
    JL_GC_PUSH2(&rx, &ry);
    rx = len_root(e, x, &dx);
    ry = len_root(e, y, &dy);
    ssize_t k = dy + d - dx;
    int r = 0;
    if (jl_is_long(rx) && jl_is_long(ry)) {
        r = jl_unbox_long(rx) != jl_unbox_long(ry) + k;
    }
    else if (jl_is_long(rx)) {
        r = jl_unbox_long(rx) - k < 0;
        if (!r && bound(e, ry))
            r = equate_len(e, ry, jl_unbox_long(rx) - k);
    }
    else if (jl_is_long(ry)) {
        r = jl_unbox_long(ry) + k < 0;
        if (!r && bound(e, rx))
            r = equate_len(e, rx, jl_unbox_long(ry) + k);
    }
    else if (rx == ry) {
        r = k != 0;
    }
    else if (bound(e, rx) && bound(e, ry)) {
        if (k == 0) {
            r = equate(e, rx, ry);
        }
        else {
            jl_value_t *hi = k > 0 ? rx : ry, *lo = k > 0 ? ry : rx;
            ssize_t off = k > 0 ? k : -k;
            jl_svec_t *sh = binding(e, hi);
            jl_svecset(sh, B_OFFTO, lo);
            jl_value_t *b = jl_box_long(off);
            jl_svecset(sh, B_OFF, b);
            e->changed = 1;
            e->cver = -1;
            r = at_least(e, lo, jl_unbox_long(jl_svecref(sh, B_MIN)) - off);
        }
    }
    JL_GC_POP();
    return r;
}

static int length_equal(menv_t *e, tlen_t x, tlen_t y)
{
    jl_value_t *nx = NULL, *ny = NULL;
    JL_GC_PUSH2(&nx, &ny);
    ssize_t kx = (ssize_t)x.k, ky = (ssize_t)y.k, d;
    nx = x.n;
    ny = y.n;
    if (nx != NULL && jl_is_typevar(nx)) {
        nx = len_root(e, nx, &d);
        kx += d;
    }
    if (ny != NULL && jl_is_typevar(ny)) {
        ny = len_root(e, ny, &d);
        ky += d;
    }
    int r = 0;
    if (nx == NULL || ny == NULL) {
        if (nx == NULL && ny != NULL && jl_is_long(ny) && ky + jl_unbox_long(ny) < kx)
            r = 1;
        else if (ny == NULL && nx != NULL && jl_is_long(nx) && kx + jl_unbox_long(nx) < ky)
            r = 1;
        else if (nx == NULL && ny != NULL && bound(e, ny))
            r = at_least(e, ny, kx - ky);
        else if (ny == NULL && nx != NULL && bound(e, nx))
            r = at_least(e, nx, ky - kx);
    }
    else {
        int ix = jl_is_long(nx), iy = jl_is_long(ny);
        if (ix && iy) {
            r = kx + jl_unbox_long(nx) != ky + jl_unbox_long(ny);
        }
        else if (ix || iy) {
            ssize_t n = ix ? kx + jl_unbox_long(nx) : ky + jl_unbox_long(ny);
            ssize_t k = ix ? ky : kx;
            jl_value_t *other = ix ? ny : nx;
            if (n - k < 0)
                r = 1;
            else if (bound(e, other))
                r = equate_len(e, other, n - k);
        }
        else if (bound(e, nx) && bound(e, ny)) {
            r = relate(e, nx, ny, ky - kx);
        }
    }
    JL_GC_POP();
    return r;
}

static int meet_tuples(menv_t *e, jl_value_t *x, jl_value_t *y, int lift)
{
    jl_value_t *vx = live_va(e, x), *vy = live_va(e, y);
    size_t k, i, nx = nfixed(x), ny = nfixed(y);
    if (tuple_pairs(x, vx, y, vy, &k))
        return 1;
    size_t mn = nx < ny ? nx : ny;
    for (i = 0; i < k; i++) {
        jl_value_t *p = tuple_el(x, vx, i), *q = tuple_el(y, vy, i);
        if (isclosed(e->ctx, p) && isclosed(e->ctx, q)) {
            if (p != BOT && q != BOT && meet_closed(e->ctx, p, q) == BOT)
                return 1;
        }
        else {
            TRY(meet(e, p, q, lift && i < mn));
        }
    }
    if (vx != NULL && vy != NULL && e->split)
        TRY(meet(e, va_T(vx), va_T(vy), 0));
    return length_equal(e, tuple_length(e, x, vx), tuple_length(e, y, vy));
}

static int sub_tuples(menv_t *e, jl_value_t *lo, jl_value_t *hi)
{
    jl_value_t *vl = live_va(e, lo), *vh = live_va(e, hi);
    size_t i, nl = nfixed(lo), nh = nfixed(hi), n = nl > nh ? nl : nh;
    for (i = 0; i < n; i++) {
        jl_value_t *p = i < nl ? jl_tparam(lo, i) : (vl ? va_T(vl) : NULL);
        jl_value_t *q = i < nh ? jl_tparam(hi, i) : (vh ? va_T(vh) : NULL);
        if (p == NULL || q == NULL)
            return 1;
        TRY(sub(e, p, q));
    }
    return length_equal(e, tuple_length(e, lo, vl), tuple_length(e, hi, vh));
}

static int equate_tuples(menv_t *e, jl_value_t *x, jl_value_t *y)
{
    jl_value_t *vx = live_va(e, x), *vy = live_va(e, y);
    size_t i, nx = nfixed(x), ny = nfixed(y);
    if (vx == NULL && vy == NULL) {
        if (nx != ny)
            return 1;
        for (i = 0; i < nx; i++)
            TRY(equate(e, jl_tparam(x, i), jl_tparam(y, i)));
        return 0;
    }
    if (vx == NULL || vy == NULL) {
        jl_value_t *f = vx == NULL ? y : x, *v = vx == NULL ? vy : vx, *g = vx == NULL ? x : y;
        size_t nf = nfixed(f), ng = nfixed(g);
        jl_value_t *T = va_T(v), *n = va_N(v);
        if (n != NULL && jl_is_typevar(n))
            n = resolve(e, n);
        if (ng < nf)
            return 1;
        int fixed_n = n != NULL && jl_is_long(n);
        if (fixed_n ? (ssize_t)ng != (ssize_t)nf + jl_unbox_long(n) : (n == NULL || !bound(e, n)))
            return 1;
        for (i = 0; i < nf; i++)
            TRY(equate(e, jl_tparam(f, i), jl_tparam(g, i)));
        for (i = nf; i < ng; i++)
            TRY(equate(e, T, jl_tparam(g, i)));
        if (fixed_n)
            return 0;
        jl_value_t *d = jl_box_long(ng - nf);
        JL_GC_PUSH1(&d);
        int r = equate(e, n, d);
        JL_GC_POP();
        return r;
    }
    size_t k;
    if (tuple_pairs(x, vx, y, vy, &k))
        return 1;
    for (i = 0; i < k; i++)
        TRY(equate(e, tuple_el(x, vx, i), tuple_el(y, vy, i)));
    TRY(equate(e, va_T(vx), va_T(vy)));
    jl_value_t *nxv = va_N(vx), *nyv = va_N(vy);
    if (nxv == NULL && nyv == NULL)
        return 0;
    if (nxv == NULL || nyv == NULL)
        return 1;
    return length_equal(e, tuple_length(e, x, vx), tuple_length(e, y, vy));
}

static int decompose(menv_t *e, int rel, jl_value_t *x, jl_value_t *y, int lift)
{
    mctx_t *ctx = e->ctx;
    if (isopaque(ctx, x) || isopaque(ctx, y))
        return !(rel == REL_SUB && isopaque(ctx, y) && !never_empty(ctx, x));
    if (istypeeq(x) && istypeeq(y))
        return equate(e, jl_typeeq_T(x), jl_typeeq_T(y));
    if ((isegal(x) && istypeeq(y) && rel != REL_EQ) || (isegal(y) && istypeeq(x) && rel == REL_MEET))
        return equate(e, jl_some_Type_T(x), jl_some_Type_T(y));
    if (istypeeq(x) || istypeeq(y) || isegal(x) || isegal(y)) {
        if (rel == REL_EQ)
            return 1;
        int tx = istypeeq(x) || isegal(x);
        jl_value_t *t = tx ? x : y, *n = tx ? y : x;
        if (jl_is_datatype(n) && !jl_subtype((jl_value_t*)jl_type_type, n)) {
            jl_value_t *cn = closedize(e, n, 1, 0);
            int r = 0;
            JL_GC_PUSH1(&cn);
            if (cn != NULL) {
                if (!issub(cn, (jl_value_t*)jl_type_type)) {
                    r = 1;
                }
                else {
                    jl_value_t *X = resolve(e, jl_some_Type_T(t));
                    if (istypeeq(t)) {
                        if (isclosed(ctx, X) && !isopaque(ctx, X) ? !jl_typeeq_meets_kind(X, cn) : !open_kind_meets(X, cn))
                            r = 1;
                    }
                    else {
                        jl_value_t *k = kind_of(ctx, X);
                        if (k != NULL && !issub(k, cn))
                            r = 1;
                    }
                }
            }
            JL_GC_POP();
            return r;
        }
        return 0;
    }
    if (!(jl_is_datatype(x) && jl_is_datatype(y)))
        return 0;
    if (jl_is_tuple_type(x) && jl_is_tuple_type(y))
        return rel == REL_MEET ? meet_tuples(e, x, y, lift) : rel == REL_SUB ? sub_tuples(e, x, y) : equate_tuples(e, x, y);
    if (jl_is_tuple_type(x) || jl_is_tuple_type(y))
        return 1;
    jl_datatype_t *dx = (jl_datatype_t*)x, *dy = (jl_datatype_t*)y, *h1 = NULL, *h2 = NULL, *s;
    if (rel == REL_EQ) {
        if (dx->name == dy->name) {
            h1 = dx;
            h2 = dy;
        }
    }
    else if ((s = common_head(dx, dy)) != NULL) {
        h1 = s;
        h2 = dy;
    }
    else if (rel == REL_MEET && (s = common_head(dy, dx)) != NULL) {
        h1 = s;
        h2 = dx;
    }
    if (h1 == NULL)
        return 1;
    size_t i, n1 = jl_nparams(h1), n2 = jl_nparams(h2);
    for (i = 0; i < n1 && i < n2; i++)
        TRY(equate(e, jl_tparam(h1, i), jl_tparam(h2, i)));
    return 0;
}

static int local_meet(menv_t *e, jl_value_t *a, jl_value_t *b, int liftb, menv_t *sub, jl_value_t **pa, jl_value_t **pb);
static int commit(menv_t *e, jl_array_t *alts, jl_value_t *keep_v);

static int meet_union(menv_t *e, jl_value_t *x, jl_value_t *u)
{
    if (e->depth >= 3 || EXHAUSTED(e->ctx))
        return 0;
    int r = 0, i, n = jl_count_union_components(u);
    menv_t sub;
    sub.root = NULL;
    jl_value_t *pa = NULL, *pb = NULL;
    jl_array_t *alts = NULL, *ms = NULL;
    JL_GC_PUSH5(&sub.root, &pa, &pb, &alts, &ms);
    alts = jl_alloc_vec_any(0);
    ms = jl_alloc_vec_any(0);
    for (i = 0; i < n; i++) {
        jl_value_t *m = jl_nth_union_component(u, i);
        if (!local_meet(e, m, x, 0, &sub, &pa, &pb)) {
            jl_array_ptr_1d_push(alts, (jl_value_t*)sub.root);
            jl_array_ptr_1d_push(ms, m);
        }
    }
    if (LEN(alts) == 0)
        r = 1;
    else if (LEN(alts) == 1)
        r = meet(e, x, AREF(ms, 0), 1);
    else
        r = commit(e, alts, NULL);
    JL_GC_POP();
    return r;
}

static int meet_where(menv_t *e, jl_value_t *u, jl_value_t *t)
{
    if (e->depth >= 3 || EXHAUSTED(e->ctx))
        return 0;
    menv_t sub;
    sub.root = NULL;
    jl_value_t *pa = NULL, *pb = NULL;
    jl_array_t *alts = NULL;
    JL_GC_PUSH4(&sub.root, &pa, &pb, &alts);
    int r = 1;
    if (!local_meet(e, u, t, 1, &sub, &pa, &pb)) {
        alts = jl_alloc_vec_any(0);
        jl_array_ptr_1d_push(alts, (jl_value_t*)sub.root);
        r = commit(e, alts, NULL);
    }
    JL_GC_POP();
    return r;
}

static int meet(menv_t *e, jl_value_t *x, jl_value_t *y, int lift)
{
    mctx_t *ctx = e->ctx;
    x = resolve(e, x);
    y = resolve(e, y);
    if (x == y || x == ANY || y == ANY)
        return 0;
    if (!(jl_is_type(x) || jl_is_typevar(x)) || !(jl_is_type(y) || jl_is_typevar(y)))
        return 1;
    int r = 0;
    jl_value_t *body = NULL, *ct = NULL;
    JL_GC_PUSH4(&x, &y, &body, &ct);
    if (isclosed(ctx, x) && isclosed(ctx, y)) {
        if (nominally_disjoint(x, y)) {
            r = 1;
            goto out;
        }
        if (issub(x, y) || issub(y, x))
            goto out;
    }
    if (lift && (is_ua(x) || is_ua(y)) && !bound(e, x) && !bound(e, y)) {
        if (is_ua(x)) {
            body = lifted(e, REL_MEET, y, x);
            r = meet(e, body, y, 1);
        }
        else {
            body = lifted(e, REL_MEET, x, y);
            r = meet(e, x, body, 1);
        }
        goto out;
    }
    if (!(is_tuple_type(x) && is_tuple_type(y)) && !has_pair(ENV_ARR(e, R_SLOTS), x, y))
        push2(ENV_W(e, R_SLOTS), x, y);
    if (bound(e, x) || bound(e, y)) {
        jl_value_t *v = bound(e, x) ? x : y, *t = bound(e, x) ? y : x;
        if (bound(e, x) && bound(e, y) && !diag_of(e, v)) {
            jl_value_t *tmp = v;
            v = t;
            t = tmp;
        }
        if (diag_of(e, v)) {
            if (is_concrete_leaf(ctx, t) || diag_of(e, t)) {
                r = equate(e, v, t);
            }
            else {
                ct = closedize(e, t, 1, 0);
                if (ct != NULL && issub(ct, (jl_value_t*)jl_type_type) && !jl_is_datatype(ct)) {
                    ct = member_kinds(ctx, ct);
                    if (ct != NULL)
                        fact(e, v, ct);
                }
                else {
                    fact(e, v, t);
                }
            }
        }
        else if (is_concrete_leaf(ctx, t) || isegal(t)) {
            fact(e, t, v);
        }
    }
    else if (is_un(x) || is_un(y)) {
        if (!(isclosed(ctx, x) && isclosed(ctx, y)))
            r = meet_union(e, is_un(x) ? y : x, is_un(x) ? x : y);
    }
    else if (!(is_ua(x) || is_ua(y))) {
        r = decompose(e, REL_MEET, x, y, lift);
    }
    else if (!(isclosed(ctx, x) && isclosed(ctx, y))) {
        r = meet_where(e, is_ua(x) ? x : y, is_ua(x) ? y : x);
    }
out:
    JL_GC_POP();
    return r;
}

static int meet_bound(menv_t *e, jl_value_t *v, jl_value_t *a, jl_value_t *b, jl_value_t **out);

static int sub_var(menv_t *e, jl_value_t *v, jl_value_t *hi)
{
    mctx_t *ctx = e->ctx;
    int r = 0, implied, have_m = 0;
    jl_value_t *c = NULL, *m = NULL, *sh = NULL, *ub;
    jl_svec_t *s;
    JL_GC_PUSH3(&c, &m, &sh);
    if (bound(e, hi)) {
        c = closedize(e, hi, 1, 0);
        if (c != NULL && sub(e, v, c)) {
            r = 1;
            goto out;
        }
        c = closedize(e, v, 0, 0);
        if (c != NULL && sub(e, c, hi))
            r = 1;
        goto out;
    }
    if (isclosed(ctx, hi)) {
        if (set_ub(e, v, hi)) {
            r = 1;
            goto out;
        }
        c = closedize(e, v, 0, 0);
        if (c != NULL && c != BOT && sub(e, c, hi))
            r = 1;
        goto out;
    }
    s = binding(e, v);
    ub = BUB(s);
    implied = ub != ANY && issub(ub, hi);
    if (!implied && ub != ANY) {
        c = subst(e, ub);
        sh = subst(e, hi);
        if (c == NULL || sh == NULL) {
            r = 1;
            goto out;
        }
        implied = issub(c, sh);
        c = sh = NULL;
    }
    if (!implied && adoptable(hi)) {
        if (ub == ANY) {
            sh = subst(e, hi);
            if (sh == NULL) {
                r = 1;
                goto out;
            }
            m = occurs((jl_tvar_t*)v, sh) ? NULL : hi;
        }
        else if (meet_bound(e, v, ub, hi, &m)) {
            r = 1;
            goto out;
        }
        have_m = m != NULL;
    }
    if (implied) {
    }
    else if (have_m && !isclosed(ctx, m)) {
        if (!jeq(m, BUB(s))) {
            jl_svecset(s, B_UB, m);
            touched(e);
        }
    }
    else {
        if (!have_m)
            m = closedize(e, hi, 1, 0);
        if (set_ub(e, v, m == NULL ? ANY : m)) {
            r = 1;
            goto out;
        }
    }
    c = closedize(e, v, 0, 0);
    if (c != NULL && c != BOT && sub(e, c, hi))
        r = 1;
out:
    JL_GC_POP();
    return r;
}

static int sub_union(menv_t *e, jl_value_t *lo, jl_value_t *u)
{
    mctx_t *ctx = e->ctx;
    int i, n = jl_count_union_components(u), r = 0;
    jl_value_t *closed = NULL, *lub = NULL, *cub = NULL;
    jl_array_t *cands = NULL;
    jl_value_t **cl;
    JL_GC_PUSHARGS(cl, n + 4);
    int nc = 0;
    for (i = 0; i < n; i++) {
        jl_value_t *m = jl_nth_union_component(u, i);
        if (isclosed(ctx, m))
            cl[nc++] = m;
    }
    closed = jl_type_union(cl, nc);
    cl[n] = closed;
    if (isclosed(ctx, lo) && issub(lo, closed))
        goto out;
    lub = closedize(e, lo, 1, 0);
    if (lub == NULL)
        lub = ANY;
    cl[n + 1] = lub;
    cands = jl_alloc_vec_any(0);
    cl[n + 2] = (jl_value_t*)cands;
    for (i = 0; i < n; i++) {
        jl_value_t *c = jl_nth_union_component(u, i);
        cub = closedize(e, c, 1, 0);
        cl[n + 3] = cub;
        if (cub == NULL || !jl_is_type(cub) || !disjoint(ctx, lub, cub))
            jl_array_ptr_1d_push(cands, c);
    }
    if (LEN(cands) == 0)
        r = 1;
    else if (LEN(cands) == 1)
        r = sub(e, lo, AREF(cands, 0));
out:
    JL_GC_POP();
    return r;
}

static int sub(menv_t *e, jl_value_t *lo, jl_value_t *hi)
{
    mctx_t *ctx = e->ctx;
    lo = resolve(e, lo);
    hi = resolve(e, hi);
    if (lo == hi || lo == BOT || hi == ANY)
        return 0;
    if (hi == BOT && never_empty(ctx, lo))
        return 1;
    int r = 0;
    jl_value_t *c = NULL;
    JL_GC_PUSH3(&lo, &hi, &c);
    int clo = isclosed(ctx, lo), chi = isclosed(ctx, hi);
    if (clo && !chi) {
        c = closedize(e, hi, 1, 0);
        if (c != NULL && !issub(lo, c)) {
            r = 1;
            goto out;
        }
    }
    if (is_ua(lo) && !(clo && chi)) {
        jl_unionall_t *u = (jl_unionall_t*)lo;
        c = inst(u->body, u->var, u->var->lb);
        if (c != NULL && sub(e, c, hi)) {
            r = 1;
            goto out;
        }
        c = inst(u->body, u->var, u->var->ub);
        if (c != NULL && sub(e, c, hi))
            r = 1;
    }
    else if (clo && chi) {
        r = !issub(lo, hi);
    }
    else if (is_un(lo)) {
        int i, n = jl_count_union_components(lo);
        for (i = 0; i < n && !r; i++)
            r = sub(e, jl_nth_union_component(lo, i), hi);
    }
    else if (bound(e, lo)) {
        r = sub_var(e, lo, hi);
    }
    else if (bound(e, hi)) {
        c = closedize(e, lo, 0, 0);
        set_lb(e, hi, c);
        c = closedize(e, hi, 1, 0);
        if (c != NULL && c != ANY)
            r = sub(e, lo, c);
    }
    else if (is_ua(hi)) {
        c = lifted(e, REL_SUB, lo, hi);
        r = sub(e, lo, c);
    }
    else if (is_un(hi)) {
        r = sub_union(e, lo, hi);
    }
    else {
        r = decompose(e, REL_SUB, lo, hi, 0);
    }
out:
    JL_GC_POP();
    return r;
}

static int relate(menv_t *e, jl_value_t *x, jl_value_t *y, ssize_t d);

static int equate(menv_t *e, jl_value_t *x, jl_value_t *y)
{
    mctx_t *ctx = e->ctx;
    x = resolve(e, x);
    y = resolve(e, y);
    if (x == y)
        return 0;
    if (has_offset(e, x) || has_offset(e, y))
        return relate(e, x, y, 0);
    int bx = bound(e, x), by = bound(e, y);
    if (bx && by) {
        int rx = rank(e, x), ry = rank(e, y);
        if (rx != ry)
            return rx < ry ? bind_eq(e, x, y) : bind_eq(e, y, x);
        return canonical(e, x, y) == x ? bind_eq(e, y, x) : bind_eq(e, x, y);
    }
    if (bx || by)
        return bx ? bind_eq(e, x, y) : bind_eq(e, y, x);
    if (isclosed(ctx, x) && isclosed(ctx, y))
        return !jeq(x, y);
    if (is_un(x) || is_un(y) || x == BOT || y == BOT) {
        fact(e, x, y);
        fact(e, y, x);
        return 0;
    }
    if (is_ua(x) || is_ua(y)) {
        jl_value_t *hx = head(jl_unwrap_unionall(x)), *hy = head(jl_unwrap_unionall(y));
        if (hx != NULL && hy != NULL && hx != hy)
            return 1;
        jl_value_t *u = is_ua(x) ? x : y, *d = is_ua(x) ? y : x;
        if (jl_is_datatype(d) && is_ua(u)) {
            jl_unionall_t *ua = (jl_unionall_t*)u;
            if (ua->var->lb != ua->var->ub) {
                int acc[2];
                occurrences(ua->body, ua->var, acc);
                if (acc[1] > 0)
                    return 1;
            }
        }
        fact(e, x, y);
        fact(e, y, x);
        return 0;
    }
    return decompose(e, REL_EQ, x, y, 0);
}

static int local_meet(menv_t *e, jl_value_t *a, jl_value_t *b, int liftb, menv_t *sub, jl_value_t **pa, jl_value_t **pb)
{
    if (nominally_disjoint(a, b))
        return 1;
    e->ctx->work++;
    local_env(sub, e);
    *pa = a;
    *pb = b;
    while (is_ua(*pa))
        *pa = lift(sub, *pa);
    while (liftb && is_ua(*pb))
        *pb = lift(sub, *pb);
    jl_array_t *el = ENV_ARR(e, R_LOCALS);
    size_t i, n = LEN(el), n0 = NB(e), n1 = NB(sub);
    if (n > 0 || n1 > n0) {
        jl_array_t *locals = ENV_W(sub, R_LOCALS);
        for (i = 0; i < n; i++)
            jl_array_ptr_1d_push(locals, AREF(el, i));
        for (i = n0; i < n1; i++)
            jl_array_ptr_1d_push(locals, jl_svecref(BND(sub, i), B_VAR));
    }
    if (meet(sub, *pb, *pa, 0))
        return 1;
    return solve(sub);
}

static int local_to(menv_t *e, menv_t *a0, jl_value_t *t) JL_NOTSAFEPOINT
{
    size_t i, n = NB(a0);
    for (i = 0; i < n; i++) {
        jl_value_t *w = jl_svecref(BND(a0, i), B_VAR);
        if (!bound(e, w) && jl_has_typevar(t, (jl_tvar_t*)w))
            return 1;
    }
    return 0;
}

enum { AG_EQ, AG_UB, AG_LB };

static int agreed(menv_t *e, jl_array_t *alts, jl_value_t *k, int what, jl_value_t **out)
{
    *out = NULL;
    size_t i, n = LEN(alts);
    jl_value_t *x = NULL, *y = NULL;
    JL_GC_PUSH2(&x, &y);
    for (i = 0; i < n; i++) {
        menv_t a;
        a.root = (jl_svec_t*)AREF(alts, i);
        a.ctx = e->ctx;
        a.version = -1;
        jl_value_t *t = k;
        if (what != AG_EQ) {
            jl_svec_t *s = binding(&a, k);
            t = what == AG_UB ? BUB(s) : BLB(s);
        }
        y = subst(&a, t);
        if (y == NULL) {
            JL_GC_POP();
            return 1;
        }
        if (i == 0) {
            x = y;
        }
        else if (!jeq(y, x)) {
            JL_GC_POP();
            return 0;
        }
    }
    *out = x;
    JL_GC_POP();
    return 0;
}

static int commit(menv_t *e, jl_array_t *alts, jl_value_t *keep_v)
{
    size_t i, j, nk = NB(e), na = LEN(alts);
    menv_t a0;
    a0.root = (jl_svec_t*)AREF(alts, 0);
    a0.ctx = e->ctx;
    a0.version = -1;
    jl_value_t *x = NULL;
    int r = 0;
    JL_GC_PUSH1(&x);
    for (i = 0; i < nk && !r; i++) {
        jl_value_t *k = jl_svecref(BND(e, i), B_VAR);
        if (resolve(e, k) != k)
            continue;
        if (keep_v != NULL) {
            jl_value_t *o = owner_of(e, k);
            if (o == NULL || resolve(e, o) != resolve(e, keep_v))
                continue;
        }
        if (agreed(e, alts, k, AG_EQ, &x)) {
            r = 1;
            break;
        }
        if (x != NULL && x != k) {
            if (!local_to(e, &a0, x))
                r = equate(e, k, x);
            continue;
        }
        int all_free = 1;
        for (j = 0; j < na; j++) {
            menv_t a;
            a.root = (jl_svec_t*)AREF(alts, j);
            a.ctx = e->ctx;
            a.version = -1;
            if (resolve(&a, k) != k) {
                all_free = 0;
                break;
            }
        }
        if (!all_free)
            continue;
        if (agreed(e, alts, k, AG_UB, &x)) {
            r = 1;
            break;
        }
        if (x != NULL && !local_to(e, &a0, x) && !jeq(x, BUB(BND(e, i))))
            fact(e, k, x);
        if (agreed(e, alts, k, AG_LB, &x)) {
            r = 1;
            break;
        }
        if (x != NULL && !local_to(e, &a0, x) && !jeq(x, BLB(BND(e, i))))
            fact(e, x, k);
    }
    JL_GC_POP();
    return r;
}

static int meet_bound_(menv_t *e, jl_value_t *v, jl_value_t *a, jl_value_t *b, jl_value_t **out)
{
    menv_t sub;
    sub.root = NULL;
    jl_value_t *pa = NULL, *pb = NULL, *m = NULL;
    jl_array_t *alts = NULL;
    JL_GC_PUSH5(&sub.root, &pa, &pb, &m, &alts);
    if (local_meet(e, a, b, 1, &sub, &pa, &pb)) {
        *out = BOT;
        JL_GC_POP();
        return 0;
    }
    m = build(&sub, pb, pa);
    if (m != NULL)
        m = wrap(&sub, m, NB(e));
    if (m == NULL)
        m = BOT;
    alts = jl_alloc_vec_any(0);
    jl_array_ptr_1d_push(alts, (jl_value_t*)sub.root);
    jl_svec_t *vs = binding(e, v);
    if (commit(e, alts, (vs != NULL && jl_svecref(vs, B_INH) == jl_true) ? NULL : v)) {
        JL_GC_POP();
        return 1;
    }
    jl_value_t *sm = subst(e, m);
    if (sm == NULL) {
        JL_GC_POP();
        return 1;
    }
    *out = occurs((jl_tvar_t*)v, sm) ? NULL : m;
    JL_GC_POP();
    return 0;
}

static int meet_bound(menv_t *e, jl_value_t *v, jl_value_t *a, jl_value_t *b, jl_value_t **out)
{
    mctx_t *ctx = e->ctx;
    *out = NULL;
    if (e->depth >= 3 || EXHAUSTED(ctx))
        return 0;
    if (ctx->bounds == NULL)
        ctx->bounds = jl_alloc_vec_any(0);
    size_t i, n = LEN(ctx->bounds);
    for (i = 0; i < n; i += 2)
        if (AREF(ctx->bounds, i) == a && AREF(ctx->bounds, i + 1) == b)
            return 0;
    push2(ctx->bounds, a, b);
    int r = meet_bound_(e, v, a, b, out);
    jl_array_del_end(ctx->bounds, 2);
    return r;
}

static int solve(menv_t *e)
{
    mctx_t *ctx = e->ctx;
    jl_value_t *lb = NULL, *ub = NULL, *cx = NULL, *cy = NULL, *m = NULL;
    jl_array_t *slots = NULL;
    int r = 0, round;
    size_t i;
    JL_GC_PUSH6(&lb, &ub, &cx, &cy, &m, &slots);
    for (round = 0; round < 32; round++) {
        e->changed = 0;
        jl_array_t *subs = ENV_ARR(e, R_SUBS);
        size_t n = LEN(subs);
        for (i = 0; i < n; i += 2)
            if (sub(e, AREF(subs, i), AREF(subs, i + 1)))
                goto unsat;
        if (e->met != e->version) {
            e->met = e->version;
            jl_array_t *meets = ENV_ARR(e, R_MEETS);
            n = LEN(meets);
            for (i = 0; i < n; i += 2)
                if (meet(e, AREF(meets, i), AREF(meets, i + 1), 0))
                    goto unsat;
        }
        for (i = 0; i < NB(e); i++) {
            jl_value_t *v = jl_svecref(BND(e, i), B_VAR);
            lb = closedize(e, v, 0, 0);
            ub = closedize(e, v, 1, 0);
            if (in_array(ctx->lengths, v) && ((is_type(lb) && lb != BOT) || (is_type(ub) && ub != ANY)))
                goto unsat;
            ssize_t mn = jl_unbox_long(jl_svecref(BND(e, i), B_MIN));
            if (mn > 0) {
                jl_value_t *rv = resolve(e, v);
                if (jl_is_long(rv) && jl_unbox_long(rv) < mn)
                    goto unsat;
            }
            if (!is_type(lb) || !is_type(ub))
                continue;
            if (!issub(lb, ub))
                goto unsat;
            if (diag_of(e, v) && BEQ(BND(e, i)) == jl_nothing && is_concrete_leaf(ctx, lb))
                if (equate(e, v, lb))
                    goto unsat;
        }
        jl_array_t *checks = ENV_ARR(e, R_CHECKS);
        for (i = 0; i < LEN(checks); i += 3) {
            jl_value_t *p = AREF(checks, i + 1);
            if (!(jl_is_type(p) || jl_is_typevar(p)))
                continue;
            lb = closedize(e, p, 0, 0);
            if (is_type(lb) && !issub(lb, AREF(checks, i + 2)))
                goto unsat;
            ub = closedize(e, p, 1, 0);
            if (is_type(ub) && !issub(AREF(checks, i), ub))
                goto unsat;
        }
        slots = ENV_ARR(e, R_SLOTS);
        for (i = 0; i < LEN(slots); i += 2) {
            jl_value_t *x = AREF(slots, i), *y = AREF(slots, i + 1);
            if (isclosed(ctx, x) && isclosed(ctx, y))
                continue;
            cx = closedize(e, x, 1, 0);
            cy = closedize(e, y, 1, 0);
            if (!is_type(cx) || !is_type(cy))
                continue;
            m = meet_or_nothing(ctx, cx, cy);
            if (m == BOT)
                goto unsat;
            if (is_type(m) && is_concrete_leaf(ctx, m)) {
                if (!isclosed(ctx, x))
                    fact(e, m, x);
                if (!isclosed(ctx, y))
                    fact(e, m, y);
            }
        }
        if (!e->changed)
            break;
    }
    JL_GC_POP();
    return 0;
unsat:
    r = 1;
    JL_GC_POP();
    return r;
}

static jl_value_t *build_local(menv_t *e, jl_value_t *u, jl_value_t *t);

static jl_value_t *tuple_of(jl_value_t **ps, size_t n)
{
    return rebuild_datatype(jl_anytuple_type, ps, n);
}

static jl_value_t *build(menv_t *e, jl_value_t *x, jl_value_t *y)
{
    mctx_t *ctx = e->ctx;
    jl_value_t *r = NULL, *c1 = NULL, *c2 = NULL;
    jl_value_t **ps = NULL;
    JL_GC_PUSH5(&x, &y, &r, &c1, &c2);
    x = subst(e, x);
    if (x == NULL)
        goto out;
    y = subst(e, y);
    if (y == NULL)
        goto out;
    if (x == y || y == ANY) {
        r = x;
        goto out;
    }
    if (x == ANY) {
        r = y;
        goto out;
    }
    if (x == BOT || y == BOT) {
        r = BOT;
        goto out;
    }
    if (isclosed(ctx, x) && isclosed(ctx, y)) {
        if (issub(x, y)) {
            r = x;
            goto out;
        }
        if (issub(y, x)) {
            r = y;
            goto out;
        }
    }
    if (istypeeq(x) && istypeeq(y)) {
        r = canonical(e, jl_typeeq_T(x), jl_typeeq_T(y));
        r = (jl_value_t*)jl_wrap_Type(r);
        goto out;
    }
    int k;
    for (k = 0; k < 2; k++) {
        jl_value_t *t = k == 0 ? x : y, *kd = k == 0 ? y : x;
        if (!(istypeeq(t) && !istypeeq(kd)))
            continue;
        c1 = closedize(e, kd, 1, 0);
        if (c1 == NULL || !issub(c1, (jl_value_t*)jl_type_type))
            continue;
        c2 = closedize(e, t, 1, 0);
        r = (c2 != NULL && issub(c1, c2)) ? kd : t;
        goto out;
    }
    if (is_un(x) || is_un(y)) {
        jl_value_t *a = is_un(x) ? x : y, *b = is_un(x) ? y : x;
        int i, n = jl_count_union_components(a);
        JL_GC_PUSHARGS(ps, n);
        for (i = 0; i < n; i++) {
            jl_value_t *m = jl_nth_union_component(a, i);
            ps[i] = (isclosed(ctx, m) && isclosed(ctx, b)) ? build(e, m, b) : build_local(e, m, b);
            if (ps[i] == NULL) {
                JL_GC_POP();
                r = NULL;
                goto out;
            }
        }
        r = jl_type_union(ps, n);
        JL_GC_POP();
        goto out;
    }
    if (bound(e, x) || bound(e, y)) {
        jl_value_t *v = bound(e, x) ? x : y, *t = bound(e, x) ? y : x;
        if (bound(e, t) && diag_of(e, t) && !diag_of(e, v)) {
            jl_value_t *tmp = v;
            v = t;
            t = tmp;
        }
        if (diag_of(e, v)) {
            r = v;
            goto out;
        }
        c1 = closedize(e, v, 1, 0);
        c2 = closedize(e, t, 1, 0);
        if (c1 == NULL)
            c1 = ANY;
        if (c2 == NULL)
            c2 = ANY;
        int vt = issub(c1, c2), tv = issub(c2, c1);
        if (vt && tv && bound(e, t))
            r = canonical(e, v, t);
        else if (vt)
            r = v;
        else if (tv)
            r = t;
        else if (isclosed(ctx, t))
            r = meet_closed(ctx, c1, t);
        else
            r = meet_closed(ctx, c1, c2);
        goto out;
    }
    if (jl_is_datatype(x) && jl_is_datatype(y)) {
        if (jl_is_tuple_type(x) && jl_is_tuple_type(y)) {
            jl_value_t *vx = live_va(e, x), *vy = live_va(e, y);
            size_t kk, i;
            if (tuple_pairs(x, vx, y, vy, &kk)) {
                r = BOT;
                goto out;
            }
            JL_GC_PUSHARGS(ps, kk + 1);
            for (i = 0; i < kk; i++) {
                jl_value_t *p = tuple_el(x, vx, i), *q = tuple_el(y, vy, i);
                ps[i] = isclosed(ctx, p) && isclosed(ctx, q) ? meet_closed(ctx, p, q) : build(e, p, q);
                if (ps[i] == NULL || ps[i] == BOT) {
                    r = ps[i];
                    JL_GC_POP();
                    goto out;
                }
            }
            if (vx == NULL || vy == NULL) {
                r = tuple_of(ps, kk);
                JL_GC_POP();
                goto out;
            }
            jl_value_t *T = build(e, va_T(vx), va_T(vy));
            if (T == NULL) {
                JL_GC_POP();
                r = NULL;
                goto out;
            }
            ps[kk] = T;
            jl_value_t *nn = NULL, *al[2];
            int j, nal = 0, fixedn = 0;
            for (j = 0; j < 2; j++) {
                jl_value_t *f = j == 0 ? x : y, *v = j == 0 ? vx : vy;
                jl_value_t *m = va_N(v);
                if (m != NULL && jl_is_typevar(m))
                    m = resolve(e, m);
                if (m != NULL && jl_is_long(m)) {
                    ssize_t d = jl_unbox_long(m) - ((ssize_t)kk - (ssize_t)nfixed(f));
                    if (d < 0) {
                        JL_GC_POP();
                        r = BOT;
                        goto out;
                    }
                    nn = jl_box_long(d);
                    fixedn = 1;
                    break;
                }
                else if (m != NULL && nfixed(f) == kk) {
                    al[nal++] = m;
                }
            }
            if (!fixedn && nal > 0)
                nn = nal == 1 ? al[0] : canonical(e, al[0], al[1]);
            c1 = nn;
            if (T == BOT) {
                r = (nn != NULL && jl_is_long(nn) && jl_unbox_long(nn) > 0) ? BOT : tuple_of(ps, kk);
                JL_GC_POP();
                goto out;
            }
            ps[kk] = make_va(T, nn);
            r = ps[kk] == NULL ? NULL : tuple_of(ps, kk + 1);
            JL_GC_POP();
            goto out;
        }
        jl_datatype_t *dx = (jl_datatype_t*)x, *dy = (jl_datatype_t*)y;
        if (dx->name == dy->name) {
            r = canonical(e, x, y);
            goto out;
        }
        if (common_head(dx, dy) != NULL) {
            r = x;
            goto out;
        }
        if (common_head(dy, dx) != NULL) {
            r = y;
            goto out;
        }
    }
    if (is_ua(x)) {
        r = build_local(e, x, y);
        goto out;
    }
    if (is_ua(y)) {
        r = build_local(e, y, x);
        goto out;
    }
    c1 = closedize(e, x, 1, 0);
    c2 = closedize(e, y, 1, 0);
    r = meet_closed(ctx, c1, c2);
out:
    JL_GC_POP();
    return r;
}

static jl_value_t *build_local(menv_t *e, jl_value_t *u, jl_value_t *t)
{
    menv_t sub;
    sub.root = NULL;
    jl_value_t *pu = NULL, *pt = NULL, *r = NULL;
    JL_GC_PUSH4(&sub.root, &pu, &pt, &r);
    if (EXHAUSTED(e->ctx)) {
        pu = closedize(e, u, 1, 0);
        pt = closedize(e, t, 1, 0);
        r = meet_closed(e->ctx, pu, pt);
        JL_GC_POP();
        return r;
    }
    if (local_meet(e, u, t, 1, &sub, &pu, &pt)) {
        JL_GC_POP();
        return BOT;
    }
    size_t i, n = NB(e);
    jl_array_t *locals = ENV_ARR(&sub, R_LOCALS);
    for (i = 0; i < n; i++) {
        jl_svec_t *b = BND(e, i);
        jl_value_t *k = jl_svecref(b, B_VAR);
        if (in_array(locals, k) || !diag_of(e, k))
            continue;
        jl_svec_t *sb = binding(&sub, k);
        jl_svecset(sb, B_LB, BLB(b));
        jl_svecset(sb, B_UB, BUB(b));
        jl_svecset(sb, B_EQ, BEQ(b));
        sub.cver = -1;
    }
    r = build(&sub, pt, pu);
    if (r != NULL)
        r = wrap(&sub, r, n);
    if (r == NULL)
        r = BOT;
    JL_GC_POP();
    return r;
}

static void wrap_visit(menv_t *e, size_t start, jl_value_t *v, arraylist_t *seen, arraylist_t *vars, int *fail)
{
    size_t i;
    for (i = 0; i < seen->len; i++)
        if (seen->items[i] == v)
            return;
    arraylist_push(seen, v);
    jl_value_t *lb = NULL, *ub = NULL;
    JL_GC_PUSH2(&lb, &ub);
    jl_svec_t *s = binding(e, v);
    lb = subst(e, BLB(s));
    ub = subst(e, BUB(s));
    if (lb == NULL || ub == NULL) {
        *fail = 1;
        JL_GC_POP();
        return;
    }
    size_t n = NB(e);
    for (i = start; i < n && !*fail; i++) {
        jl_value_t *w = jl_svecref(BND(e, i), B_VAR);
        if (w != v && (jl_has_typevar(ub, (jl_tvar_t*)w) || jl_has_typevar(lb, (jl_tvar_t*)w)))
            wrap_visit(e, start, w, seen, vars, fail);
    }
    arraylist_push(vars, v);
    JL_GC_POP();
}

static ssize_t list_index(arraylist_t *l, void *x) JL_NOTSAFEPOINT
{
    size_t i;
    for (i = 0; i < l->len; i++)
        if (l->items[i] == x)
            return i;
    return -1;
}

static jl_value_t *wrap_bound(menv_t *e, size_t start, jl_value_t *v, int upper, arraylist_t *vars, jl_array_t *fresh, arraylist_t *order)
{
    jl_svec_t *s = binding(e, v);
    jl_value_t *b = subst(e, upper ? BUB(s) : BLB(s));
    if (b == NULL)
        return NULL;
    if (isclosed(e->ctx, b))
        return b;
    JL_GC_PUSH1(&b);
    size_t i, n = NB(e);
    for (i = start; i < n; i++) {
        jl_value_t *w = jl_svecref(BND(e, i), B_VAR);
        if (!jl_has_typevar(b, (jl_tvar_t*)w))
            continue;
        ssize_t j = list_index(vars, w);
        if (j < 0 || j >= (ssize_t)LEN(fresh) || !jl_is_typevar(AREF(fresh, j))) {
            b = closedize(e, v, upper, 0);
            JL_GC_POP();
            return b;
        }
    }
    for (i = 0; i < order->len; i++) {
        size_t j = (size_t)order->items[i];
        jl_value_t *nb = inst(b, (jl_tvar_t*)vars->items[j], AREF(fresh, j));
        if (nb == NULL) {
            b = closedize(e, v, upper, 0);
            JL_GC_POP();
            return b;
        }
        b = nb;
    }
    JL_GC_POP();
    return b;
}

static jl_value_t *wrap(menv_t *e, jl_value_t *t, ssize_t only)
{
    if (t == BOT)
        return t;
    size_t start = only < 0 ? 0 : (size_t)only;
    size_t i, n = NB(e);
    arraylist_t direct, seen, vars, order;
    arraylist_new(&direct, 0);
    for (i = start; i < n; i++) {
        jl_value_t *v = jl_svecref(BND(e, i), B_VAR);
        if (jl_has_typevar(t, (jl_tvar_t*)v))
            arraylist_push(&direct, v);
    }
    if (direct.len == 0) {
        arraylist_free(&direct);
        return t;
    }
    arraylist_new(&seen, 0);
    arraylist_new(&vars, 0);
    arraylist_new(&order, 0);
    jl_value_t *r = NULL, *lb = NULL, *ub = NULL, *body = NULL;
    jl_array_t *fresh = NULL;
    JL_GC_PUSH6(&t, &r, &lb, &ub, &body, &fresh);
    int fail = 0;
    for (i = 0; i < direct.len && !fail; i++)
        wrap_visit(e, start, (jl_value_t*)direct.items[i], &seen, &vars, &fail);
    if (fail) {
        r = NULL;
        goto out;
    }
    fresh = jl_alloc_vec_any(0);
    for (i = 0; i < vars.len; i++) {
        jl_value_t *v = (jl_value_t*)vars.items[i];
        lb = closedize(e, v, 0, 0);
        ub = closedize(e, v, 1, 0);
        if (!is_type(lb) || !is_type(ub)) {
            if (lb != NULL && lb == ub) {
                jl_array_ptr_1d_push(fresh, ub);
                continue;
            }
            if (!is_type(lb))
                lb = BOT;
            if (!is_type(ub))
                ub = ANY;
        }
        if (!issub(lb, ub)) {
            r = BOT;
            goto out;
        }
        int acc[2];
        occurrences(t, (jl_tvar_t*)v, acc);
        int covariant_only = acc[1] == 0 && !(acc[0] > 1 && diag_of(e, v)) && list_index(&direct, v) >= 0;
        if (jeq(lb, ub) || covariant_only) {
            jl_array_ptr_1d_push(fresh, ub);
        }
        else {
            lb = wrap_bound(e, start, v, 0, &vars, fresh, &order);
            ub = wrap_bound(e, start, v, 1, &vars, fresh, &order);
            if (lb == NULL || ub == NULL) {
                r = NULL;
                goto out;
            }
            body = (jl_value_t*)jl_new_typevar(((jl_tvar_t*)v)->name, lb, ub);
            jl_array_ptr_1d_push(fresh, body);
            arraylist_push(&order, (void*)(uintptr_t)i);
        }
    }
    body = t;
    for (i = 0; i < vars.len; i++) {
        jl_value_t *f = AREF(fresh, i);
        jl_value_t *nb = inst(body, (jl_tvar_t*)vars.items[i], f);
        if (nb == NULL) {
            r = f == BOT ? BOT : NULL;
            goto out;
        }
        body = nb;
    }
    for (i = order.len; i > 0; i--) {
        size_t j = (size_t)order.items[i - 1];
        body = jl_type_unionall((jl_tvar_t*)AREF(fresh, j), body);
    }
    r = body;
out:
    arraylist_free(&direct);
    arraylist_free(&seen);
    arraylist_free(&vars);
    arraylist_free(&order);
    JL_GC_POP();
    return r;
}

static int binders(jl_value_t *t, jl_tvar_t *v) JL_NOTSAFEPOINT
{
    if (is_un(t))
        return binders(((jl_uniontype_t*)t)->a, v) + binders(((jl_uniontype_t*)t)->b, v);
    if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        return (u->var == v) + binders(u->var->lb, v) + binders(u->var->ub, v) + binders(u->body, v);
    }
    if (jl_is_some_Type(t))
        return binders(jl_some_Type_T(t), v);
    if (jl_is_vararg(t))
        return binders(va_T(t), v);
    if (jl_is_datatype(t)) {
        int c = 0;
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            c += binders(jl_tparam(t, i), v);
        return c;
    }
    return 0;
}

static void push_var(jl_array_t **vars, jl_value_t *v)
{
    if (*vars == (jl_array_t*)jl_an_empty_vec_any)
        *vars = jl_alloc_vec_any(0);
    jl_array_ptr_1d_push(*vars, v);
}

static jl_value_t *lift_covariant(mctx_t *ctx, jl_value_t *t, jl_array_t **vars, jl_value_t *other, jl_value_t *A, jl_value_t *B)
{
    if (is_ua(t)) {
        jl_tvar_t *v = ((jl_unionall_t*)t)->var;
        int fresh = isopaque(ctx, (jl_value_t*)v) || binders(A, v) + binders(B, v) > 1;
        jl_value_t *u = fresh ? rename_unionall(t) : t;
        JL_GC_PUSH1(&u);
        push_var(vars, (jl_value_t*)((jl_unionall_t*)u)->var);
        jl_value_t *r = lift_covariant(ctx, ((jl_unionall_t*)u)->body, vars, NULL, A, B);
        JL_GC_POP();
        return r;
    }
    if (is_un(t)) {
        int i, n = jl_count_union_components(t), changed = 0;
        jl_value_t **ms;
        JL_GC_PUSHARGS(ms, n);
        for (i = 0; i < n; i++) {
            jl_value_t *m = jl_nth_union_component(t, i);
            ms[i] = lift_covariant(ctx, m, vars, NULL, A, B);
            changed |= ms[i] != m;
        }
        jl_value_t *r = changed ? jl_type_union(ms, n) : t;
        JL_GC_POP();
        return r;
    }
    if (jl_is_tuple_type(t)) {
        size_t i, n = jl_nparams(t), nf = nfixed(t);
        int changed = 0;
        jl_value_t **ps;
        JL_GC_PUSHARGS(ps, n);
        for (i = 0; i < nf; i++) {
            jl_value_t *p = jl_tparam(t, i);
            jl_value_t *o = other != NULL && jl_is_tuple_type(other) && i < jl_nparams(other) ? jl_tparam(other, i) : NULL;
            int same = o != NULL && !jl_is_vararg(o) && (o == p || (isclosed(ctx, p) && isclosed(ctx, o)));
            ps[i] = same ? p : lift_covariant(ctx, p, vars, NULL, A, B);
            changed |= ps[i] != p;
        }
        if (n > nf)
            ps[nf] = jl_tparam(t, nf);
        jl_value_t *r = changed ? tuple_of(ps, n) : t;
        JL_GC_POP();
        return r;
    }
    return t;
}

static int in_list(arraylist_t *a, void *x) JL_NOTSAFEPOINT
{
    size_t i;
    for (i = 0; i < a->len; i++)
        if (a->items[i] == x)
            return 1;
    return 0;
}

static void length_vars(arraylist_t *acc, jl_value_t *t) JL_NOTSAFEPOINT
{
    if (jl_is_vararg(t)) {
        jl_value_t *n = va_N(t);
        if (n != NULL && jl_is_typevar(n) && !in_list(acc, n))
            arraylist_push(acc, n);
        length_vars(acc, va_T(t));
    }
    else if (is_un(t)) {
        int i, n = jl_count_union_components(t);
        for (i = 0; i < n; i++)
            length_vars(acc, jl_nth_union_component(t, i));
    }
    else if (istypeeq(t)) {
        length_vars(acc, jl_typeeq_T(t));
    }
    else if (jl_is_datatype(t)) {
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            length_vars(acc, jl_tparam(t, i));
    }
    else if (is_ua(t)) {
        length_vars(acc, ((jl_unionall_t*)t)->body);
    }
}

static int splittable(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (is_un(t))
        return 1;
    if (!jl_is_tuple_type(t))
        return 0;
    size_t i, nf = nfixed(t);
    for (i = 0; i < nf; i++)
        if (splittable(jl_tparam(t, i)))
            return 1;
    jl_value_t *va = tuple_va(t);
    return va != NULL && va_N(va) == NULL && va_T(va) != ANY;
}

static void expand_cases(jl_array_t *out, jl_value_t *t, jl_array_t *alts, size_t nf, size_t total, jl_value_t *va)
{
    jl_value_t **buf;
    JL_GC_PUSHARGS(buf, nf + 2);
    size_t *idx = (size_t*)alloca(nf * sizeof(size_t));
    size_t c, i;
    memset(idx, 0, nf * sizeof(size_t));
    for (c = 0; c < total; c++) {
        for (i = 0; i < nf; i++) {
            jl_array_t *as = (jl_array_t*)AREF(alts, i);
            buf[i] = as == NULL ? jl_tparam(t, i) : AREF(as, idx[i]);
        }
        if (va != NULL)
            buf[nf] = va;
        buf[nf + 1] = tuple_of(buf, va != NULL ? nf + 1 : nf);
        jl_array_ptr_1d_push(out, buf[nf + 1]);
        for (i = nf; i-- > 0; ) {
            jl_array_t *as = (jl_array_t*)AREF(alts, i);
            if (as == NULL)
                continue;
            if (++idx[i] < LEN(as))
                break;
            idx[i] = 0;
        }
    }
    JL_GC_POP();
}

static void deps(mctx_t *ctx, arraylist_t *acc, jl_value_t *t) JL_NOTSAFEPOINT
{
    size_t i = acc->len;
    free_vars(ctx, acc, t);
    for (; i < acc->len; i++) {
        jl_tvar_t *v = (jl_tvar_t*)acc->items[i];
        free_vars(ctx, acc, v->lb);
        free_vars(ctx, acc, v->ub);
    }
}

static int has_tuple(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (is_un(t))
        return has_tuple(((jl_uniontype_t*)t)->a) || has_tuple(((jl_uniontype_t*)t)->b);
    return jl_is_tuple_type(jl_unwrap_unionall(t));
}

typedef struct {
    jl_value_t *owner;
    int diag, cov, inv;
} varinfo_t;

static int input_diag(jl_array_t *allv, varinfo_t *info, jl_value_t *t) JL_NOTSAFEPOINT
{
    size_t i;
    if (!jl_is_typevar(t))
        return 0;
    for (i = 0; i < LEN(allv); i++)
        if (AREF(allv, i) == t)
            return info[i].diag;
    return 0;
}

static char *independent_slots(mctx_t *ctx, jl_value_t *a, jl_value_t *b, jl_array_t *allv, varinfo_t *info)
{
    if (!(jl_is_tuple_type(a) && jl_is_tuple_type(b)))
        return NULL;
    size_t na = jl_nparams(a), nb = jl_nparams(b), np = na > nb ? na : nb;
    size_t fa = nfixed(a), fb = nfixed(b), n = fa < fb ? fa : fb, i, j, k, l;
    if (n == 0)
        return NULL;
    arraylist_t *ds = (arraylist_t*)malloc_s(np * sizeof(arraylist_t));
    for (i = 0; i < np; i++) {
        arraylist_new(&ds[i], 0);
        if (i < na)
            deps(ctx, &ds[i], jl_tparam(a, i));
        if (i < nb)
            deps(ctx, &ds[i], jl_tparam(b, i));
    }
    char *keep = (char*)malloc_s(n);
    for (i = 0; i < n; i++) {
        jl_value_t *p = jl_tparam(a, i), *q = jl_tparam(b, i);
        if (ds[i].len == 0 || (isclosed(ctx, p) && input_diag(allv, info, q)) ||
            (isclosed(ctx, q) && input_diag(allv, info, p))) {
            keep[i] = 1;
            continue;
        }
        int ok = !has_tuple(p) && !has_tuple(q);
        for (j = 0; ok && j < np; j++) {
            if (j == i)
                continue;
            for (k = 0; ok && k < ds[i].len; k++)
                for (l = 0; ok && l < ds[j].len; l++)
                    if (ds[i].items[k] == ds[j].items[l])
                        ok = 0;
        }
        keep[i] = ok;
    }
    for (i = 0; i < np; i++)
        arraylist_free(&ds[i]);
    free(ds);
    return keep;
}

static jl_array_t *expand(mctx_t *ctx, jl_value_t *t, size_t limit, int *split, jl_value_t *other, const char *keep)
{
    jl_array_t *out = jl_alloc_vec_any(0), *ps = NULL, *sub = NULL, *alts = NULL;
    JL_GC_PUSH4(&out, &ps, &sub, &alts);
    if (is_un(t)) {
        int i, n = jl_count_union_components(t);
        for (i = 0; i < n; i++) {
            sub = expand(ctx, jl_nth_union_component(t, i), limit, split, other, keep);
            size_t j;
            for (j = 0; j < LEN(sub); j++)
                jl_array_ptr_1d_push(out, AREF(sub, j));
        }
        goto done;
    }
    if (jl_is_tuple_type(t) && jl_nparams(t) > 0 && splittable(t)) {
        size_t i, total = 1, nf = nfixed(t);
        size_t no = other != NULL && jl_is_tuple_type(other) ? nfixed(other) : 0;
        jl_value_t *va = tuple_va(t);
        int drop = va != NULL && !(va_N(va) != NULL || va_T(va) == ANY), changed = 0;
        alts = jl_alloc_vec_any(nf);
        for (i = 0; i < nf; i++) {
            jl_value_t *p = jl_tparam(t, i);
            jl_value_t *q = i < no ? upper(jl_tparam(other, i)) : NULL;
            if (!splittable(p) || (q != NULL && keep != NULL && keep[i])) {
                if (q != NULL && nominally_disjoint(upper(p), q))
                    goto done;
                continue;
            }
            ps = expand(ctx, p, limit, split, NULL, NULL);
            if (q != NULL) {
                size_t j, k = 0;
                for (j = 0; j < LEN(ps); j++)
                    if (!nominally_disjoint(upper(AREF(ps, j)), q))
                        jl_array_ptr_set(ps, k++, AREF(ps, j));
                if (k == 0)
                    goto done;
                if (k < LEN(ps))
                    jl_array_del_end(ps, LEN(ps) - k);
            }
            if (LEN(ps) == 1 && AREF(ps, 0) == p)
                continue;
            jl_array_ptr_set(alts, i, ps);
            changed = 1;
            total *= LEN(ps);
            if (total > limit) {
                *split = 0;
                out = jl_alloc_vec_any(0);
                jl_array_ptr_1d_push(out, t);
                goto done;
            }
        }
        if (!changed)
            jl_array_ptr_1d_push(out, t);
        else
            expand_cases(out, t, alts, nf, total, va);
        if (drop)
            expand_cases(out, t, alts, nf, total, NULL);
        goto done;
    }
    jl_array_ptr_1d_push(out, t);
done:
    JL_GC_POP();
    return out;
}

static int only_length(jl_value_t *t, jl_tvar_t *v) JL_NOTSAFEPOINT
{
    if (t == NULL || t == jl_nothing)
        return 1;
    if (t == (jl_value_t*)v)
        return 0;
    if (is_un(t))
        return only_length(((jl_uniontype_t*)t)->a, v) && only_length(((jl_uniontype_t*)t)->b, v);
    if (is_ua(t)) {
        jl_unionall_t *u = (jl_unionall_t*)t;
        return only_length(u->var->lb, v) && only_length(u->var->ub, v) && only_length(u->body, v);
    }
    if (istypeeq(t))
        return only_length(jl_typeeq_T(t), v);
    if (jl_is_vararg(t))
        return only_length(va_T(t), v);
    if (jl_is_datatype(t)) {
        size_t i, n = jl_nparams(t);
        for (i = 0; i < n; i++)
            if (!only_length(jl_tparam(t, i), v))
                return 0;
    }
    return 1;
}

static jl_value_t *peel(jl_value_t *t, jl_tvar_t *v, ssize_t m, jl_tvar_t *w)
{
    if (t == NULL || t == jl_nothing || !(jl_is_type(t) || jl_is_typevar(t)) || !jl_has_typevar(t, v))
        return t;
    jl_value_t *r = NULL;
    if (is_un(t)) {
        jl_value_t *ms[2] = { NULL, NULL };
        JL_GC_PUSH2(&ms[0], &ms[1]);
        ms[0] = peel(((jl_uniontype_t*)t)->a, v, m, w);
        ms[1] = peel(((jl_uniontype_t*)t)->b, v, m, w);
        r = (ms[0] && ms[1]) ? jl_type_union(ms, 2) : NULL;
        JL_GC_POP();
        return r;
    }
    if (is_ua(t)) {
        r = peel(((jl_unionall_t*)t)->body, v, m, w);
        if (r == NULL)
            return NULL;
        JL_GC_PUSH1(&r);
        r = jl_type_unionall(((jl_unionall_t*)t)->var, r);
        JL_GC_POP();
        return r;
    }
    if (istypeeq(t)) {
        r = peel(jl_typeeq_T(t), v, m, w);
        if (r == NULL)
            return NULL;
        JL_GC_PUSH1(&r);
        r = (jl_value_t*)jl_wrap_Type(r);
        JL_GC_POP();
        return r;
    }
    if (!jl_is_datatype(t))
        return t;
    size_t i, n = jl_nparams(t);
    jl_value_t **ps;
    JL_GC_PUSHARGS(ps, n + (size_t)m + 1);
    for (i = 0; i < n; i++) {
        ps[i] = peel(jl_tparam(t, i), v, m, w);
        if (ps[i] == NULL) {
            JL_GC_POP();
            return NULL;
        }
    }
    size_t np = n;
    if (jl_is_tuple_type(t) && n > 0 && jl_is_vararg(jl_tparam(t, n - 1)) && va_N(jl_tparam(t, n - 1)) == (jl_value_t*)v) {
        jl_value_t *T = va_T(jl_tparam(t, n - 1));
        np = n - 1;
        ssize_t j;
        for (j = 0; j < m; j++)
            ps[np++] = T;
        ps[np] = make_va(T, (jl_value_t*)w);
        np++;
    }
    r = rebuild_datatype((jl_datatype_t*)t, ps, np);
    JL_GC_POP();
    return r;
}

static int length_only(menv_t *e, jl_value_t *a, jl_value_t *b, jl_tvar_t *v) JL_NOTSAFEPOINT
{
    if (!only_length(a, v) || !only_length(b, v))
        return 0;
    size_t j;
    for (j = 0; j < NB(e); j++) {
        jl_svec_t *sj = BND(e, j);
        if (!only_length(BLB(sj), v) || !only_length(BUB(sj), v) || (BEQ(sj) != (jl_value_t*)v && !only_length(BEQ(sj), v)))
            return 0;
    }
    return 1;
}

static int rewrite_length(menv_t *e, jl_value_t **pa, jl_value_t **pb, jl_tvar_t *v, ssize_t m, jl_tvar_t *w)
{
    jl_value_t *x = peel(*pa, v, m, w);
    if (x == NULL)
        return 1;
    *pa = x;
    x = peel(*pb, v, m, w);
    if (x == NULL)
        return 1;
    *pb = x;
    JL_GC_PUSH1(&x);
    size_t j;
    for (j = 0; j < NB(e); j++) {
        jl_svec_t *sj = BND(e, j);
        int f;
        for (f = B_LB; f <= B_EQ; f++) {
            x = peel(jl_svecref(sj, f), v, m, w);
            if (x == NULL) {
                JL_GC_POP();
                return 1;
            }
            jl_svecset(sj, f, x);
        }
    }
    JL_GC_POP();
    return 0;
}

static int peel_lengths(menv_t *e, jl_value_t **pa, jl_value_t **pb)
{
    size_t i, n = NB(e);
    jl_tvar_t *w = NULL;
    jl_value_t *r = NULL;
    JL_GC_PUSH2(&w, &r);
    r = subst(e, *pa);
    if (r == NULL) {
        JL_GC_POP();
        return 1;
    }
    *pa = r;
    r = subst(e, *pb);
    if (r == NULL) {
        JL_GC_POP();
        return 1;
    }
    *pb = r;
    int st = 0;
    for (i = 0; i < n && !st; i++) {
        jl_svec_t *s = BND(e, i);
        jl_tvar_t *v = (jl_tvar_t*)jl_svecref(s, B_VAR);
        if (jl_svecref(s, B_OFFTO) == jl_nothing || resolve(e, (jl_value_t*)v) != (jl_value_t*)v)
            continue;
        ssize_t d;
        r = len_root(e, (jl_value_t*)v, &d);
        if (jl_is_long(r)) {
            jl_svecset(s, B_EQ, r);
            e->cver = -1;
        }
        else if (d > 0 && length_only(e, *pa, *pb, v))
            st = rewrite_length(e, pa, pb, v, d, (jl_tvar_t*)r);
    }
    for (i = 0; i < n && !st; i++) {
        jl_svec_t *s = BND(e, i);
        jl_tvar_t *v = (jl_tvar_t*)jl_svecref(s, B_VAR);
        ssize_t m = jl_unbox_long(jl_svecref(s, B_MIN));
        if (m <= 0 || resolve(e, (jl_value_t*)v) != (jl_value_t*)v || jl_svecref(s, B_OFFTO) != jl_nothing)
            continue;
        if (!length_only(e, *pa, *pb, v))
            continue;
        w = jl_new_typevar(v->name, BOT, ANY);
        add_binding(e, w, BOT, ANY, NULL, 0, 0, 0);
        st = rewrite_length(e, pa, pb, v, m, w);
    }
    JL_GC_POP();
    return st;
}

static void mark_inhabited(menv_t *e, jl_value_t *t) JL_NOTSAFEPOINT
{
    if (!jl_is_tuple_type(t))
        return;
    size_t i, n = nfixed(t);
    for (i = 0; i < n; i++) {
        jl_value_t *p = jl_tparam(t, i);
        jl_svec_t *s = jl_is_typevar(p) ? binding(e, p) : NULL;
        if (s != NULL)
            jl_svecset(s, B_INH, jl_true);
        else
            mark_inhabited(e, p);
    }
}

static int intersect_case(mctx_t *ctx, jl_array_t *vars, varinfo_t *info, int *diag, jl_value_t *a, jl_value_t *b, arraylist_t *empty, int split, jl_svec_t **res)
{
    menv_t e;
    e.root = NULL;
    jl_value_t *body = NULL, *r = NULL, *z = NULL;
    JL_GC_PUSH6(&e.root, &body, &r, &z, &a, &b);
    init_env(&e, ctx);
    e.split = split;
    size_t i, n = LEN(vars);
    for (i = 0; i < n; i++) {
        jl_tvar_t *v = (jl_tvar_t*)AREF(vars, i);
        add_binding(&e, v, v->lb, v->ub, info[i].owner, diag[i], info[i].cov, info[i].inv);
        if (v->lb != BOT)
            fact(&e, v->lb, (jl_value_t*)v);
        if (v->ub != ANY)
            fact(&e, (jl_value_t*)v, v->ub);
    }
    mark_inhabited(&e, a);
    mark_inhabited(&e, b);
    add_param_bounds(&e, a);
    add_param_bounds(&e, b);
    int st = 1;
    z = jl_box_long(0);
    for (i = 0; i < empty->len; i++)
        if (bind_eq(&e, (jl_value_t*)empty->items[i], z))
            goto out;
    if (meet(&e, a, b, 1))
        goto out;
    push2(ENV_W(&e, R_MEETS), a, b);
    e.met = e.version;
    if (solve(&e))
        goto out;
    if (peel_lengths(&e, &a, &b))
        goto out;
    body = build(&e, a, b);
    if (body == NULL)
        goto out;
    r = wrap(&e, body, -1);
    if (r == NULL || r == BOT)
        goto out;
    *res = jl_svec(3, e.root, body, r);
    st = 0;
out:
    JL_GC_POP();
    return st;
}

static jl_value_t *subst_or(menv_t *e, jl_value_t *t)
{
    jl_value_t *r = subst(e, t);
    return r == NULL ? t : r;
}

static jl_value_t *rename_vars(jl_value_t *t, jl_array_t *from, jl_array_t *to)
{
    size_t i, n = LEN(from);
    JL_GC_PUSH1(&t);
    for (i = 0; i < n; i++) {
        jl_value_t *r = inst(t, (jl_tvar_t*)AREF(from, i), AREF(to, i));
        if (r != NULL)
            t = r;
    }
    JL_GC_POP();
    return t;
}

static jl_value_t *case_env_bound(mctx_t *ctx, jl_array_t *cases, size_t c, jl_value_t *v, int upper)
{
    menv_t ce;
    ce.root = (jl_svec_t*)jl_svecref(AREF(cases, c), 0);
    ce.ctx = ctx;
    ce.version = -1;
    jl_svec_t *s = binding(&ce, v);
    return subst_or(&ce, upper ? BUB(s) : BLB(s));
}

static jl_value_t *merge_build(mctx_t *ctx, jl_array_t *cases)
{
    size_t i, j, n = LEN(cases);
    jl_value_t *res = NULL, *lb = NULL, *ub = NULL, *x = NULL;
    jl_array_t *freev = NULL, *bodies = NULL, *copies = NULL, *from = NULL;
    menv_t me;
    me.root = NULL;
    JL_GC_PUSH9(&res, &lb, &ub, &x, &freev, &bodies, &copies, &from, &me.root);
    freev = jl_alloc_vec_any(0);
    for (i = 0; i < n; i++) {
        menv_t ce;
        ce.root = (jl_svec_t*)jl_svecref(AREF(cases, i), 0);
        ce.ctx = ctx;
        ce.version = -1;
        size_t k, nb = NB(&ce);
        for (k = 0; k < nb; k++) {
            jl_value_t *v = jl_svecref(BND(&ce, k), B_VAR);
            if (resolve(&ce, v) != v)
                continue;
            size_t f;
            for (f = 0; f < LEN(freev); f += 2)
                if (AREF(freev, f) == v)
                    break;
            if (f == LEN(freev)) {
                x = (jl_value_t*)jl_alloc_vec_any(0);
                push2(freev, v, x);
            }
            x = jl_box_long(i);
            jl_array_ptr_1d_push((jl_array_t*)AREF(freev, f + 1), x);
        }
    }
    size_t nfree = LEN(freev) / 2;
    int *sh = (int*)calloc(nfree + 1, sizeof(int));
    for (i = 0; i < nfree; i++) {
        jl_value_t *v = AREF(freev, 2 * i);
        jl_array_t *is = (jl_array_t*)AREF(freev, 2 * i + 1);
        int same = 1;
        for (j = 0; j < LEN(is) && same; j++) {
            size_t c = jl_unbox_long(AREF(is, j));
            x = case_env_bound(ctx, cases, c, v, 0);
            if (j == 0)
                lb = x;
            else if (!jeq(x, lb))
                same = 0;
            x = case_env_bound(ctx, cases, c, v, 1);
            if (j == 0)
                ub = x;
            else if (!jeq(x, ub))
                same = 0;
        }
        sh[i] = same;
    }
    int changed = 1;
    while (changed) {
        changed = 0;
        for (i = 0; i < nfree; i++) {
            if (!sh[i])
                continue;
            jl_value_t *v = AREF(freev, 2 * i);
            size_t c0 = jl_unbox_long(AREF((jl_array_t*)AREF(freev, 2 * i + 1), 0));
            lb = case_env_bound(ctx, cases, c0, v, 0);
            ub = case_env_bound(ctx, cases, c0, v, 1);
            for (j = 0; j < nfree; j++) {
                jl_tvar_t *w = (jl_tvar_t*)AREF(freev, 2 * j);
                if (!sh[j] && (jl_has_typevar(lb, w) || jl_has_typevar(ub, w))) {
                    sh[i] = 0;
                    changed = 1;
                    break;
                }
            }
        }
    }
    copies = jl_alloc_vec_any(0);
    for (i = 0; i < 2 * n; i++) {
        x = (jl_value_t*)jl_alloc_vec_any(0);
        jl_array_ptr_1d_push(copies, x);
    }
    for (i = 0; i < nfree; i++) {
        if (sh[i])
            continue;
        jl_tvar_t *v = (jl_tvar_t*)AREF(freev, 2 * i);
        jl_array_t *is = (jl_array_t*)AREF(freev, 2 * i + 1);
        for (j = 0; j < LEN(is); j++) {
            size_t c = jl_unbox_long(AREF(is, j));
            x = (jl_value_t*)jl_new_typevar(v->name, BOT, ANY);
            jl_array_ptr_1d_push((jl_array_t*)AREF(copies, 2 * c), (jl_value_t*)v);
            jl_array_ptr_1d_push((jl_array_t*)AREF(copies, 2 * c + 1), x);
        }
    }
    init_env(&me, ctx);
    for (i = 0; i < nfree; i++) {
        jl_tvar_t *v = (jl_tvar_t*)AREF(freev, 2 * i);
        jl_array_t *is = (jl_array_t*)AREF(freev, 2 * i + 1);
        for (j = 0; j < LEN(is); j++) {
            if (sh[i] && j > 0)
                continue;
            size_t c = jl_unbox_long(AREF(is, j));
            menv_t ce;
            ce.root = (jl_svec_t*)jl_svecref(AREF(cases, c), 0);
            ce.ctx = ctx;
            ce.version = -1;
            from = (jl_array_t*)AREF(copies, 2 * c);
            jl_array_t *to = (jl_array_t*)AREF(copies, 2 * c + 1);
            jl_value_t *cv = (jl_value_t*)v;
            if (!sh[i]) {
                size_t q;
                for (q = 0; q < LEN(from); q++)
                    if (AREF(from, q) == (jl_value_t*)v)
                        cv = AREF(to, q);
            }
            lb = rename_vars(case_env_bound(ctx, cases, c, (jl_value_t*)v, 0), from, to);
            ub = rename_vars(case_env_bound(ctx, cases, c, (jl_value_t*)v, 1), from, to);
            int diag;
            if (sh[i]) {
                diag = 1;
                size_t q;
                for (q = 0; q < LEN(is); q++) {
                    menv_t cq;
                    cq.root = (jl_svec_t*)jl_svecref(AREF(cases, jl_unbox_long(AREF(is, q))), 0);
                    cq.ctx = ctx;
                    cq.version = -1;
                    if (!diag_of(&cq, (jl_value_t*)v))
                        diag = 0;
                }
            }
            else {
                diag = diag_of(&ce, (jl_value_t*)v);
            }
            int cov, inv;
            uses_of(&ce, (jl_value_t*)v, &cov, &inv);
            add_binding(&me, (jl_tvar_t*)cv, lb, ub, NULL, diag, cov, inv);
        }
    }
    free(sh);
    bodies = jl_alloc_vec_any(0);
    for (i = 0; i < n; i++) {
        x = rename_vars(jl_svecref(AREF(cases, i), 1), (jl_array_t*)AREF(copies, 2 * i), (jl_array_t*)AREF(copies, 2 * i + 1));
        jl_array_ptr_1d_push(bodies, x);
    }
    x = jl_type_union((jl_value_t**)jl_array_data(bodies, jl_value_t*), LEN(bodies));
    res = wrap(&me, x, -1);
    if (res == NULL) {
        bodies = jl_alloc_vec_any(0);
        for (i = 0; i < n; i++)
            jl_array_ptr_1d_push(bodies, jl_svecref(AREF(cases, i), 2));
        res = jl_type_union((jl_value_t**)jl_array_data(bodies, jl_value_t*), n);
    }
    JL_GC_POP();
    return res;
}

static jl_value_t *merge_cases(mctx_t *ctx, jl_array_t *cases)
{
    size_t i, j, n = LEN(cases);
    jl_array_t *keep = NULL;
    jl_value_t *res = NULL;
    JL_GC_PUSH2(&keep, &res);
    if (n <= 64) {
        keep = jl_alloc_vec_any(0);
        for (i = 0; i < n; i++) {
            jl_value_t *ri = jl_svecref(AREF(cases, i), 2);
            int covered = 0;
            for (j = 0; j < n && !covered; j++) {
                if (j == i)
                    continue;
                jl_value_t *rj = jl_svecref(AREF(cases, j), 2);
                if (issub(ri, rj) && (j < i || !issub(rj, ri)))
                    covered = 1;
            }
            if (!covered)
                jl_array_ptr_1d_push(keep, AREF(cases, i));
        }
        cases = keep;
    }
    res = LEN(cases) == 1 ? jl_svecref(AREF(cases, 0), 2) : merge_build(ctx, cases);
    JL_GC_POP();
    return res;
}

static jl_value_t *intersect_cases(mctx_t *ctx, jl_array_t *avars, jl_value_t *a, jl_array_t *bvars, jl_value_t *b)
{
    jl_array_t *allv = NULL, *as = NULL, *bs = NULL, *cases = NULL;
    jl_svec_t *res = NULL;
    JL_GC_PUSH5(&allv, &as, &bs, &cases, &res);
    jl_value_t *r = BOT;
    size_t i, j, k, q, na_vars = LEN(avars), nv = na_vars + LEN(bvars);
    allv = nv == 0 ? (jl_array_t*)jl_an_empty_vec_any : jl_alloc_vec_any(nv);
    varinfo_t *info = (varinfo_t*)malloc_s((nv + 1) * sizeof(varinfo_t));
    int *diag = (int*)malloc_s((nv + 1) * sizeof(int));
    arraylist_t lens, empty;
    arraylist_new(&lens, 0);
    arraylist_new(&empty, 0);
    for (k = 0; k < nv; k++) {
        int side = k >= na_vars;
        jl_array_t *vars = side ? bvars : avars;
        jl_value_t *body = side ? b : a;
        jl_tvar_t *v = (jl_tvar_t*)AREF(vars, side ? k - na_vars : k);
        int acc[2], occ[2];
        occurrences_in(vars, body, v, acc);
        occurrences(body, v, occ);
        jl_value_t *owner = NULL;
        if (occ[0] == 0 && occ[1] == 0) {
            jl_value_t *o = NULL;
            int count = 0;
            for (j = 0; j < LEN(vars); j++) {
                jl_tvar_t *w = (jl_tvar_t*)AREF(vars, j);
                if (w != v && (jl_has_typevar(w->lb, v) || jl_has_typevar(w->ub, v))) {
                    count++;
                    o = (jl_value_t*)w;
                }
            }
            if (count == 1)
                owner = o;
        }
        info[k].owner = owner;
        info[k].diag = acc[0] > 1 && acc[1] == 0;
        info[k].cov = acc[0];
        info[k].inv = acc[1];
        jl_array_ptr_set(allv, k, (jl_value_t*)v);
    }
    length_vars(&empty, a);
    length_vars(&empty, b);
    for (i = 0; i < empty.len; i++)
        if (in_array(avars, (jl_value_t*)empty.items[i]) || in_array(bvars, (jl_value_t*)empty.items[i]))
            arraylist_push(&lens, empty.items[i]);
    for (i = 0; i < lens.len; i++)
        if (!in_array(ctx->lengths, (jl_value_t*)lens.items[i])) {
            if (ctx->lengths == (jl_array_t*)jl_an_empty_vec_any)
                ctx->lengths = jl_alloc_vec_any(0);
            jl_array_ptr_1d_push(ctx->lengths, (jl_value_t*)lens.items[i]);
        }
    int split = lens.len <= 4;
    if (!split)
        lens.len = 0;
    int any_split = splittable(a) || splittable(b);
    if (any_split) {
        char *keep = independent_slots(ctx, a, b, allv, info);
        as = expand(ctx, a, 128, &split, b, keep);
        bs = expand(ctx, b, 128, &split, a, keep);
        free(keep);
    }
    cases = jl_alloc_vec_any(0);
    size_t nl = lens.len, na = any_split ? LEN(as) : 1, nbs = any_split ? LEN(bs) : 1, mask;
    for (mask = 0; mask < ((size_t)1 << nl); mask++) {
        for (i = 0; i < na; i++) {
            for (j = 0; j < nbs; j++) {
                jl_value_t *ai = any_split ? AREF(as, i) : a, *bi = any_split ? AREF(bs, j) : b;
                empty.len = 0;
                for (q = 0; q < nl; q++)
                    if ((mask >> q) & 1)
                        arraylist_push(&empty, lens.items[q]);
                for (q = 0; q < nv; q++) {
                    jl_tvar_t *v = (jl_tvar_t*)AREF(allv, q);
                    diag[q] = info[q].diag && (q < na_vars ? is_diagonal(avars, ai, v) : is_diagonal(bvars, bi, v));
                }
                res = NULL;
                if (!intersect_case(ctx, allv, info, diag, ai, bi, &empty, split, &res))
                    jl_array_ptr_1d_push(cases, (jl_value_t*)res);
                if (ctx->unsupported)
                    goto out;
            }
        }
    }
    r = LEN(cases) == 0 ? BOT : merge_cases(ctx, cases);
out:
    arraylist_free(&lens);
    arraylist_free(&empty);
    free(info);
    free(diag);
    JL_GC_POP();
    return r;
}

static jl_value_t *intersect_open(mctx_t *ctx, jl_value_t *A, jl_value_t *B)
{
    jl_array_t *avars = NULL, *bvars = NULL;
    jl_value_t *a = NULL, *b = NULL, *x = NULL;
    JL_GC_PUSH6(&avars, &bvars, &a, &b, &x, &B);
    avars = bvars = (jl_array_t*)jl_an_empty_vec_any;
    a = A;
    while (is_ua(a)) {
        push_var(&avars, (jl_value_t*)((jl_unionall_t*)a)->var);
        a = ((jl_unionall_t*)a)->body;
    }
    size_t i;
    for (i = 0; i < LEN(avars); i++) {
        if (jl_has_typevar(B, (jl_tvar_t*)AREF(avars, i))) {
            B = rename_unionall(B);
            break;
        }
    }
    b = B;
    while (is_ua(b)) {
        push_var(&bvars, (jl_value_t*)((jl_unionall_t*)b)->var);
        b = ((jl_unionall_t*)b)->body;
    }
    x = lift_covariant(ctx, a, &avars, b, A, B);
    b = lift_covariant(ctx, b, &bvars, a, A, B);
    a = x;
    jl_value_t *r = intersect_cases(ctx, avars, a, bvars, b);
    JL_GC_POP();
    return r;
}

static void memo_store(mctx_t *ctx, jl_value_t *A, jl_value_t *B, jl_value_t *r)
{
    JL_GC_PUSH1(&r);
    if (ctx->memo == (jl_array_t*)jl_an_empty_vec_any)
        ctx->memo = jl_alloc_vec_any(0);
    push2(ctx->memo, A, B);
    jl_array_ptr_1d_push(ctx->memo, r);
    JL_GC_POP();
}

static jl_value_t *meet_or_nothing_(mctx_t *ctx, jl_value_t *A, jl_value_t *B, int check)
{
    size_t i, n;
    int closed = isclosed(ctx, A) && isclosed(ctx, B);
    if (closed) {
        n = LEN(ctx->memo);
        for (i = 0; i < n; i += 3)
            if (jl_egal(AREF(ctx->memo, i), A) && jl_egal(AREF(ctx->memo, i + 1), B))
                return AREF(ctx->memo, i + 2);
    }
    if (nominally_disjoint(A, B))
        return BOT;
    if (check) {
        jl_value_t *r = issub(A, B) ? A : issub(B, A) ? B : NULL;
        if (r != NULL) {
            if (closed)
                memo_store(ctx, A, B, r);
            return r;
        }
    }
    jl_value_t *la, *ua, *lb, *ub;
    if (type_range(ctx, A, &la, &ua) && type_range(ctx, B, &lb, &ub)) {
        jl_value_t *l = NULL, *u = NULL, *r = NULL;
        JL_GC_PUSH3(&l, &u, &r);
        jl_value_t *ls[2] = { la, lb };
        l = jl_type_union(ls, 2);
        u = meet_closed(ctx, ua, ub);
        if (!issub(l, u)) {
            r = BOT;
        }
        else if (issub(u, l)) {
            r = (jl_value_t*)jl_wrap_Type(l);
        }
        else {
            jl_value_t *src = is_ua(A) ? A : B;
            r = (jl_value_t*)jl_new_typevar(((jl_unionall_t*)src)->var->name, l, u);
            u = (jl_value_t*)jl_wrap_Type(r);
            r = jl_type_unionall((jl_tvar_t*)r, u);
        }
        JL_GC_POP();
        return r;
    }
    if (closed && is_atom(ctx, A) && is_atom(ctx, B)) {
        int k;
        for (k = 0; k < 2; k++) {
            jl_value_t *s = k == 0 ? A : B, *kd = k == 0 ? B : A;
            if ((istypeeq(s) || isegal(s)) && !(istypeeq(kd) || isegal(kd)))
                return in_kind(ctx, s, kd) ? s : BOT;
        }
        return BOT;
    }
    n = ctx->active.len;
    for (i = 0; i < n; i += 2) {
        jl_value_t *x = (jl_value_t*)ctx->active.items[i], *y = (jl_value_t*)ctx->active.items[i + 1];
        if ((jeq(x, A) && jeq(y, B)) || (jeq(x, B) && jeq(y, A))) {
            ctx->cut++;
            return NULL;
        }
    }
    int cut = ctx->cut;
    arraylist_push(&ctx->active, A);
    arraylist_push(&ctx->active, B);
    jl_value_t *r = intersect_open(ctx, A, B);
    ctx->active.len -= 2;
    if (closed && ctx->cut == cut && r != NULL)
        memo_store(ctx, A, B, r);
    return r;
}

static jl_value_t *meet_or_nothing(mctx_t *ctx, jl_value_t *A, jl_value_t *B)
{
    return meet_or_nothing_(ctx, A, B, 1);
}

static jl_value_t *type_meet(jl_value_t *A, jl_value_t *B, int check)
{
    mctx_t ctx;
    ctx.opaque = ctx.memo = ctx.lengths = ctx.implied = ctx.bounds = NULL;
    ctx.cut = 0;
    ctx.unsupported = 0;
    ctx.work = 0;
    jl_value_t *r = NULL;
    JL_GC_PUSH8(&ctx.opaque, &ctx.memo, &ctx.lengths, &ctx.implied, &ctx.bounds, &A, &B, &r);
    ctx.opaque = ctx.lengths = ctx.memo = (jl_array_t*)jl_an_empty_vec_any;
    arraylist_new(&ctx.active, 0);
    if (jl_has_free_typevars(A) || jl_has_free_typevars(B)) {
        arraylist_t fv;
        arraylist_new(&fv, 0);
        free_vars(&ctx, &fv, A);
        free_vars(&ctx, &fv, B);
        size_t i;
        if (fv.len > 0)
            ctx.opaque = jl_alloc_vec_any(0);
        for (i = 0; i < fv.len; i++)
            jl_array_ptr_1d_push(ctx.opaque, (jl_value_t*)fv.items[i]);
        arraylist_free(&fv);
    }
    int k;
    for (k = 0; k < 2 && LEN(ctx.opaque) > 0; k++) {
        jl_value_t *t = k == 0 ? A : B;
        int hit = 0;
        while (is_ua(t)) {
            if (isopaque(&ctx, (jl_value_t*)((jl_unionall_t*)t)->var))
                hit = 1;
            t = ((jl_unionall_t*)t)->body;
        }
        if (hit) {
            if (k == 0)
                A = rename_unionall(A);
            else
                B = rename_unionall(B);
        }
    }
    r = meet_or_nothing_(&ctx, A, B, check);
    if (ctx.unsupported)
        r = NULL;
    arraylist_free(&ctx.active);
    JL_GC_POP();
    return r;
}

JL_DLLEXPORT jl_value_t *jl_type_meet(jl_value_t *A, jl_value_t *B)
{
    return type_meet(A, B, 1);
}

jl_value_t *jl_type_meet_unrelated(jl_value_t *A, jl_value_t *B)
{
    return type_meet(A, B, 0);
}

#ifdef __cplusplus
}
#endif
