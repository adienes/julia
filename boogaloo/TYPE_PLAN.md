# Type intersection: design plan

Where the `type-boogaloo` branch stands, and the deeper changes that would make
`subtype.c` simpler and easier to trust. Several fixes on the branch patch symptoms
of the same few design problems; the proposals below target those problems.

## Current state (2026-09-28)

Commits on top of the original branch:

| commit | what |
|---|---|
| `c560c6c295` | build fix: `jl_gc_wb` slot argument (#62737) |
| `5f0dbbcbab` | consolidate constraint collection, drop private bounds and no-effect mechanisms |
| `a886d25c53` | invariant fallback keeps a parameter unless the equality forces a closed result |
| `8dbf757896` | same check for open results when a parameter is a union |
| `d5b03d30a8` | seeded soundness property test (`test/testhelpers/IntersectionWitnesses.jl`) |
| `2a10d240b6` | #63155 adopted: resolve dependent static-parameter bounds before classifying them (fixes the `invalid_bound` / `pinned` / `widened` dispatch cases) |

Measurements (witness fuzzer, 20k iterations, 9,537 valid pairs):

| | master | original branch | now |
|---|---|---|---|
| pairs whose result misses a type in both inputs | 112 | 56 (+1 new) | 11 (0 new) |
| results a subtype of neither input | 85 | 22 | 31 |
| Base/Core signature census diffs vs master | — | 128 tighter | 128 tighter |
| C lines vs master | — | −17 | −31 |

Precision cost of the current fixes: 92 of 18,850 fuzz results are looser than a
correct master answer. 26 further "looser" results were master bugs the fuzzer's
witnesses had missed.

## Validation tooling (`boogaloo/`, untracked)

- `fuzz2.jl N SEED OUT`: random pairs where B generalizes a witness of A; records
  results and any witness `W` with `W <: A`, `W <: B`, `!(W <: A ∩ B)`.
- `fcompare.jl`, `loosecheck.jl`, `loose2.jl`: compare fuzz runs (soundness, precision,
  hidden master bugs).
- `census.jl`, `census_check.jl`: intersections of all method-signature pairs in the
  same Base/Core function (~113k pairs), timing and diffs against master.
- `snap.jl save|check`: exact snapshot of ~235k result strings, for behavior-preserving
  refactors.
- `minimize.jl`: shrinks failing triples to minimal shapes.
- `ablate.sh MASK`: scores a temporary `JL_ABLATE` bitmask build on tests, fuzz, census.
- Master reference binary: scratchpad `boogaloo-base` worktree at `10e19b824d`.

## Principles

1. **Direction is part of every operation's contract.** Each helper either
   over-approximates, under-approximates, or is exact. Most unsoundness came from
   using a result in a position that needed the other direction (e.g. an approximate
   value in an invariant slot).
2. **Invariant positions are exact or variable.** A covariant slot can hold an
   over-approximation; an invariant slot cannot, because it names one parameter value.
3. **One notion of a variable's admissible values.** The set of values a variable may
   take is different from the type written into a covariant result slot.
4. **No pass should see state from a discarded pass.**

## Proposals

### 1. Range variables for unforced invariant slots

**Problem.** When `x == y` is not fixed by the equality, the fallback now returns one
of the two parameters. Which one is arbitrary (`y` if it has free variables), and it is
a source of results that differ between `A ∩ B` and `B ∩ A`.

**Proposal.** Return a fresh variable `R` with `lo <: R <: hi`, where `lo` joins the
least types `x` and `y` can denote (already computed by `invariant_admits_least`)
and `hi` meets their greatest types. `bound_var_below` already creates fresh result
variables for `Vararg` lengths through `push_innervar`.

**Removes.** The side choice and the order dependence it causes.

**Risk.** A range variable loses correlations with variables of the returned side
that also occur elsewhere; this may cost precision where the side approach keeps it.

**Status.** Tried 2026-09-28, not adopted. A prototype (`invariant_range`: fresh
`jl_new_typevar(name, lo, hi)` pushed with `push_innervar(e->vars, ...)`, `lo`/`hi`
from bound substitution and `jl_type_intersection` of the upper values) gave, on the
20k fuzz run: unsound pairs unchanged (11), results a subtype of neither input
31 → 58, one `test/subtype.jl` failure, and argument-order symmetry slightly worse
(9,346 vs 9,355 of 9,538 pairs with `A ∩ B == B ∩ A`). It is tighter and symmetric on
specific shapes (a union of two variables against `Union{Nothing,S}`), but the side
choice is not what drives order dependence; that comes from scope order (see 6).

### 2. Separate admissible range from result-slot approximation

**Problem.** A binding's `lb`/`ub` sometimes mean "values the variable may take" and
sometimes "type to write in a covariant result slot" (`constraintkind == 1`). The
branch tells them apart with `required_upper`, `upper_is_slot`,
`required_upper_bound`, and nearby with `lb_required`, `lb_certainty`,
`bound_channel`, `spell_channel`. The reintersection stale-pin bug came from reading
one meaning where the other was stored.

**Proposal.** Explicit fields for the admissible range and for the slot
approximation; every update names which one it changes.

**Removes.** Most of the tags above and the conditionals that interpret them.

**Risk.** Medium; many touch points in `intersect_var` and `finish_bound_env`.

**Audit (2026-09-28).** `ub` doubles as the slot value only for `constraintkind == 1`
variables with a single covariant occurrence: `intersect_var` narrows `ub` to
`var.ub ∩ type` and returns the variable (also `set_bound(&bb->ub, ...)` on the
`e->triangular` path). Readers that need the admissible range instead go through
`required_upper_bound` (`closed_upper_bound`, `constrain_var`, the union
disjointness shortcut in `constrain_subtype`) or check `upper_is_slot` (pinned
unfolding in `constrain_subtype`, `constrain_var`). `required_upper` is set by
`constrain_var` and the two pins in `constrain_equal`, saved/restored per variable,
AND-merged in `merge_env`, compared in `env_unchanged`, and read by the
`constraintkind == 1` early return in `intersect_var` and the reintersection decision
in `intersect_unionall`.

**First step.** Replace `required_upper` and `upper_is_slot` with an explicit
admissible upper bound per binding (`rub`, initially `var->ub`): required narrowing
(`constrain_var`, `constrain_equal` pins) updates `rub` and `ub`; slot narrowing
updates only `ub`. `required_upper_bound(v)` becomes `v->rub`; the flag tests become
`rub != var->ub`. Costs one more saved root per variable (save/restore/merge layout
goes from 3 to 4 roots). Not behavior-preserving; validate with fuzz, census, tests.
Later steps: move slot narrowing to its own field so `ub` means only the admissible
range, then drop the `constraintkind == 1` special cases that re-derive it.

**Finding (2026-09-28).** The first step would not simplify: `upper_is_slot` is
decided at read time from occurrence counts (`cov_count < 2`, `concrete`) because a
slot narrowing becomes a real constraint once a second covariant occurrence appears,
so a separate field still needs the same conditional. Deleting `required_upper`
outright changes only 2 of ~235k snapshot results, but fails 6 test lines, including
triangular cases (`Tuple{X, Ref{U}} where {U, X<:U}` against `Tuple{Int8, Ref{Int}}`)
where master correctly returns `Union{}`: the flag is load-bearing. Blocked on the
two-phase design below.

### 3. Occurrence analysis before intersection, not reintersection

**Problem.** `intersect_unionall` learns `constraintkind` from occurrence counts
during the pass, and reruns the body when it changes, keeping outer bounds from the
first pass (`restore_env(..., 0)`). Outer bindings can then reference variables that
the second pass renames.

**Proposal.** Compute occurrence facts for both inputs up front (the position-aware
`has_typevar` walker, `var_occurs_invariant`, `jl_always_occurs_covariant` already do
most of this) and choose each variable's treatment once.

**Removes.** Reintersection and its stale-state bugs.

**Risk.** The diagonal rule depends on union branches; a static analysis can only
approximate it.

**Finding (2026-09-28).** Probe over the census and a 2k fuzz run: 648k variable
bindings, 1,721 re-intersections (0.27%). Structural counts predict the final
`constraintkind` for 75% of bindings; nearly all disagreements are variables that
only occur invariantly, where the kind does not matter, and ~200 re-intersections are
triggered by one invariant occurrence being counted twice (the equality is checked in
both directions). But choosing the kind statically and skipping the occurrence-driven
re-run changes 187 snapshot results, e.g. `Tuple{typeof(rem), T, Type{T}} where
T<:Integer` against `Tuple{typeof(rem), Signed, Type{Unsigned}}` loses its `Union{}`.
Re-intersection is also backward propagation: tuple elements are processed left to
right, and the re-run (which keeps bounds, `restore_env(..., 0)`) is how `Type{T}`
pinning `T` in a later slot reaches an earlier slot. Not viable as written; see the
two-phase design below.

### 4. One solver: join over successful witnesses

**Problem.** Constraint collection is a second, partial solver inside `subtype()`
with about ten "unsupported" exits.

**Evidence.** Replacing it with "retain the join of the bounds of every successful
union choice" (`exists_subtype` + `merge_env`) cut unsound pairs from 18 to 4 in a
5k run, but lost precision (81 looser census rows) and produced a result that
inference trusted wrongly. The failures traced to per-binding state (occurrences,
certainty, spelling) that does not merge cleanly; see `boogaloo/stepA_plus_joined.patch`.

**Proposal.** After 2 and 3 shrink binding state, revisit the join formulation.

**Removes.** `constrain_subtype`, `constrain_equal`, `constrain_owned_equal`,
`constraint_identity`, `constrain_unfolded`, `e->constraints`.

**Risk.** Large; depends on 2 and 3.

### 5. Meets as binding conjuncts, not `Intersect` values

**Problem.** `Intersect` is type-shaped and can escape into code expecting types;
the removed private-bounds subsystem existed only to contain it.

**Proposal.** A binding holds a short list of upper-bound conjuncts, widened only
when written out.

**Risk.** Medium; substitution must handle the list.

**Finding (2026-09-28).** Since private bounds were removed, `Intersect` only lives at
the top of a binding's upper bound and is widened in three places when written out;
`subtype` and `intersect` handle it structurally (~50 references). Treating it as an
internal constructor is reasonable; changing the representation is low value.

### 6. Symmetric variables in intersection

**Problem.** Intersection inherits subtyping's left/right asymmetry (`R` flips,
`flip_vars`, `var_outside`); many results differ between argument orders.

**Proposal.** All variables existential during intersection, with a scope order that
does not depend on argument order.

**Risk.** Highest; invasive.

### 7. `Vararg` lengths as integer constraints

**Problem.** Length variables are integers encoded as type variables with `Loffset`
bookkeeping; substitution cannot express `N = N' + 1`, which forced special cases
(including in the new invariant check).

**Proposal.** Separate integer constraints with explicit offsets.

**Risk.** Medium; independent of the rest.

### 8. Soundness as a tested property

**Problem.** The witness check found every real bug this round, including hidden
master bugs, while the branch added about 30 tests each pinning one exact result.

**Proposal.** A small seeded property test in `test/subtype.jl`, and a debug-build
assertion that spot-checks random witnesses.

**Risk.** Small.

## Revised direction: two-phase intersection

The findings on 2, 3 and 4 share one cause. `intersect` discovers variable constraints
and builds the result in the same left-to-right pass, and patches the consequences:

- re-intersection re-runs a variable's body to push later constraints into earlier
  slots, keeping first-pass state (and with it, stale variables);
- `ub` doubles as a slot value because the result for a covariant slot is built before
  the variable's final constraints are known, which is what `required_upper` /
  `upper_is_slot` untangle;
- constraint collection is a second solver because the first one commits to results
  while still collecting constraints.

The long-term design is two phases: first solve every variable's bounds over the whole
structure (a fixpoint, with union choice points enumerated and merged as
`intersect_all` + `merge_env` already do), then build the result by instantiating the
solved bounds (`finish_bound_env` already does this part per scope). Pieces that map
onto it: constraint collection is phase 1 for required relations; `finish_bound_env`
and `jl_instantiate_with_typeenv` are phase 2; re-intersection is an approximation of
the fixpoint.

A realistic first step is a phase-1 prototype for a restricted fragment (tuples,
invariant datatype parameters, unions, no `Vararg`, no `Type{}`), run alongside the
current algorithm and compared with the fuzzer and census, before deciding whether to
replace anything.

### Prototype (`boogaloo/TwoPhase.jl`, 2026-09-29)

About 960 lines of Julia; snapshots `TwoPhase.v*.jl`. All variables of both inputs are
unknowns (no left/right roles).

Phase 1 (`relate!`, `equate!`, `sub!`, `solve!`): invariant positions give equalities
(unification, necessary member conditions for union equalities, tuple equalities with
`Vararg` lengths); declared bounds and constructor parameter bounds give subtype
relations; a covariant slot holding a concrete leaf requires the variable to contain it;
two diagonal variables meeting in a slot are equal. Covariant `where`s are lifted to
unknowns (exact outside `Vararg` elements). Relations propagate to a fixpoint; anything
not handled is skipped, which only loosens.

Phase 2 (`build`, `wrap`): covariant slots are computed from solved bounds; a `where`
that cannot be lifted (inside a `Vararg` element) is solved locally (`build_local`),
dropping what the local solve learns about outer variables. `wrap` substitutes pinned
variables, turns covariant-only variables into their upper bound unless they were
diagonal, and keeps variables that only occur in kept bounds.

Disjunctions are cases, solved separately and merged like `intersect_all` + `merge_env`:

- unions in covariant positions (open or closed) are split; the diagonal rule applies
  only to variables diagonal both in the input and in the branch;
- a possibly empty `Vararg` constrains its element only when nonempty, so a length
  variable is solved pinned to 0 and free, and an unbounded `Vararg` dropped and kept;
- cases contained in another case are dropped; the rest share one `where`, with a
  variable copied per case when its bounds differ between cases.

Comparison with the current C algorithm:

| | prototype | C |
|---|---|---|
| fuzz pairs with a missed witness (3 seeds, 2,823 pairs) | 0 | 5 |
| fuzz pairs with `A ∩ B == B ∩ A` | 2,823 (100%) | 2,762 (97.8%) |
| fuzz results vs C: same / tighter / looser / incomparable | 96.0% / 3.3% / 0.3% / 0.4% | |
| Base/Core signature pairs supported | 113,645 of 113,724 (99.93%) | |
| those vs C, both orders: same / looser / tighter / incomparable | 99.87% / 246 / 25 / 16 | |
| symmetric on Base pairs | 100% | 99.98% |
| census rows where C is not below the prototype: witnesses missed | 0 of 15 | |

Remaining looser families on Base: C proves emptiness through `Type{<:X{T}}` against a
closed `UnionAll` in more shapes; the meet of an open bound with a closed one
(`KeySet{<:Any, <:Dict}` gives `T<:Dict` where C keeps `Dict{K}`); dependent bounds
through open `UnionAll` bounds (`T1<:CodeUnits{T}`). Some rows are C bugs: `unsafe_convert`
`(Type{T}, T)` against `(Type{Ref{T}}, Ref{NTuple{N,T}})` is empty in C but contains
`N = 0`; `Tuple{Vector{T}, Vararg{T}}` against `Tuple{Vector, Vararg{Int}}` misses
`Tuple{Vector{String}}` in C. Both are wrong in every release from 1.6 to nightly, have no
GitHub issue (closest: #32392, a `Vararg` length variable giving `Union{}`, closed 2019),
and no test on this branch. The `unsafe_convert` one has no dispatch consequence (the
`NTuple` method is more specific and covers the overlap). Open issues the prototype
also fixes: #63345 (diagonal variable whose other occurrence is in a lifted bound),
#40407 (result not below an input).

Findings that constrain the C design:

- **Results must be decidable by the subtype algorithm that consumes them.** C's `<:` is
  not transitive on some shapes: `W == X` and `X <: R` but `W <: R` false when `R` is a
  union of separately wrapped `where`s and `W` has a covariant union, and when a bound is
  a union of other existential variables (`S<:Union{Y,Z}`). Cases therefore merge under
  one `where`, and bounds adopted from constraints are data types or `where`s, never
  unions of variables. The per-case copies keep precision where `merge_env` joins.
- **`Vararg` emptiness is a disjunction.** Element constraints are conditional on the
  `Vararg` having elements; treating them as unconditional misses `N = 0` members
  (34 census rows before the split).
- **Local solves must not leak.** Variables introduced inside a local solve are wrapped
  there, including ones introduced by lifting during the local relation.
- `subst` before storing an equality: an occurs check on the substituted value with an
  unsubstituted stored value creates cycles once a length is pinned.

Not covered: 79 Base pairs (mostly `Tuple` equalities with different fixed lengths and an
unbounded `Vararg` against a length variable); envout (static parameter values); the
`Type{}` kinds beyond `TypeEq`; performance (the prototype calls C `<:` and `typeintersect`
on closed types and checks case containment with `<:`).

Next: the C implementation, as a replacement for `intersect_all`'s body. The prototype's
structure maps onto existing pieces: case enumeration is `Runions`/`Lunions` state,
`merge_cases` is `merge_env` (with per-case copies instead of joins), `wrap` is
`finish_unionall`/`finish_bound_env`, and phase 1 replaces the in-walk bound updates and
reintersection. Before that, a validation mode that runs the prototype's algorithm next to
C in debug builds would give broader coverage than the census (package signatures).

### Static parameter values (`TwoPhase.intersect_env`, 2026-09-29)

C does not use static parameter values from the intersection itself (except when `b` is
`Type`; the source has a TODO to compute them directly). `jl_type_intersection_env_s`
intersects, then matches the result against `b` with `jl_subtype_env` and reads the values
(with the `svec(tvar, constrained)` markers) off that match; if the match fails it falls
back to `b`'s variables as unconstrained markers. The hacks are around the result, not the
values: for tuple inputs a union result goes through `switch_union_tuple` (two members
differing in one slot) or is replaced by `b` itself, for `typeintersect` too.

The two-phase design keeps that split: parameter values stay on the subtype side, and the
intersection only has to deliver a result that is one tuple and below `b`. What made that
hold on Base:

- **Case merging.** Cases that differ in one slot merge into a union in that slot, pairwise
  until none do (exact); cases that differ in several slots holding the same concrete type
  merge into one diagonal variable bounded by the union of those types (exact, C's shape
  for `Tuple{/, T, T}` against unions of integer types). Union results for tuple inputs:
  414 → 10.
- **Meets of bounds.** A variable with two bounds, at least one open, gets their meet from
  a local solve over both bounds' `where`s (`meet_bound`), instead of the closed one. What
  the local solve learns about a variable that occurs only in this variable's bounds is
  kept (it cannot matter when the variable is empty); other facts are dropped. Bounds that
  are unions of data types are adopted; unions with a bare variable member are not (C's
  `<:` cannot decide them).
- **Quantifier order.** In `lo <: hi` with a `where` on the left, each instance of the left
  is checked before the right's `where` is lifted, so the right's variables are per
  instance.

Results on Base signature pairs, both orders (19,246 nonempty):

| | count |
|---|---|
| same parameter-value grades as C (defined / constrained / free per slot) | 19,219 |
| prototype better | 23 |
| prototype worse | 4 (2 are a C `N = 0` bug; 2 need a variable shared by two bounds unified) |
| results still a union (fall back to `b`, as C) | 10 |
| results not below both inputs | 76 (40 identical to C: the meet of a variable and a type, `N - 1` lengths, `DataType ∩ Type{<:Tuple}`) |

Census after these changes: 99.96% identical to C, 38 looser, 39 tighter, 14 incomparable;
fuzz and witness checks clean.

For the C port this means: no `envout` code in the intersection, the result contract is
"one tuple below `b` for tuple inputs", and the fallback to `b` stays only for results that
cannot be one tuple.

## Intended semantics (2026-09-30)

Differences from C are allowed when they follow from a stated rule. Every row where the
prototype differs from C must fall into a named class below; rows that do not are bugs.
`boogaloo/classify.jl` does the attribution (used by `tp_classify.jl` on the census and
`tp_fuzzclass.jl` on the fuzzer).

### Invariants (checked)

- **I1 Soundness relative to `<:`.** `W <: A && W <: B` implies `W <: A ∩ B`, with `<:` the
  runtime's subtype. That subtype ignores declared parameter bounds, so `W` includes
  spellings such as `Broadcasted{<:Any, <:Any, <:Any, <:Any}` whose variables are not
  narrowed by them. Checked by fuzz witnesses, by instances of C's result (classifier
  class `tp_unsound`), and on logged runtime queries.
- **I2 Symmetry.** `A ∩ B == B ∩ A`.
- **I3 Decidable results.** Results use forms the runtime's `<:` decides: one `where` over
  all cases (never a union of separately wrapped `where`s), and no variable bounded by a
  union with a bare-variable member.
- **I4 Method-matching contract.** For tuple inputs the env entry point returns one tuple
  (cases merged exactly, otherwise `b`); static parameter values come from matching the
  result against `b`.
- **I5** No free variables in a result unless the inputs have them.

### Deliberate differences from C

- **D1 Declared parameter bounds decide emptiness only; they never narrow a result.** C
  narrows in some paths (`Rational{Tr}` gets `Tr<:Integer` through the intersection walk)
  and not in others (the subtype fast paths return the input unnarrowed). Narrowing breaks
  I1 for types spelled without the bound (found on logged runtime queries). Effect: rows
  where the prototype is looser by `<:` and equal on every valid instance (class
  `looser_declared`, checked by narrowing the prototype's result by the declared bounds
  and testing it against C's).
- **D2 `Vararg` emptiness is a case split.** Element constraints hold only when the
  `Vararg` has elements; a length variable is solved pinned to 0 and free, an unbounded
  `Vararg` dropped and kept. Where C drops the `N = 0` members, the prototype keeps them
  (class `looser_c_unsound`: C's result misses a type in both inputs).
- **D3 Diagonal rule.** A variable is diagonal only if it is diagonal in its input and in
  the union branch being solved; occurrences in other variables' bounds count as
  invariant (#63345).
- **D4 Local solves** (a `Vararg` element, a union member, the meet of two bounds) decide
  whether their part is empty and resolve their own variables. An outer variable may be
  specialized to a closed value in the part's result; it is never re-expressed through
  another outer variable, and a diagonal one is never pinned. Facts about outer variables
  are kept only for a variable that occurs solely in the bounds being met.
- **D5 Unions in a certain slot** constrain by what all feasible members imply: no feasible
  member makes the slot empty, one makes it that member, several commit the facts they
  share. A slot is certain when every member of the result has it (fixed tuple slots,
  including those matched against the other side's `Vararg`); a `where` may only become an
  unknown of the whole type in a certain slot that is not a `Vararg` element.
- **D6 Cases merge exactly** (a union in the one slot where they differ; one diagonal
  variable when every differing slot holds the same concrete type); a variable whose
  bounds differ between cases is copied per case, not joined.

### Known precision gaps (sound, looser than C)

- **P1** A variable occurring in the bounds of two variables gets no facts from either
  bound's meet (`el_same`, 2 census rows).
- **P2** Union equalities keep only necessary member conditions (fuzz rows such as
  `T = Union{Int64, Z}` without `String <: Z`).
- **P3** Meets the type language cannot express: a covariant slot meeting a variable and a
  type, `N - 1` lengths, `DataType ∩ Type{<:Tuple}` (C returns the same).

### Current attribution

| | census (both orders) | fuzz (3 seeds, both orders) |
|---|---|---|
| same | 227,203 | 5,420 |
| tighter (no witness against it) | 43 | 172 |
| C misses a common type (D2 and others) | 31 | 6 |
| D1 declared bounds | 8 | 17 |
| looser by `<:`, no valid instance outside C | 2 | 5 |
| looser, unattributed | 3 (P1 ×2, one `N = 0` row the sampler misses) | 6 (P2) |
| incomparable, unattributed | 0 | 20 |
| prototype unsound (`tp_unsound`, fuzz witnesses) | 0 | 0 |

Logged runtime queries (`log_replay.jl` with the uncommitted `jl_intersection_log` hook in
`jl_type_intersection_env_s`; each part is one process running test files and loading
stdlib packages, replaying every unique query afterwards):

| part | unique queries (calls) | supported | identical to C | looser / tighter / incomparable | missed witness |
|---|---|---|---|---|---|
| dict, tuple, iterators, Printf | 698,286 (4.95M) | 98.95% | 99.99% | 58 / 10 / 1 | 0 |
| reduce, missing, Dates | 346,840 (1.10M) | 99.36% | 99.98% | 37 / 16 / 8 | 0 |
| broadcast, sorting, LinearAlgebra, SparseArrays, Statistics | 962,392 (3.63M) | 99.42% | 99.99% | 49 / 17 / 11 | 0 |

Static parameter values on these: 307,320 same grade as C, 24 better, 19 worse. After the
`where` versus data type equality rule, unsupported queries on the tuple/dict workload are
0.11%. No result leaks a variable. `TypeEgal{X}` is supported as a
closed singleton: it meets `Type{T}` only with `T == X`, and a diagonal variable (possibly a
kind such as `DataType`) contains it rather than equals it.

The 20 incomparable fuzz rows are not yet attributed one by one; the examined ones are P2.

## Size and cost of a C port (2026-09-30)

The prototype is 1,136 lines of code (no comments or blanks). The intersection part of
`subtype.c` it would replace is about 2,900 lines in 48 functions. C for the same logic
needs 2-3x the lines (GC rooting, no closures, explicit worklists), so a direct port would
be about the size of what it replaces. The case for it is the semantics (sound, symmetric,
every difference from C attributed to a rule, no state-dependent flags), not line count.
Before porting, the prototype should be ablated the way the branch's C was: remove each
mechanism in turn and keep only those that change a classified result.

Work per query on the census (`tp_stats.jl`): 2.25 solved cases per query on average
(1 case for 71% of queries); 6.5% of queries split on `Vararg` emptiness (43,544 extra
cases, most of them empty); 0.28 bound meets and 0.05 containment checks per query. The
prototype takes about 300 µs per logged runtime query in Julia, C's current algorithm about
3 µs on the same queries (1.99 s against 212 s for the first replay part); the prototype
also calls C's `<:` and `typeintersect` on closed types.

## Ablation of the prototype (2026-10-01)

Each mechanism was switched off in turn (`boogaloo/TwoPhase.ablate.jl`, `ablate_snap.jl`)
and every result compared with the baseline: 227,290 census rows, 5,804 fuzz rows, and
60,000 logged runtime queries whose types reload in another process (`capture_queries.jl`).
Changed rows are classified as looser, tighter, incomparable, unsound (a fuzz witness or an
instance of the baseline's or C's result in both inputs is lost), or newly unsupported.
Representation-only mechanisms were judged on invariant I4 instead (how often the
method-matching entry point falls back to `b`).

Removed (no change on any row, or only on I4 and there none either):

| mechanism | note |
|---|---|
| `pinned_value` in `build` | `wrap` already substitutes a variable whose bounds coincide |
| closed-contains shortcuts in `build` | the general paths give the same results |
| `closedize(lo) <: hi` check in `sub!` | implied by the other emptiness checks |
| closed-pair disjointness checks in `relate!` | subsumed by the slot check after solving |
| bound meet in `set_ub!` | the meet in `sub!` covers every case seen |
| `TypeEgal{X} <: v` lower bound for diagonal variables | never changed a result, runtime queries included |
| lifting in the second `relate!` pass | identical results; a fixpoint over further passes never found anything new |
| `Binding.cub` | unused |

Kept, with the rows that change without them (looser unless stated):

| mechanism | rows | rule |
|---|---|---|
| supertype-chain equalities in `relate!` | 528 | structural |
| concrete-leaf rule (`T` contains a concrete slot) | 221 | structural |
| `union_equate!` | 162 | P2 |
| instance check for a `where` on the left of `<:` | 90 | quantifier order |
| `A <: B` fast path | 90 (incl. 12 unsupported, 16 tighter) | I3 |
| splitting open unions | 88 | D5/D6 |
| adopting open bounds | 70 | D4 |
| variables only in kept bounds (`inbound`) | 66 | |
| supertype-chain equalities in `sub!` | 56 | structural |
| bound meet in `sub!` | 42 | D4 |
| `Vararg` length equations | 32 | D2 |
| tuple decomposition in `sub!` | 30 | structural |
| per-member local solves in `build` | 30 | D4 |
| slot emptiness after solving | 22 | |
| `relate_union!` | 10 | D5 |
| second `relate!` pass | 8 | |
| correlated-variable preference in `build` | 8 | |
| union representation preference (`canonical`) | 6 | I3 |
| `where` lifting in `relate!` | 6 | |
| restore outer bindings, owner copy-back, candidate filter, declared-bound checks, closed-member union rule | 4 each | D4, D4, P2, D1, — |
| diagonal pair, per-branch diagonal | 3 each | D3 |
| case containment (`subsume`) | I4: 110 more fallbacks to `b`, 48 more worse parameter values | I4 |
| case merges (`merge_pairs`, diagonal merge) | I4: 18 and 8 fallbacks | I4 |
| `rewrap_closed` | run did not finish in 30 min (performance) | |

One fix came out of the pass: the rule that lets a diagonal variable meet `Type{X}` without
being below it also skipped `T <: DataType`, giving `Tuple{T, T} where T` for
`Tuple{DataType, Any} ∩ NTuple{N, T}`; it now applies only when the closed form is not a plain
data type (C's answer).

After the pass the prototype is 1,116 code lines, identical to the baseline on all 293,094
rows except those two.

## Consolidation (2026-10-01, `boogaloo/TwoPhase.jl` = `TwoPhase.v26.jl`)

The prototype was restructured so each rule appears once, every step checked against the
293,094-row baseline (census, fuzz, 60,000 runtime queries) and a 44,076-row subset.

Structure:

- **Three relations, one decomposition.** `relate!`, `sub!` and `equate!` repeated the same
  structural walk. Now `meet!`, `sub!` and `equate!` each keep only their rules for
  variables, unions and `where`s, and hand everything else to `decompose!`: the parameter of
  `Type{T}` and the one value of `TypeEgal{X}`, tuple elements, and the invariant parameters
  at the head two nominal types share (`common_head`, also used by `build`).
- **Every meet is in a certain slot.** With `Vararg` emptiness a case split (D2), a kept
  `Vararg` has elements, so the `certain` flag was always true in effect; it is gone.
- **One local solve.** Bound meets, union members and `Vararg` elements share `local_meet`;
  what each keeps from it is stated next to it (facts about owned variables; outer bindings
  restored except closed specializations; facts every feasible member implies).
- **Closed approximations are one rule.** A non-tuple type is closed by turning its
  variables (environment unknowns or free variables of an enclosing `where`) into `where`s
  over their closed bounds; the structural fallbacks are gone.
- **Case merging**: identical bodies and one-slot differences merge pairwise; the diagonal
  merge is separate. `wrap` collects and orders its variables in one depth-first pass.
- The file reads top-down: types as data, unknowns, closed approximations, phase 1, phase 2,
  cases, entry points. Prototype-only statistics and dead helpers are removed.

Semantic changes, each on purpose:

- **Equality of unions is mutual subtyping** (`x == y` as `x <: y` and `y <: x`), replacing
  the separate necessary-member rules. It closes part of P2: `T = Union{Int64, Z}` now gets
  `String <: Z` (4 rows tighter).
- **One rule for a type below a union**: the members its closed form can meet are the
  candidates; a closed type below the union of the closed members is done (the old rule
  checked single members, which is unsound for a tuple whose parts are covered by different
  members); no candidate is empty; one open candidate gets the subtype fact.
- A union member's own `where` is local to its feasibility solve.

Result: 1,116 → 1,007 code lines. Against the baseline: 8 rows tighter (no witness against
them), 2 newly supported (`Core.AnyType ∩ TypeVar`, empty, as in C), nothing else changed.
Fuzz: sound and symmetric, 31 looser rows (was 33). I4 unchanged.

## Self-contained, total, one fixpoint (2026-10-02, `TwoPhase.v29.jl`)

Five changes, each checked against the v26 results (quick set: 44,076 rows; full: 293,094).

**One fixpoint (item 3).** The two `meet!` passes are gone: the meets of a case are facts,
applied again whenever a binding has changed since they were last applied (`env.version`,
`env.met`). Two rules made this terminate and cost the same as before:

- one witness per fact: a `where` lifted for the fact `t <: u` (or `t` meets `u`) is lifted
  once, keyed by the fact's types up to `==` and tagged by the relation (`lifted!`);
- a fact implied by a variable's current bound is a no-op (`ub <: hi` with the unknowns
  held fixed).

Local solves are one-shot approximations the enclosing fixpoint evaluates again; they do not
iterate their own meets (doing so nested the fixpoint three deep, 30-90x slower on union-heavy
signatures). Results: identical to v26 on every row.

**One commit rule (item 6).** `commit!(env, alternatives, keep)` commits, as facts, the equality
or bounds every feasible alternative of a local solve agrees on, for the variables `keep`
selects, never a fact mentioning a local variable. Union members use it for every variable;
the meet of two bounds of `v` for the variables only in the bounds of `v`'s equivalence class
(ownership follows equality; keyed by identity it lost facts whenever the other variable was
chosen as representative). It replaces the wholesale copying of bindings. Results: identical.

**Self-contained (item 1).** No `typeintersect` call is left; only the runtime's `<:` is used.
Closed meets recurse into the prototype, memoized (a closed meet is a function of its two
types; a result computed with a cycle cut short is not cached), with a cycle meaning "unknown"
(a skipped check, or the existing bound kept). What C's intersection used to decide is now
stated as rules:

- closed atoms (nominal types, `Type{X}`, `TypeEgal{X}`) meet only by containment; the
  singletons `Type{X}` and `TypeEgal{X}` meet a kind by `X isa kind` (the runtime's `<:`
  says `Type{DataType} <: DataType` is false);
- nominal types without a common head share no member; a tuple type and a nominal type share
  no member;
- `Type{X}` meets a nominal type only if that is above `Type` or a kind containing `X`'s kind
  (`kind_of`: `typeof(X)` for closed `X`, else `DataType`, `Union`, `UnionAll`, `TypeEq` by
  shape, unknown for a variable);
- `Type{T}` meeting a kind is the contained side if they are comparable, else the `Type{T}`
  side (C picks inconsistently);
- closed tuple slots where one contains the other are settled before lifting: both sides get
  the smaller, which is not lifted (two copies of one signature slot no longer become two
  systems of unknowns; the pathological `copyto!` pair went from >30 min to 0.5 s);
- a closed comparable pair in `meet!` adds no facts;
- only non-tuple slot pairs are recorded for the emptiness check (a tuple slot is empty
  exactly when an element slot is).

Results: 1 row tighter (`UnionAll ∩ Type{Core.TypeApp}` is empty: `TypeApp` is a `DataType`),
5 rows incomparable (`Type{T} where T>:X` against a kind, by the rule above). Quick set 13x
faster: mismatches are refuted in phase 1 instead of after a full solve.

**Total (item 2).** The remaining `Unsupported` exits are inputs that are not types (a `Vararg`
not last in a tuple, a length that is not an integer or variable, a non-type). Otherwise:

- (superseded in v33 by opaque variables, see the decisions below) a variable free in the
  inputs is rigid: never bound or wrapped, its bounds stand for it in
  closed approximations, and it equals a type only within its bounds (a value other than a
  type only if unbounded); results hold for every value of it (C treats it as distinct from
  every type: 2 runtime queries where C says empty and the prototype allows `N == 7`);
- equality is mutual subtyping wherever structure does not decide (unions, `where`s,
  `Union{}`); a nominal type, `Type{X}` and a tuple whose fixed elements are never empty are
  never below `Union{}`;
- a `Vararg` of any length never equals a fixed-length tuple or one of some length: the
  runtime's `<:` does not take a `Vararg` element to be `Union{}`
  (`Type{Tuple{}} <: Type{Tuple{Vararg{E}}} where E` is false);
- past the case budget (128 per side) or with more than 4 length variables a union stays
  whole and a `Vararg` is not split on emptiness; such a case does not meet the trailing
  `Vararg` elements (`env.split`).

Census: no `Unsupported` and no error on any of the 268,960 pairs; 184 newly supported rows
in the quick set, the 19 without free variables identical to C, the 165 with free variables
identical to C except the 2 above.

**Budgets (item 4).** Every budget now gives a sound over-approximation: local solves nest
3 deep (a deeper check is skipped), `solve!` stops after 32 rounds (fewer facts), `closedize`
after depth 32 (`Any`/`Union{}`), the case and length budgets as above.

**Representation (item 5).** `canonical` no longer compares `string`s: between two spellings
the facts make equal it prefers the closed one, then fewer bare variables in unions, then the
smaller (`type_size`), then a total structural order (`type_order`: kind, module and name,
parameters), which C can implement. The choice matters only where facts under-determine a
union equality (P2): 6 quick-set rows differ from v28 (2 tighter, 2 looser, 2 incomparable),
all sound and order-independent. `rank` (local < owned < other) and the correlated-variable
preference in `build` stay; both have stated reasons.

v29 against v26 on the quick set: everything above, nothing else. Fuzz: sound, symmetric, no
missed witness. I4: 7,528 fallbacks to `b` (v26: 7,536), parameter values unchanged.

**Case enumeration (item 7), measured and not changed.** On the census (227,448 queries,
136 s) 99% of top-level cases are empty (654,482 of 660,936) and take 57% of case time; 72%
of queries have one case. Dropping union members that cannot meet the other side's slot
before enumerating (decided by heads alone, `plainly_disjoint`) removes 45% of the cases and
settles 38,160 queries with no case at all, but costs as much as it saves (137 s); deciding it
with closed meets costs 6x more. The time is in solving each case, not in how many there are,
so cases stay enumerated up front. For the C port this is an implementation choice (C's
union-state stack is lazy) as long as it yields the same cases; the per-case cost is what C
changes. The slowest pairs are union-heavy `LinearAlgebra` factorization signatures (about 6 s
cold in Julia).

## Oracle: precision measured, not compared (2026-10-02, `boogaloo/oracle.jl`)

`oracle.jl` defines intersection by its members. A type denotes the concrete types below it;
over a finite universe built from the fuzzer's blocks (leaf, abstract and union types; `Ref`
(members `Base.RefValue`), `Vector`, `Complex`, two-element tuples; 176 values a variable can
take, concrete ones included so a covariant variable can stand for any concrete type), it
enumerates the members of each input and result and checks them with the runtime's `<:` on
concrete types. For a result `R` of `A ∩ B`: *missing* = members of both not below `R`
(unsound), *excess* = members of `R` not in both (imprecise). Types with more variables use
fewer values per variable; every enumerated member is still a real member, so this lowers
coverage, never correctness. A judgment past 3,000 members or 20 s is "too big".

On the fuzz rows where the prototype and C differ (v30: 239 rows, 106 judged):

| | exact | excess | missing |
|---|---|---|---|
| prototype | 100 | 6 | 0 |
| C | 39 | 63 | 4 |

No row has C exact and the prototype not. The 6 rows with prototype excess (3 pairs, both
orders) are **inexpressible**: their exact answer needs a slot variable with two upper bounds
(`S<:T` and `S<:Signed`), which `where` cannot write; C has the same or more excess. This
confirms P3 as a limit of the type language rather than a gap.

Two prototype gaps the oracle found, both fixed (v30):

- **Singleton slots.** A slot whose two sides' closed forms meet in one concrete type holds that
  type, so both sides contain it (a generalization of the concrete-leaf rule; applied every
  round of `solve!`). `Tuple{Ref{Ref{T}}, T} ∩ Tuple{Ref{Ref{X}}, Union{Int, Y}}` now gets
  `Nothing <: T` in the `Y` case.
- **Diagonal lower bounds.** A diagonal variable is concrete, so a concrete lower bound is its
  value.

**P2 (union-equality tie-break) is now a rule, not an accident (v31).** Flipping the final
structural order changed 24 fuzz rows; where the oracle could judge both spellings, either
both were exact (cosmetic) or the spelling whose variables occur elsewhere in the inputs was
exact and the other had excess (3, 19, 2 members): picking a variable used once forgets the
correlation the facts could not express. `canonical` now prefers, after the closed spelling
and fewer bare union variables, the one whose variables occur more often in the inputs
(`env.uses`). After that only 6 rows depend on the structural order, and the judged ones are
exact either way. Against v30: 4 rows tighter, 2 incomparable by `<:` but tighter by hand
(the `Vector{Union{Missing, Nothing, T}}` row: the old spelling allowed a union and a `T` that
disagree).

Coverage limit: 55% of the differing rows are too big for the oracle (wide `Any` slots, several
variables). The census types (`AbstractArray` and other Base hierarchies) are outside its
universe.

## Remaining heuristics reviewed (2026-10-02, `TwoPhase.v32.jl`)

- **P1 measured.** A bound meet of `v` commits facts only for variables that occur in no other
  bound: those facts hold only if `v` is nonempty (`Type{Union{}}` is a member). On the census
  46 of 227,448 queries derive such a fact and drop it, and the examples are cosmetic
  (`Tuple{T1,T1} where {T<:Integer, T1<:Rational{T}}` against C's `Tuple{T,T} where
  T<:Rational`, equal given the declared bound). The exact fix is a case split on emptiness for
  variables that occur only invariantly (as for `Vararg`, D2), doubling cases per such variable;
  not adopted, stated as a precision trade-off.
- **One preference between equal spellings (v32).** `build`'s "correlated variable" preference
  and the v31 occurrence rule are one criterion: prefer the spelling that stays tied to the
  inputs, by invariant occurrences, then all occurrences (a variable occurring once covariantly
  becomes its bound and drops the correlation). `canonical` applies it everywhere and the
  `correlated` flag is gone. Against v31: 2 rows tighter; the oracle shows the new result exact
  (old: 1 excess, C: 32).
- **Kept, with their principle:**
  - local-solve depth 3: a budget; a skipped local solve only loosens;
  - `closedize`: the one approximation of an open type by a closed one (variables as `where`s
    over their closed bounds);
  - `rank` (local < owned < other): a local variable must never represent an outer one (its
    facts would leak), and an owned one must not represent a variable that matters elsewhere
    (commits are filtered by ownership);
  - `settle_closed_slots`: lifting a `where` serves correlation; a closed slot comparable with the
    other side's needs no unknowns.

### Subtype-side questions: decisions (v33)

Rule: follow what C does unless a semantic reason says otherwise; each departure is listed.

1. **`Type{X}` against a kind.** Kept as C: a singleton `Type{X}`/`TypeEgal{X}` meets a kind by
   `isa` (C's kind mask), although `Type{DataType} <: DataType` is false. One departure,
   `Type{T} where T>:X` against a kind (5 census rows): C returns the kind when `T`'s upper bound
   is `Any`, the prototype returns `Type{T} where T>:X`. Both are over-approximations and neither
   contains the other. C's own rule is "the kind if `Type{T}` contains it, else `Type{T}`" (its
   comment calls `Type{T}` the conservative answer); `ub == Any` stands in for "contains it" but
   ignores the lower bound. The prototype checks containment exactly, so the rule is C's.
2. **`Vararg` element never `Union{}`.** Kept as C (already so since v29).
3. **`<:` not transitive for some unions of tuples.** Kept as C: the prototype works around it
   (cases merge under one `where`) and changes nothing in `<:`. This is a C bug, related to #63478.
4. **Free variables: changed to C.** Since #61876 (`bef642966f`) `<:` treats a free variable as
   opaque: one type, equal only to itself, above `Union{}` alone and below only `Any` and the
   unions that contain it. Its bounds play no part (`Tuple{M} <: Tuple{Real}` is false for
   `M<:Integer`; 1.12 said true). The prototype's "rigid" rule (results hold for every value within
   the bounds) was not soundness relative to `<:` (I1): by `<:`, `Ref{M}` and `Ref{Int}` have no
   common subtype, so C's `Union{}` is exact and the old `Ref{Int}` was excess. The prototype now
   adds the free variables to `OPAQUE`, so they are closed and the runtime's `<:` decides them.
   - Diagonal rule as in `<:`: a bare opaque variable counts as concrete (`Tuple{M, M} <:
     Tuple{T, T} where T`); a type that contains one does not (`Vector{M}`, since
     `isconcretetype` is false).
   - `Type{M}` against a kind: the kind of `M` is unknown, so `Type{M}` meets every kind except
     `TypeofBottom` (`M` is not `Union{}`), and does not meet `TypeVar`.
   - The rigid code is deleted (the `equate!` branch, the "value fits an unbounded variable"
     rule). New: `isopaque`, `in_kind`, one `decompose!` rule (an opaque variable is in no
     nominal type, tuple or `Type{X}`, and only an empty type is below it), and a module-local
     `<:` (the builtin rejects a `TypeVar`).
   - Departures from C, both in C's `intersect_type_type`, which still reads a free variable's
     bounds: `Type{M} ∩ TypeofBottom` (C: `Type{Union{}}`, though `M` is not `Union{}` by `<:`)
     and `Type{A} ∩ DataType` with `A` unbounded (C: `DataType`). The prototype follows `<:`.
   - C's own asymmetry: `Tuple{M, Vararg{Int}} ∩ (Tuple{Vararg{T}} where T)` is `Tuple{M}` in
     one argument order and `Union{}` in the other. The prototype gives `Tuple{M}` both ways.

Checks: quick set unchanged on all 43,903 rows without free variables. All 1,127 rows with free
variables (173 in the quick set, 954 runtime queries) equal C, including the 17 where v32 differed.
On the 37 probe pairs only the two departures above differ. Quick set 9.8 s (v32: 11.2 s).

## Progress

- 8: done (`d5b03d30a8`, `test/testhelpers/IntersectionWitnesses.jl`).
- 1: tried, not adopted (see above).
- 2: audited; first step would not simplify; flag deletion is load-bearing.
- 3: probed; re-intersection doubles as backward propagation, so static kinds alone
  change results.
- Two-phase prototype: sound and symmetric on fuzz and Base, covers 99.93% of Base
  signature pairs with 99.87% agreement; the differences are mostly C imprecision or
  C bugs (see above).
- 5: downgraded.

## Order

The phase-1 prototype (above) is the main line; 2, 3 and 4 follow from it rather than
preceding it. 6 (symmetric variables) and 7 (`Vararg` lengths) are independent and
could be folded into the prototype's design. 1 and 5 are closed.

## Open issues

- Remaining unsound families: triangular `S<:T` (`S<:T` narrows `T`'s upper bound, a
  master bug), and variables in invariant positions of unions (`Ref{T}` inside a
  union member).
- The `bench.jl` dispatch claims (`invalid_bound`, `pinned`, `widened`) now hold on
  the branch after adopting #63155.
- Master's stack-overflow pair is fixed in one order only (`@test_broken`); reduced
  variants of the `Ref{Tuple{S, Ref{S}}}` family can still overflow in long-running
  processes on master and the branch.
- Trim candidates, re-measured with complete failure lists: only the
  `jl_type_intersection_env_s` fallback is single-test (1 test line). The others fail
  several branch-added tests: `constraint_scope` 9 lines, `intersect_unionall_`
  re-narrowing 5, the `required_upper` early return 4, the reintersection tweak 4
  (the last three include triangular cases where master is precise).

## C port (2026-10-02)

`src/intersection.c` (~3,100 lines, added to `src/Makefile`): a function-by-function port of the
prototype. `jl_type_meet(A, B)` (exported, for testing) and `jl_type_meet_unrelated` (for the
runtime entry, which has already checked `A <: B` and `B <: A`). `JULIA_TYPE_MEET=1` routes
`jl_type_intersection_env_s` and `intersect_types` to it (temporary switch in `src/subtype.c`;
`b === Type` with an env still uses the old code). `jl_typeeq_meets_kind` is exported from
`subtype.c` so both the port and the prototype use C's kind mask.

Design of the port: return codes instead of `Unsat` (phase 1 `int`, phase 2 `NULL`); every
environment's state lives in GC-visible svecs/arrays (`menv_t.root`: bindings as
`svec(var, lb, ub, eq, owner, diag, cov, inv)`, facts, slots, meets, lifts, checks); a per-call
context holds the opaque variables, the active pairs and the closed-meet memo; insertion order
replaces `IdDict` order. Type construction that can throw (`jl_apply_type`, substitution) runs
under `JL_TRY`. `check_supported` dropped (the runtime's constructors guarantee it); lifted
covariant `where`s are renamed only when their variable is bound more than once in the inputs.

Agreement: identical to the prototype on 45,028 / 45,030 quick-set pairs (the 2 differ only in
a tie-break that the prototype takes from `IdDict` order).

Speed (104k pairs: census sample + 60k runtime queries): through the runtime entry 0.35 s vs
0.09 s for the old code (3.9x); median per pair 5x, p90 36x; worst pairs ~29 ms (LinearAlgebra
triangular signatures with `StridedMatrix` bounds). End to end the switch is not slower:
`test/dict.jl` 6.8 s vs 11.0 s, `ambiguous` 3.1 s vs 3.9 s, `tuple.jl`/`iterators.jl` within noise.

### Semantics found by the test suite (v34–v42)

`test/subtype.jl` with the port first gave 135 failures. Fixed in the prototype and the port,
each deliberate (quick set unchanged except where noted):

- v34: the diagonal rule applies to every `where`, including ones lifted in a local solve
  (results were not below the inputs: `Tuple{Tuple{Int,Vararg{Integer}}} ∩
  Tuple{Vararg{Tuple{T,T} where T}}` gave `Tuple{Tuple{Int,Integer}}`).
- v35 (soundness): a `Vararg` whose length resolved to an integer equals that many fixed
  elements (was `Unsat`: `Pair{NTuple{N,Int},NTuple{N,Int}} where N ∩
  Pair{<:Tuple{Int,Vararg{Any}},NTuple{4,Int}}` was empty); `v == Union{v, rest...}` is
  `rest <: v`, not an occurs-check failure (`Tuple{Ref{T},T,Ref{T}} where T ∩
  Tuple{Ref{Union{Missing,S}},Any,Ref{S}} where S>:Int8` was empty).
- v36: `Type{X}` meets a kind by C's kind mask (its members are all type objects `== X`, e.g.
  the `UnionAll` `Tuple{S} where Vector<:S<:Vector` is in `Type{Tuple{Vector}}`); `TypeEgal{X}`
  by `isa`. Corrects the earlier "question 1" conclusion, which held only for `TypeEgal`.
  Quick set: 2 rows (`UnionAll ∩ Type{Core.TypeApp}`) now nonempty, as in C.
- v37: a closed upper fact `v <: hi` also checks `lb(v) <: hi` (the meet of bounds is an
  approximation; `Type{Int} ∩ DataType` stays `Type{Int}`).
- v38: `build_local` lifts the `where`s of both sides (symmetry: two `where`s in a `Vararg`
  element gave different results by argument order).
- v39/v40: `build_local` keeps what the local solve learned about outer variables, except
  diagonal ones (it reset all but closed values; results were not below the inputs).
- v41: a `where` that stays in place (a `Vararg` element) meets by a local solve whose
  implications are committed (`meet_where!`); phase 1 previously learned nothing from it.
- v42 (speed): the "implied by the current bound" check compares substituted forms; the solve
  re-ran a full bound meet every round once two bounds had become equal (`el_same` pair 2.5 s
  -> 2.5 ms, `muladd(T,T,T) ∩ muladd(Complex,...)` 188 -> 60 ms).

Remaining `test/subtype.jl` failures with the port (65 at 31 lines; 23 `@test_broken` now pass):
- expectation updates (new result tighter or equally good, or old one wrong): 1083 (tighter;
  the test asserts `B <: I` though `B ⊄ A`), 1100 (D1: declared bounds never narrow), 1770,
  2056 (tighter, exact), 1793 (old answer misses `Tuple{Int8,Tuple{Int},Int}`), 2074, 2100
  (empty is right), 2135/2141 and 2226/2227 (now symmetric);
- precision losses, sound: Vararg lengths from subtype facts (1137, 2408, 2415) and length
  correlation (2376, 2381, 2386, 2401); a length variable used as a type (2033, 2036, 2039);
  one `N` shared by an element type (2172); `Val{Val{B}}` (2287/2288); lost `T>:Int` (2761);
  a tuple element variable with a concrete-leaf bound is not `Union{}` (2832); `Union{Int,T}
  == Integer` (2893/2894); `Type{Int} ∩ DataType` as `TypeEgal{Int}` (2955/2956).
Other suites with the port: `ambiguous` passes except the printed candidate (covariant-only
`where T<:Integer` is printed as `Integer`), `detect_ambiguities(Core, Base)` finds none;
`reflection`, `dict`, `tuple`, `iterators` pass.
- v43: a covariant slot holding a value other than a type is empty (`Tuple{..., :N}`); a variable
  used as a `Vararg` length takes integer values, so a type bound on it is unsatisfiable
  (`Tuple{Int,Vararg{Any,T}} where T ∩ Tuple{T,Vararg{Any,T}} where T` is empty).
- v44: tuple subtype facts with `Vararg`s give element facts and a length relation (they were
  skipped); a length variable gets a minimum (`N ≥ c`) from a length relation with an unbounded
  `Vararg`; before building, a length variable used only as a length with minimum `m` is written
  as `m` elements and a fresh length (`Type{<:Tuple{Any,Vararg}} ∩ Type{NTuple{N,Int}} where N`
  is `Type{Tuple{Int,Vararg{Int,N}}} where N`); build picks between two aligned length variables
  with `canonical` (it took the second, an asymmetry).
- v45: length variables related by an offset (`N == N1 + 1`) are kept as root plus offset
  (`relate!`, `len_root`), equalities through them go to the roots, and before building each one
  used only as a length is written as that many elements and its root; case bodies are
  substituted first so aliases reach their roots.

`test/subtype.jl` with the port after v45: 33 failures (from 135), 23 `@test_broken` now pass.
Remaining: expectation updates (1083, 1100, 1770, 1793, 2056, 2074, 2100, 2135/2141, 2226/2227,
2415/2416: `Pair{NTuple{N,Int},Val{N}} where N` keeps the correlation, C's answer keeps the
non-emptiness; incomparable) and precision (2172, 2287, 2761: facts from a nonempty variable's
bounds about other variables are not committed by `meet_bound`, 2832, 2893, 2955).
- v46: a variable that is a fixed tuple element of its case body is not empty, so `meet_bound`
  commits what the meet of its bounds implies about all variables, not only the ones it owns
  (`Tuple{Ref{Union{Int,T}},Ref{T}} where {U,T<:U} ∩ Tuple{S,S} where S` keeps `T>:Int`).
- v47: `Type{X}` with `X` open meets a kind by `X`'s structure (a type object equal to an
  instance of a data type is a data type or a pinning `where`, of a tuple also a union, of
  `Type{Y}` a `TypeEq` or `TypeofBottom`); v36 had skipped the check for open `X`, so
  `Type{RefValue{T}} where T` met the `TypeEgal` kind and `ismutable(::RefValue)` no longer folded.

Port bug found by the test suite: `merge_diagonal` read `jl_nparams` of a non-tuple body
(`Type{Float64}`, a `TypeEq`) and allocated garbage on the stack (`core`, `arrayops`, `sets`
workers died during inference of `convert(Tuple{Type{Int},Type{Float64}}, ...)`); fixed.

Status with `JULIA_TYPE_MEET=1` (2026-10-03): `subtype.jl` 31 failures (list above minus 2761),
23 `@test_broken` pass; passing groups: core, worlds, reflection, keywordargs, tuple, namedtuple,
missing, broadcast, abstractarray, arrayops, errorshow, iterators, functional, dict, sets, ranges,
compiler (incl. Compiler/inline), ambiguous (except the printed candidate). `show.jl:2539`:
`Union{Nothing, AbstractVecOrMat{<:Integer}}` prints as `Union{Nothing, AbstractVecOrMat{T}
where T<:Integer}`: Julia splits the input into one `where` per union member, so the alias's `T`
(two uses) wins `canonical` over the input's anonymous variables (one use each); the type is
equal, the printed alias names `T`.
- v48: a variable used only as tuple elements (not `Vararg` elements) is not `Union{}`, so with
  a concrete-leaf upper bound it is that type (`Ref{Union{Missing,Tuple{T,T}}} where T<:Int8` is
  `Ref{Union{Missing,Tuple{Int8,Int8}}}`, as `<:` excludes the invalid `Tuple{Union{}}`). Quick
  set: 8 rows tighter in this way.
- v49 (budget): at most 2000 local solves per intersection; past that, union/`where`/bound meets
  add no facts and `build_local` meets closed approximations (sound). Found by SparseArrays'
  `higherorderfns.jl`: `Type{X} where X<:Tuple{AbstractVector{S}, U, ..., U}` (32 slots of
  `U = Union{Number, AbstractVector{<:Number}, AbstractMatrix{<:Number}}`) against
  `Type{Y} where Y<:Tuple{Vararg{E}} where E` nests union local solves 3 deep across all slots;
  the port ran out of memory (15 GB) or overflowed the subtype union stack (assertion in
  `statestack_set`). Now 0.3 s (old code: 3 ms); the testset passes (6.8 s vs 2.5 s). The quick
  set does not reach the budget. The scaling probe (k union slots) was non-monotone: up to
  3^k <= 128 cases each solve ~0.5 s; the budget brings k = 4 from 40 s to 0.07 s.

Rebased onto master (2026-10-03; 16 commits, only the uncommitted `src/Makefile` line
conflicted). Query corpora regenerated (closure type names changed): `queries.jls` (385,084
runtime queries without free variables), `quickpairs.jls`, baseline `ablate_quick49.jls`; old ones
kept as `*.prerebase`. Port = prototype on 44,074 / 44,076 (2 tie-break spellings).
`subtype.jl` with the port: 34 failures = the previous 31 minus 2832 plus 2873/2874/2881/2882.
Those four expect `Tuple{U,U} ... where U<:Int64` unpinned while 2832 expects the pinned
`Tuple{Int8,Int8}`: the old C pinned in one case and not the other, and `==` cannot equate the two
spellings (`<:` excludes `U = Union{}` for membership but considers it when comparing them).
v48 pins consistently; keep it and update 2873/2881.
- v50: v48 reverted. The pinned spelling is strictly below the unpinned one by `<:`, and the
  unpinned one is a common subtype of the inputs, so pinning dropped a common subtype (unsound
  relative to `<:`; my witness checks only sampled concrete types). `ablate_snap.jl` now also
  tries the previous result and C's results (both orders) as witnesses: it flags 6 of v48's 8
  pinned rows as unsound. Test 2832 changed to `W <: typeintersect(A, B)` (`@test_broken`: the
  branch's C pins `A ∩ B`, unsoundly, but not `B ∩ A`).
- v51–v53 (#54356 family): a union member whose type the runtime cannot construct is empty, but
  only for unions in covariant positions (inside an invariant parameter the failure invalidates
  the enclosing type); a member that is a value (`Union{B, Val{B}}` with `B = 1`) is empty; a
  `Vararg` with a negative length is no tuple; construction and substitution use the runtime's
  non-throwing checks (`jl_instantiate_with_typeenv(..., nothrow)`), which ignore a declared bound's
  invalid union member (`C54356{Nothing,C}` with `T<:Union{S,Complex{S}}`) but reject genuinely
  invalid types (`NamedTuple{(:a,),Tuple{Int,Int}}`); `wrap` never returns a body with free
  variables (a variable whose closed bound is one value is that value).

Correction: earlier `subtype.jl` counts with the port came from a harness whose module lacks
`include`, so the file stopped at line 3137 and never ran its last ~400 lines. Through
`runtests.jl` (one worker), with the port: 46 failures at 33 lines, 30 `@test_broken` passing
unexpectedly, 1 error (2061, a structural assertion on the result). Remaining: expectation
updates and sound precision differences listed above, plus 3251 (old C proves empty; ours keeps
`Tuple{Type{S},Val{S},Val{S}} where {A,S<:Val{A}}`), 3261/3262 (ours tighter than expected, which
is not below either input), 3403 and 3422/3427 (ours loses one correlation, contains the
expected), 3459–3499 (diagonal variables over `Type{X}` slots under the kind semantics; ours is
sound but the tuple fixup in the runtime entry replaces our union of tuples by `b`), 3540/3555
(environment variable names for an unused `where` variable). Default mode (branch C): passes.

- v54 (3459–3499, diagonal variables over `Type{X}` slots): a diagonal variable meeting a closed
  `Type{X}` / `TypeEgal{X}` (or a union of them) gets the fact `v <: Union{kinds}`, the kinds
  from the same mask C uses for `Type{X} ∩ kind` (`jl_typeeq_meets_kind`); and a diagonal, inhabited
  variable whose closed upper bound is a concrete DataType is that type. The slot itself is built
  by the existing kind rule (`Type{X}` is the tighter of `Type{X}` and the kind), so results stay
  below `A`. Rejected: tagging by `typeof(X)` (what old C's intersection effectively does). It is
  unsound by `<:`: `Type{Tuple{Int}}` contains `TypeEgal{Tuple{S} where S<:Int}`, a `UnionAll`, so
  `Tuple{Int64, TypeEgal{Vector}, TypeEgal{u}}` is a common subtype of
  `Tuple{Any, Union{Type{Vector},Type{Int64}}, Type{Tuple{Int}}}` and `Tuple{Int64,S,S} where S`
  that old C's result drops (now a test). Also rejected: widening a mixed-kind `Type{X}` slot to
  the variable (keeps `r <: B`). Tests 3492 (`r <: B`) and 3499 (`r <: A`) cannot both hold
  under a sound rule: their deciding cases, `(Type{Vector}, Type{Float64})` and
  `(Type{Vector}, Type{Int64})`, are the same shape, and old C tells them apart only through the
  `typeof(X)` choice. Updated: 3459 (`==` ours, plus the `Tuple{Int}` witness test), 3465 (`==`:
  same type, `where` spelled outside), 3475, 3477 (the runtime's tuple fixup turns our union into
  `A`), 3483 (`r <: A` in place of `r <: B` and the mixed-pair exclusion). Quick set: no change.

`subtype.jl` triage after v54 (41 failures, 30 unexpected passes, 1 error before):
- 30 `@test_broken` now pass (results below an input, witnesses kept, emptiness found): now `@test`.
- Old expectations unsound by `<:`: 1793 (#30335 `Tuple{I}` variant: `I` ranges over
  `[Int, Integer]` since `Tuple` is covariant, so `Tuple{Integer, Tuple{Int}, Int}` is common and the
  expected `LT<:Union{Tuple{Int},Int}` drops it); 1100 (#20671: the expected adds the declared
  `T<:Number`, but `Tuple{Type{SIQ{T,…}}, SIQ{T,…}} where {T,…}` with `T` unbounded is below both
  inputs). Tried: declared parameter bounds as facts outside unions (results gain `T<:Number` like
  C). Rejected: 32 quick-set rows unsound for the same reason (`Rational{Tr} where Tr` unbounded is
  common when an input spells it so).
- Ours tighter: 1770, 2074, 3261/3262, #34170 (2135/2141, the "better result" the test names).
- Exact where old C over-approximated: #38423 (2100) is empty (`Type{Union{Ref{L},Val}}` is never `==`
  to `Type{Union{Ref{L1},Val{R}}}`); the test was a stack-overflow regression.
- Incomparable, both sound: 2056 (`Type{Union{Missing,Int}}` vs `Union`), 2226/2227 (symmetric
  `Tuple{Dict{I,T},I,I}`), 2415/2416 (ours keeps `N` correlated and drops `N ≥ 1`; C keeps `≥ 1` and
  decorrelates; `N - 1` is not expressible), 2956/2957 (`U<:Type{Int}` vs C's `U<:DataType`; the
  exact `TypeEgal{Int}` would need `Type{X} ∩ typeof(X) = TypeEgal{X}` for parameterless `X`, which
  holds but changes inference-visible meets).
- Precision gaps, now `@test_broken` plus a containment check: 1083 (#21118, `T <: A`), 2172
  (#39698, `N` correlation), 2287/2288 (#41096, disjunction `Val{B} <: Union{Val{true},Val{false}}`),
  2894/2895 (`Union{Int,T} == Integer` should collapse to `Integer`), 3251 (emptiness needs an occurs
  argument), 3403 and 3422/3427 (#57852, lost correlation).
- Spelling only: 2061 (`N` lifted out of `Array{T}`; test now unwraps), 3543/3558 (#61634 env slot
  names: the merged variable is named by the symmetric tie-break; slot identified by bound).

## Performance pass (2026-10-03, v55–v56)

Corpus: quick pairs + runtime queries (429,160). Pure intersection time via `jl_type_intersection`:
old C 0.244 s; port 0.832 s → 0.695 s. Through `typeintersect` (incl. the entry's own fast paths):
0.41 s old vs 1.04 s → 0.89 s port. Queries under 1 µs (84%) are at parity; 1–20 µs ~2× (the
fixpoint re-applies facts and re-meets each round; spread over sub/closedize/rewrap/build, no
single hot spot); the tail decides the total: 333 queries over 100 µs cost 287 ms (old 23 ms),
the top 10 alone 40% of the excess (LinearAlgebra `copy`/`imag` on triangular types with
`StridedMaybeAdjOrTransMat` bounds: 34 ms vs 0.6 ms; cost is `meet_union` over the huge union in
one bound meet: local solves, `jl_type_union`'s pairwise `simple_subtype`, `<:` on members).
- v55: `expand` keeps a union whole at a tuple slot closed in both inputs (it cannot interact
  with a variable); closed element pairs are met as closed meets in `meet_tuples!` (emptiness
  only) and in the tuple build. This also removed a latent bug where `build` sent closed pairs
  through structural rules that assume phase 1 checked them (the kind rule turned
  `TypeEgal{Int128} ∩ Type{Int16}` into `Type{Int16}`; a closed inner `Tuple{Vararg}` pair lost
  its empty case). `promote_result` with two 10-member `TypeEgal` unions: 469 ms (TP) / 12 ms
  (C) → 0.1 / 0.04 ms.
- v56: `settle_closed_slots` removed (redundant with v55); `lift_covariant` leaves slots closed
  in both alone.
- C only (memo, same results): implication checks in `set_ub` cached per query by
  (bound, fact, closed bound up to renaming); closed meets memoized before the `<:` shortcuts;
  `expand` builds each case tuple once and returns the input when nothing splits.
Quick set unchanged at every step; C = prototype on 44,074/44,076.

### Tail (v57, 2026-10-03)

Tail = 333 queries over 100 µs: 287 ms → 120 ms (old C 23 ms); total 0.695 s → 0.521 s.
- A local solve inherits its parent's facts, so the fact that started a bound meet started it
  again inside itself, one level deeper, up to the depth cap (`StridedMatrix ∩ AbstractMatrix`:
  43 local solves where 9 suffice). `meet_bound` now returns no new bound when the same pair of
  bounds is already being met (the depth cap's fallback). LinearAlgebra triangular ∩
  `StridedMaybeAdjOrTransMat`: 30.7 → 3.6 ms in C.
- `local_meet` first checks nominal disjointness (single inheritance: two nominal non-tuple heads
  share a member only along one supertype chain), before spending work.
- `set_ub` fails the case when a variable every member has (a tuple element) gets upper bound
  `Union{}`, instead of finding the empty slot a round later.
- `expand` drops union members nominally disjoint from the other input's slot (a bare variable
  there stands for its declared upper bound), and keeps a slot whole when splitting separates
  nothing: closed in both; not a tuple and its variables occur in no other slot; or closed
  against a diagonal variable. `muladd(::StridedMaybeAdjOrTransMat…)` vs `muladd(::T,::T,::T)`:
  23 → 0.1 ms; `promote_result` with two 10-member `TypeEgal` unions vs `Type{T}`, `Type{S}`: 100
  cases → 1; `not_sametype` (diagonal over a 10×10 nested tuple): 100 cases → 1.
- Port tie-break fix: C's `type_order` compared module paths through `Main` (`Main.Base`) while
  the prototype uses `string(module)` (`Base`); C now stops at Main/Base/Core like `fullname`.
  That was the cause of the two long-standing C/prototype "spelling" differences: C = prototype
  on all 44,076 quick pairs.
Quick set unchanged by every step. Remaining tail: union-heavy synthetic pairs at 1–2 ms
(old 0.1 ms) and the triangular family (13 local solves on huge types, ~250 µs each).

### Mid-range overhead (v58, 2026-10-04)

Corpus total 0.521 s → 0.417 s (old C 0.244 s, 1.71×). Split by origin: real runtime queries
(385k, logged from sessions) 270 ms vs old 175 ms (1.54×, tail only 6.6 ms); quick pairs (census +
synthetic fuzz) 147 ms vs 67 ms (2.18×, carries almost all of the remaining tail).
- Rule (prototype + C): two closed `Type{T}` ranges (`Type{X}` or `Type{T} where lb<:T<:ub`) meet
  in `Type{T} where Union{lb1,lb2}<:T<:ub1∩ub2` (empty unless the lower fits the upper, `Type{l}`
  when they coincide); this is exact under `==`-class semantics. These pairs came from
  `::Type{<:X}` dispatch and each ran a nested solve (7 µs → 1 µs). `meet_or_nothing` also
  rejects nominally disjoint pairs before anything else.
- C only, results unchanged: identity shortcut in closed meets (the declared-bound facts
  re-applied each round met a bound with itself: solve samples halved); `closedize` cached per
  env while its version is unchanged (fresh `TypeVar`s otherwise defeat the closed-meet memo),
  invalidated at the few direct binding writes; closed element pairs go straight to the memoized
  closed meet; `jeq` rejects DataTypes with different names without subtyping; env arrays,
  the query's context arrays and variable lists are allocated on first write, `intersect_cases`
  bookkeeping is C-side (`varinfo_t`, `arraylist_t`), the cycle list is an `arraylist_t`, a query
  with nothing to split skips `expand`; `unique_pairs` removed (pushes already dedup).
  Allocations for a one-case query 36 → 14.
Runtime queries reach `solve` rarely (28,821 solves for 385,084 queries, 1.2 rounds each); what
remains is diffuse (closed element meets, fact application, build, GC).

## Code size if the port replaces C (2026-10-04)

Measured on a scratch copy of `subtype.c` with both entries routed to the port: removing the old
intersection functions and everything only they use (iterating `-Wunused-function`) deletes
1,356 lines (7,623 → 6,267). Not yet counted: the constraint machinery that only intersection
enters (`constrain_*`, `constraint_*`, `required_upper_bound`, `upper_is_slot`, `reachable_var`:
~340 lines; `constrains_param_static` stays, the env fallback uses it) and the
`e->intersection` branches in shared subtype code (16 uses, ~80 lines). Total ≈ 1,780 lines
removed against 3,943 added (`src/intersection.c`: 150 functions; largest `build` 192,
`merge_build` 151, `closedize_` 108, `subst_` 99, `solve` 95, `intersect_cases` 93, `wrap` 90).
Net ≈ +2,150 lines. The prototype (1,799 lines of Julia) is itself about the size of the C it
replaces, so "less code" needs a smaller algorithm, not a tighter port.

## Ablation for size (2026-10-04, `TwoPhase.abl.jl` + temporary `JULIA_TYPE_MEET_ABLATE` in C)

Each rule switched off, measured on the quick corpus (prototype) and on test/subtype.jl
(port). Needed: `canonical` (34 quick rows change, 25 looser), `diag_equate` and `slot_concrete`
(an unsound row each), `kind_rule` (16 test failures), `peel` (48), `mins` (24), `meet_where` (8),
`member_kinds` (8), `param_bounds` (ambiguous.jl:175: `Complex{String}` must be empty, else a
false ambiguity), `implied` (a subtype.jl query does not finish within an hour).
No effect on the quick corpus, subtype.jl, ambiguous.jl or core.jl, also combined:
`merge_diagonal` (with `diagonal_slots`), `diag_ub_equate`, `final_disjoint`, `open_kind_meets`
(≈ 45 prototype lines, ≈ 110 C lines). `merge` as a whole (return the union of the case
results; ≈ 130 prototype lines, ≈ 280 C lines) changes only subtype.jl 2616/2618 (now pass) and
3271 (expectation), and passes ambiguous and core; checking inference/inline next, since the
runtime entry has to squash a union of tuples for method matching.
Perf-only rules (`nominal`, `prune`, `independent`, `type_range`, `bounds_guard`,
`inhabited_unsat`): no result changes; removable at a speed cost.
`merge` off also passes Compiler/inference and Compiler/inline. Through the method-matching entry
(`jl_type_intersection_with_env`, which squashes a union of tuples), all 385,084 runtime queries give
the same result (14 respelled, `==`); among the quick pairs 4 synthetic rows get looser and 1
incomparable. Removing every rule with no measured effect plus `merge` saves ≈ 175 prototype
lines and ≈ 390 C lines (≈ 10% of the port): the port would still be ≈ 3,550 lines against
≈ 1,780 removed. Ablation alone does not reach "less code". The temporary C switches are
removed again.
Done (v59): removed `merge_pairs`, `merge_diagonal` (+ `diagonal_slots`, `same_slot`,
`tuple_with_union`), `diag_ub_equate`, `final_disjoint`. Kept, contrary to the first ablation
read: the shared-`where` part of `merge_cases` (show.jl:2539: without it
`Union{Nothing, AbstractVecOrMat{<:Integer}}` prints as two unions because the alias match needs
one `T`) and `open_kind_meets` (two runtime queries, `Type{Memory{T}} where T>:Int64` against the
kinds `TypeEgal`/`TypeEq`, go from `Union{}` to `A`; now a test). Prototype 1,799 → 1,733,
C 3,943 → 3,804. All 385,084 runtime queries give the same method-matching result; 56 quick
pairs respelled (`==`). subtype.jl 2616/2618 now pass, 3271 is now `== S` (tighter).

## Reference snapshot (2026-10-04, branch `type-boogaloo`)

Committed as a reference, not for landing. The port is opt-in (`JULIA_TYPE_MEET=1`); the test
expectations in this commit assume it (several fail with the old C). Static analysis:
`subtype.c` and the header are clean; `intersection.c` is not: the thread-safety pass wants
`JL_CANSAFEPOINT` annotations on most functions, and the GC checker reports 19 findings
(14 non-rooted arguments to functions that may GC, 3 arguments possibly GCed, 2 safepoints in
`JL_NOTSAFEPOINT` functions). The serialized corpora (`*.jls`) and the prototype snapshots are not
committed: `ablate_snap.jl` with `QUICK=1` regenerates `quickpairs.jls`; the runtime-query corpus
came from a temporary logging hook in `jl_type_intersection_env_s` that is not committed.
Scripts: `TwoPhase.jl` (prototype, v59), `c_diff.jl` (C vs prototype), `ablate_snap.jl` +
`classify.jl` + `census_pairs.jl` (snapshot/diff harness), `oracle.jl` + `oracle_fuzz.jl`
(finite-universe precision oracle), `c_time3.jl` (timing).
