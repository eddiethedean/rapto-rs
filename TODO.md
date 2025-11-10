# Raptors TODO

## Short-Term Enhancements

- [x] Add broadcasting-aware arithmetic for mismatched shapes.
- [x] Support additional numeric dtypes (e.g., `i32`, `f32`) in `RustArray`.
- [ ] Implement slicing/indexing utilities on the Python layer.
- [ ] Improve NumPy interoperability with zero-copy views where safe.
- [ ] Validate integer reduction rules (e.g., configurable rounding for mean) and document error semantics.

## Testing & Tooling

- [ ] Introduce property-based tests for arithmetic and reductions.
- [ ] Add benchmarks comparing 1-D and 2-D operations to NumPy.

## Documentation & DX

- [ ] Expand usage guides covering broadcasting and advanced indexing.
- [ ] Document contribution workflow, coding standards, and release steps.
- [ ] Add a dtype reference section covering float/int behaviors and conversion caveats.


