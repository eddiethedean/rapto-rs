use crate::simd;
use crate::tiling::TileSpec;

const DIRECT_REDUCTION_LIMIT: usize = 1 << 20;
const DIRECT_PARALLEL_MIN_ELEMENTS: usize = 1 << 21;
const DIRECT_MIN_ROWS_PER_CHUNK: usize = 96;
pub(crate) const SMALL_DIRECT_THRESHOLD: usize = 1 << 12;

fn recommended_accumulators(len: usize, max: usize) -> usize {
    if len >= 1 << 22 {
        max
    } else if len >= 1 << 20 {
        max.saturating_sub(1).max(1)
    } else if len >= 1 << 18 {
        (max / 2).max(1)
    } else if len >= 1 << 16 {
        3.min(max).max(1)
    } else {
        1
    }
}

fn kahan_add(sum: &mut f64, comp: &mut f64, value: f64) {
    let y = value - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

#[derive(Clone, Copy)]
pub enum GlobalOp {
    Sum,
    Mean,
}

impl GlobalOp {
    fn finalize(&self, total: f64, elements: usize) -> f64 {
        match self {
            GlobalOp::Sum => total,
            GlobalOp::Mean => {
                if elements == 0 {
                    0.0
                } else {
                    total / elements as f64
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReduceOutcome {
    pub value: f64,
    pub tiles_processed: usize,
    pub parallel: bool,
    pub partial_buffer: usize,
}

pub fn reduce_full_f64(
    data: &[f64],
    rows: usize,
    cols: usize,
    op: GlobalOp,
    pool: Option<&rayon::ThreadPool>,
    allow_parallel: bool,
) -> ReduceOutcome {
    let elements = rows.saturating_mul(cols);
    if data.is_empty() || elements == 0 {
        return ReduceOutcome {
            value: op.finalize(0.0, elements),
            tiles_processed: 0,
            parallel: false,
            partial_buffer: 1,
        };
    }

    let allow_parallel = allow_parallel || elements >= DIRECT_PARALLEL_MIN_ELEMENTS;

    if elements <= DIRECT_REDUCTION_LIMIT {
        let direct_pool = if allow_parallel { pool } else { None };
        let total = direct_sum_f64(data, rows, cols, direct_pool);
        return ReduceOutcome {
            value: op.finalize(total, elements),
            tiles_processed: 1,
            parallel: false,
            partial_buffer: 1,
        };
    }

    let spec = TileSpec::for_shape(rows.max(1), cols.max(1));
    let tiles_processed = estimate_tiles(rows, cols, &spec);
    let partial_buffer = partial_buffer_len(&spec, rows);

    let mut parallel_used = false;
    let total = if allow_parallel {
        if let Some(pool) = pool {
            if rows >= spec.row_block * 2 {
                parallel_used = true;
                parallel_sum_f64(data, cols, &spec, pool, partial_buffer)
            } else {
                sequential_sum_f64(data, cols, &spec, partial_buffer, rows)
            }
        } else {
            sequential_sum_f64(data, cols, &spec, partial_buffer, rows)
        }
    } else {
        sequential_sum_f64(data, cols, &spec, partial_buffer, rows)
    };

    ReduceOutcome {
        value: op.finalize(total, elements),
        tiles_processed,
        parallel: parallel_used,
        partial_buffer,
    }
}

pub fn reduce_full_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    op: GlobalOp,
    pool: Option<&rayon::ThreadPool>,
    allow_parallel: bool,
) -> ReduceOutcome {
    let elements = rows.saturating_mul(cols);
    if data.is_empty() || elements == 0 {
        return ReduceOutcome {
            value: op.finalize(0.0, elements),
            tiles_processed: 0,
            parallel: false,
            partial_buffer: 1,
        };
    }

    let allow_parallel = allow_parallel || elements >= DIRECT_PARALLEL_MIN_ELEMENTS;

    if elements <= DIRECT_REDUCTION_LIMIT {
        let direct_pool = if allow_parallel { pool } else { None };
        let total = direct_sum_f32(data, rows, cols, direct_pool);
        return ReduceOutcome {
            value: op.finalize(total, elements),
            tiles_processed: 1,
            parallel: false,
            partial_buffer: 1,
        };
    }

    let spec = TileSpec::for_shape(rows.max(1), cols.max(1));
    let tiles_processed = estimate_tiles(rows, cols, &spec);
    let partial_buffer = partial_buffer_len(&spec, rows);

    let mut parallel_used = false;
    let total = if allow_parallel {
        if let Some(pool) = pool {
            if rows >= spec.row_block * 2 {
                parallel_used = true;
                parallel_sum_f32(data, cols, &spec, pool, partial_buffer)
            } else {
                sequential_sum_f32(data, cols, &spec, partial_buffer, rows)
            }
        } else {
            sequential_sum_f32(data, cols, &spec, partial_buffer, rows)
        }
    } else {
        sequential_sum_f32(data, cols, &spec, partial_buffer, rows)
    };

    ReduceOutcome {
        value: op.finalize(total, elements),
        tiles_processed,
        parallel: parallel_used,
        partial_buffer,
    }
}

fn direct_sum_f64(data: &[f64], rows: usize, cols: usize, pool: Option<&rayon::ThreadPool>) -> f64 {
    if data.len() <= SMALL_DIRECT_THRESHOLD {
        return data.iter().copied().sum();
    }
    if let Some(pool) = pool {
        let elements = rows.saturating_mul(cols);
        let threads = pool.current_num_threads().max(1);
        let chunk_rows = (rows / threads).max(1);
        if elements >= DIRECT_PARALLEL_MIN_ELEMENTS || chunk_rows >= DIRECT_MIN_ROWS_PER_CHUNK {
            let chunk_len = cols
                .saturating_mul(chunk_rows)
                .max(cols.saturating_mul(DIRECT_MIN_ROWS_PER_CHUNK.min(rows)));
            return pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(chunk_len)
                    .map(|chunk| {
                        let acc = recommended_accumulators(chunk.len(), 8);
                        simd::reduce_sum_f64(chunk, acc)
                            .unwrap_or_else(|| chunk.iter().copied().sum())
                    })
                    .sum()
            });
        }
    }
    let acc = recommended_accumulators(data.len(), 8);
    simd::reduce_sum_f64(data, acc).unwrap_or_else(|| data.iter().copied().sum())
}

fn direct_sum_f32(data: &[f32], rows: usize, cols: usize, pool: Option<&rayon::ThreadPool>) -> f64 {
    if data.len() <= SMALL_DIRECT_THRESHOLD {
        return data.iter().map(|&v| v as f64).sum();
    }
    if let Some(pool) = pool {
        let elements = rows.saturating_mul(cols);
        let threads = pool.current_num_threads().max(1);
        let chunk_rows = (rows / threads).max(1);
        if elements >= DIRECT_PARALLEL_MIN_ELEMENTS || chunk_rows >= DIRECT_MIN_ROWS_PER_CHUNK {
            let chunk_len = cols
                .saturating_mul(chunk_rows)
                .max(cols.saturating_mul(DIRECT_MIN_ROWS_PER_CHUNK.min(rows)));
            return pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(chunk_len)
                    .map(|chunk| {
                        let acc = recommended_accumulators(chunk.len(), 8);
                        simd::reduce_sum_f32(chunk, acc)
                            .unwrap_or_else(|| chunk.iter().map(|&v| v as f64).sum::<f64>())
                    })
                    .sum()
            });
        }
    }
    let acc = recommended_accumulators(data.len(), 8);
    simd::reduce_sum_f32(data, acc).unwrap_or_else(|| data.iter().map(|&v| v as f64).sum())
}

fn sequential_sum_f64(
    data: &[f64],
    cols: usize,
    spec: &TileSpec,
    buffer: usize,
    rows: usize,
) -> f64 {
    if cols >= 1024 {
        let mut total = 0.0f64;
        let mut comp = 0.0f64;
        for row in data.chunks_exact(cols) {
            let sum = simd::reduce_sum_f64(row, recommended_accumulators(row.len(), 8))
                .unwrap_or_else(|| row.iter().copied().sum());
            kahan_add(&mut total, &mut comp, sum);
        }
        return total + comp;
    }
    let tile_cols = aligned_tile_cols(spec, cols);
    let slots = buffer.max(1).min(rows.max(1));
    let mut partials = vec![0.0f64; slots];
    let mut comps = vec![0.0f64; slots];
    let mut cursor = 0usize;

    for row in data.chunks(cols) {
        let mut col = 0usize;
        while col < row.len() {
            let end = (col + tile_cols).min(row.len());
            let slice = &row[col..end];
            let sum = simd::reduce_sum_f64(slice, spec.accumulators)
                .unwrap_or_else(|| slice.iter().sum::<f64>());
            kahan_add(&mut partials[cursor], &mut comps[cursor], sum);
            cursor = (cursor + 1) % partials.len();
            col = end;
        }
    }

    let mut total = 0.0f64;
    let mut comp = 0.0f64;
    for (value, c) in partials.into_iter().zip(comps.into_iter()) {
        kahan_add(&mut total, &mut comp, value + c);
    }
    total
}

fn sequential_sum_f32(
    data: &[f32],
    cols: usize,
    spec: &TileSpec,
    buffer: usize,
    rows: usize,
) -> f64 {
    let tile_cols = aligned_tile_cols(spec, cols);
    let slots = buffer.max(1).min(rows.max(1));
    let mut partials = vec![0.0f64; slots];
    let mut comps = vec![0.0f64; slots];
    let mut cursor = 0usize;

    for row in data.chunks(cols) {
        let mut col = 0usize;
        while col < row.len() {
            let end = (col + tile_cols).min(row.len());
            let slice = &row[col..end];
            let sum = simd::reduce_sum_f32(slice, spec.accumulators)
                .unwrap_or_else(|| slice.iter().map(|&v| v as f64).sum::<f64>());
            kahan_add(&mut partials[cursor], &mut comps[cursor], sum);
            cursor = (cursor + 1) % partials.len();
            col = end;
        }
    }

    let mut total = 0.0f64;
    let mut comp = 0.0f64;
    for (value, c) in partials.into_iter().zip(comps.into_iter()) {
        kahan_add(&mut total, &mut comp, value + c);
    }
    total
}

fn parallel_sum_f64(
    data: &[f64],
    cols: usize,
    spec: &TileSpec,
    pool: &rayon::ThreadPool,
    buffer: usize,
) -> f64 {
    use rayon::prelude::*;

    let chunk_elems = cols.saturating_mul(spec.row_block.max(1));
    let (sum, comp) = pool.install(|| {
        data.par_chunks(chunk_elems.max(1))
            .fold(
                || (0.0f64, 0.0f64),
                |mut acc, chunk| {
                    let rows = chunk.len() / cols;
                    let value = sequential_sum_f64(chunk, cols, spec, buffer, rows);
                    kahan_add(&mut acc.0, &mut acc.1, value);
                    acc
                },
            )
            .reduce(
                || (0.0f64, 0.0f64),
                |mut acc, value| {
                    kahan_add(&mut acc.0, &mut acc.1, value.0);
                    if value.1 != 0.0 {
                        kahan_add(&mut acc.0, &mut acc.1, value.1);
                    }
                    acc
                },
            )
    });
    sum + comp
}

fn parallel_sum_f32(
    data: &[f32],
    cols: usize,
    spec: &TileSpec,
    pool: &rayon::ThreadPool,
    buffer: usize,
) -> f64 {
    use rayon::prelude::*;

    let chunk_elems = cols.saturating_mul(spec.row_block.max(1));
    let (sum, comp) = pool.install(|| {
        data.par_chunks(chunk_elems.max(1))
            .fold(
                || (0.0f64, 0.0f64),
                |mut acc, chunk| {
                    let rows = chunk.len() / cols;
                    let value = sequential_sum_f32(chunk, cols, spec, buffer, rows);
                    kahan_add(&mut acc.0, &mut acc.1, value);
                    acc
                },
            )
            .reduce(
                || (0.0f64, 0.0f64),
                |mut acc, value| {
                    kahan_add(&mut acc.0, &mut acc.1, value.0);
                    if value.1 != 0.0 {
                        kahan_add(&mut acc.0, &mut acc.1, value.1);
                    }
                    acc
                },
            )
    });
    sum + comp
}

fn aligned_tile_cols(spec: &TileSpec, cols: usize) -> usize {
    if cols == 0 {
        return 0;
    }
    let align = spec.lane_width.max(1);
    let base = spec.col_block.max(align);
    let aligned = ((base + align - 1) / align) * align;
    aligned.min(cols.max(align))
}

fn partial_buffer_len(spec: &TileSpec, rows: usize) -> usize {
    spec.accumulators.clamp(2, 32).min(rows.max(1))
}

fn estimate_tiles(rows: usize, cols: usize, spec: &TileSpec) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    let row_tiles = (rows + spec.row_block - 1) / spec.row_block;
    let col_tiles = (cols + spec.col_block - 1) / spec.col_block;
    row_tiles * col_tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_accumulators_increase_with_input_size() {
        assert_eq!(recommended_accumulators(1 << 10, 8), 1);
        assert_eq!(recommended_accumulators(1 << 18, 8), 4);
        assert_eq!(recommended_accumulators(1 << 20, 8), 7);
        assert_eq!(recommended_accumulators(1 << 22, 8), 8);
    }

    #[test]
    fn direct_sum_matches_scalar_sum_for_small_inputs() {
        let data: Vec<f64> = (0..16).map(|value| value as f64).collect();
        let expected: f64 = data.iter().copied().sum();
        let total = direct_sum_f64(&data, 4, 4, None);
        assert_eq!(total, expected);
    }

    #[test]
    fn reduce_full_handles_mean_operation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let outcome = reduce_full_f64(&data, 2, 2, GlobalOp::Mean, None, false);
        assert_eq!(outcome.value, 2.5);
        assert_eq!(outcome.tiles_processed, 1);
        assert!(!outcome.parallel);
    }

    #[test]
    fn partial_buffer_len_respects_bounds() {
        let spec = TileSpec::for_shape(512, 256);
        assert_eq!(partial_buffer_len(&spec, 1), 1);
        let expected = spec.accumulators.clamp(2, 32);
        assert_eq!(partial_buffer_len(&spec, 1024), expected);
    }

    #[test]
    fn estimate_tiles_returns_zero_when_empty() {
        let spec = TileSpec::for_shape(0, 0);
        assert_eq!(estimate_tiles(0, 10, &spec), 0);
        assert_eq!(estimate_tiles(10, 0, &spec), 0);
    }

    #[test]
    fn estimate_tiles_counts_row_and_column_blocks() {
        let spec = TileSpec::for_shape(64, 64);
        let tiles = estimate_tiles(128, 128, &spec);
        assert!(tiles >= 4);
    }
}
