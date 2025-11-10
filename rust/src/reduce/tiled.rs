use crate::simd;
use crate::tiling::TileSpec;

const DIRECT_REDUCTION_LIMIT: usize = 1 << 20;
const SMALL_DIRECT_THRESHOLD: usize = 1 << 12;

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

    if elements <= DIRECT_REDUCTION_LIMIT {
        let total = direct_sum_f64(data);
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

    if elements <= DIRECT_REDUCTION_LIMIT {
        let total = direct_sum_f32(data);
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

fn direct_sum_f64(data: &[f64]) -> f64 {
    if data.len() <= SMALL_DIRECT_THRESHOLD {
        return data.iter().copied().sum();
    }
    simd::reduce_sum_f64(data, 1).unwrap_or_else(|| data.iter().copied().sum())
}

fn direct_sum_f32(data: &[f32]) -> f64 {
    if data.len() <= SMALL_DIRECT_THRESHOLD {
        return data.iter().map(|&v| v as f64).sum();
    }
    simd::reduce_sum_f32(data, 1).unwrap_or_else(|| data.iter().map(|&v| v as f64).sum())
}

fn sequential_sum_f64(
    data: &[f64],
    cols: usize,
    spec: &TileSpec,
    buffer: usize,
    rows: usize,
) -> f64 {
    let tile_cols = aligned_tile_cols(spec, cols);
    let slots = buffer.max(1).min(rows.max(1));
    let mut partials = vec![0.0f64; slots];
    let mut cursor = 0usize;

    for row in data.chunks(cols) {
        let mut col = 0usize;
        while col < row.len() {
            let end = (col + tile_cols).min(row.len());
            let slice = &row[col..end];
            let sum = simd::reduce_sum_f64(slice, spec.accumulators)
                .unwrap_or_else(|| slice.iter().sum::<f64>());
            partials[cursor] += sum;
            cursor = (cursor + 1) % partials.len();
            col = end;
        }
    }

    partials.into_iter().sum()
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
    let mut cursor = 0usize;

    for row in data.chunks(cols) {
        let mut col = 0usize;
        while col < row.len() {
            let end = (col + tile_cols).min(row.len());
            let slice = &row[col..end];
            let sum = simd::reduce_sum_f32(slice, spec.accumulators)
                .unwrap_or_else(|| slice.iter().map(|&v| v as f64).sum::<f64>());
            partials[cursor] += sum;
            cursor = (cursor + 1) % partials.len();
            col = end;
        }
    }

    partials.into_iter().sum()
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
    pool.install(|| {
        data.par_chunks(chunk_elems.max(1))
            .map(|chunk| {
                let rows = chunk.len() / cols;
                sequential_sum_f64(chunk, cols, spec, buffer, rows)
            })
            .sum()
    })
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
    pool.install(|| {
        data.par_chunks(chunk_elems.max(1))
            .map(|chunk| {
                let rows = chunk.len() / cols;
                sequential_sum_f32(chunk, cols, spec, buffer, rows)
            })
            .sum()
    })
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
