use crate::{
    function::Function,
    tape::{Tape, Var},
};

/// Represents a sparse matrix in Compressed Sparse Column (CSC) format.
#[derive(Default)]
pub struct SparseStorage<T> {
    num_rows: usize,
    num_cols: usize,
    pub col_ptrs: Vec<u32>,
    pub row_idx: Vec<u32>,
    values: Vec<T>,
}

impl<T> SparseStorage<T> {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Number of explicitly stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Column pointers; length `num_cols() + 1`. Column `j`'s entries
    /// occupy `col_ptrs()[j]..col_ptrs()[j + 1]` in `row_indices()`
    /// and `values()`.
    pub fn col_ptrs(&self) -> &[u32] {
        &self.col_ptrs
    }

    /// Row index of each stored entry, sorted within each column.
    pub fn row_indices(&self) -> &[u32] {
        &self.row_idx
    }

    /// Value of each stored entry, in the same order as `row_indices()`.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Iterator over stored entries as `(row, col, &value)`,
    /// in column-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.num_cols).flat_map(move |col| {
            let start = self.col_ptrs[col] as usize;
            let end = self.col_ptrs[col + 1] as usize;
            (start..end).map(move |k| (self.row_idx[k] as usize, col, &self.values[k]))
        })
    }
}

impl<T: Default + Copy> SparseStorage<T> {
    /// Entry (i, j), or default (zero) if structurally absent.
    pub fn get(&self, row: usize, col: usize) -> T {
        let start = self.col_ptrs[col] as usize;
        let end = self.col_ptrs[col + 1] as usize;
        self.row_idx[start..end]
            .iter()
            .position(|&r| r as usize == row)
            .map(|k| self.values[start + k])
            .unwrap_or_default()
    }
}

use std::fmt;

impl<T: fmt::Display + Default + Copy> fmt::Display for SparseStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Densify: CSC is column-major, but display is row-major,
        // so gather everything up front.
        let mut dense = vec![T::default(); self.num_rows * self.num_cols];
        for col in 0..self.num_cols {
            let start = self.col_ptrs[col] as usize;
            let end = self.col_ptrs[col + 1] as usize;
            for k in start..end {
                let row = self.row_idx[k] as usize;
                dense[row * self.num_cols + col] = self.values[k];
            }
        }

        // Pre-format every entry so columns can be aligned. Respects a
        // precision flag, e.g. {:.3}.
        let cells: Vec<String> = dense
            .iter()
            .map(|v| match f.precision() {
                Some(p) => format!("{v:.p$}"),
                None => format!("{v}"),
            })
            .collect();

        // One width per column.
        let mut widths = vec![0usize; self.num_cols];
        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                widths[col] = widths[col].max(cells[row * self.num_cols + col].len());
            }
        }

        for row in 0..self.num_rows {
            write!(f, "[")?;
            for col in 0..self.num_cols {
                if col > 0 {
                    write!(f, "  ")?;
                }
                write!(
                    f,
                    "{:>w$}",
                    cells[row * self.num_cols + col],
                    w = widths[col]
                )?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

pub struct Gradient {
    grad: Function,
}

pub struct Jacobian {
    num_rows: usize,
    num_cols: usize,
    jvp: Function,
}

pub struct Hessian {}

impl Gradient {
    pub fn new(tape: &Tape, inputs: &[Var], output: Var) -> Self {
        Self {
            grad: tape.compile(inputs, &[output]).backward(),
        }
    }

    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        self.grad.eval(x)
    }
}

impl Jacobian {
    pub fn new(tape: &Tape, inputs: &[Var], outputs: &[Var]) -> Self {
        Self {
            num_rows: outputs.len(),
            num_cols: inputs.len(),
            jvp: tape.compile(inputs, outputs).forward(),
        }
    }

    pub fn eval(&self, x: &[f64]) -> SparseStorage<f64> {
        let n = self.num_cols;
        let m = self.num_rows;

        let mut mat = SparseStorage {
            num_rows: m,
            num_cols: n,
            col_ptrs: Vec::with_capacity(n + 1),
            row_idx: Vec::new(),
            values: Vec::new(),
        };
        mat.col_ptrs.push(0);

        let mut args = vec![0.0; 2 * n];
        args[..n].copy_from_slice(x);

        for j in 0..n {
            args[n + j] = 1.0;

            let out = self.jvp.eval(&args);
            let tangents = &out[m..]; // returns are [primals..., tangents...]

            for (i, &v) in tangents.iter().enumerate() {
                if v != 0.0 {
                    mat.row_idx.push(i as u32);
                    mat.values.push(v);
                }
            }

            args[n + j] = 0.0;
            mat.col_ptrs.push(mat.row_idx.len() as u32);
        }
        mat
    }
}

impl Hessian {
    pub fn new(tape: &Tape, inputs: &[Var], output: Var) -> Self {
        todo!("Implement Hessian new")
    }

    pub fn eval(&self, x: &[f64]) -> SparseStorage<f64> {
        todo!("Implement Hessian evaluation")
    }
}
