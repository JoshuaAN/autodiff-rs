#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sparsity {
    num_rows: usize,
    num_cols: usize,
    col_idx: Vec<u32>,
    row: Vec<u32>,
}

impl Sparsity {
    pub fn new(
        num_rows: usize,
        num_cols: usize,
        col_idx: Vec<u32>,
        row: Vec<u32>,
    ) -> Result<Self, SparsityError> {
        if col_idx.len() != num_cols + 1 {
            return Err(SparsityError::ColIdxWrongLength {
                expected: num_cols + 1,
                got: col_idx.len(),
            });
        }
        if col_idx[0] != 0 {
            return Err(SparsityError::ColIdxFirstNotZero);
        }
        for c in 0..num_cols {
            if col_idx[c] > col_idx[c + 1] {
                return Err(SparsityError::ColIdxNotMonotone { col: c });
            }
        }
        let nnz = *col_idx.last().unwrap() as usize;
        if row.len() != nnz {
            return Err(SparsityError::RowLengthMismatch {
                expected: nnz,
                got: row.len(),
            });
        }
        for c in 0..num_cols {
            let range = col_idx[c] as usize..col_idx[c + 1] as usize;
            for k in range {
                if row[k] as usize >= num_rows {
                    return Err(SparsityError::RowOutOfRange {
                        col: c,
                        position: k,
                        row: row[k],
                    });
                }
                if k > col_idx[c] as usize && row[k - 1] >= row[k] {
                    return Err(SparsityError::RowsNotStrictlyIncreasing {
                        col: c,
                        position: k,
                    });
                }
            }
        }
        Ok(Sparsity {
            num_rows,
            num_cols,
            col_idx,
            row,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    pub fn nnz(&self) -> usize {
        *self.col_idx.last().unwrap() as usize
    }

    pub fn col_range(&self, c: usize) -> std::ops::Range<usize> {
        self.col_idx[c] as usize..self.col_idx[c + 1] as usize
    }

    pub fn rows(&self) -> &[u32] {
        &self.row
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparsityError {
    ColIdxWrongLength {
        expected: usize,
        got: usize,
    },
    ColIdxFirstNotZero,
    ColIdxNotMonotone {
        col: usize,
    },
    RowLengthMismatch {
        expected: usize,
        got: usize,
    },
    RowOutOfRange {
        col: usize,
        position: usize,
        row: u32,
    },
    RowsNotStrictlyIncreasing {
        col: usize,
        position: usize,
    },
}
