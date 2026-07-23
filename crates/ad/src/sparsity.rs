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

    pub fn from_dense(rows: usize, cols: usize) -> Self {
        let col_idx = (0..=cols).map(|c| (c * rows) as u32).collect();
        let row = (0..cols).flat_map(|_| 0..rows as u32).collect();
        Sparsity {
            num_rows: rows,
            num_cols: cols,
            col_idx,
            row,
        }
    }

    // pub fn from_pairs(rows: usize, cols: usize, pairs: &Vec<(usize, usize)>) -> Self {
    //     let mut col_idx = Vec::new();
    //     let mut row = Vec::new();
    //     for (r, c) in pairs {

    //     }

    //     Sparsity {
    //         num_rows: rows,
    //         num_cols: cols,
    //         col_idx,
    //         row,
    //     }
    // }

    pub fn from_columns(rows: usize, cols: usize, columns: Vec<Vec<u32>>) -> Self {
        todo!("Implement Sparsity::from_columns")
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

    pub fn transpose(&self) -> Sparsity {
        todo!("Implement Sparsity::transpose")
    }

    /// Greedy distance-2 coloring of the columns (Curitis-Powell-Reid).
    pub fn uni_coloring(&self) -> Coloring {
        let n = self.num_cols();
        let t = self.transpose();
        let mut color = vec![u32::MAX; n];

        // blocking[c] = j implies color c is blocked for column j.
        let mut blocking = vec![usize::MAX; n + 1];

        for col in 0..n {
            for i in self.col_range(col) {
                let row = self.row[i] as usize;
                // Iterate through all columns which conflict with the current column.
                for j in t.col_range(row) {
                    let neighbor = t.row[j] as usize;
                    let c = color[neighbor];
                    if c != u32::MAX {
                        blocking[c as usize] = col;
                    }
                }
            }
            let mut c = 0;
            while blocking[c] == col {
                c += 1;
            }
            color[col] = c as u32;
        }

        let num_colors = color.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        Coloring {
            num_colors,
            color,
        }
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

pub struct Coloring {
    num_colors: usize,
    color: Vec<u32>,
}

impl Coloring {
    pub fn num_colors(&self) -> usize {
        self.num_colors
    }

    pub fn colors(&self) -> &[u32] {
        &self.color
    }

    pub fn color(&self, col: usize) -> u32 {
        self.color[col]
    }
}