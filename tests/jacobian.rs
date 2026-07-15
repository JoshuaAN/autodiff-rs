use tape::{
    drivers::{Jacobian, SparseStorage},
    tape::{Tape, Var},
};

fn build(tape: &Tape) -> (Vec<Var<'_>>, Vec<Var<'_>>) {
    let x = tape.param();
    let y = tape.param();
    let z = tape.param();
    let f0 = (x * y).sin() * z + x / (y + 3.0);
    let f1 = (x + y * z).cos() - z * z * x;
    (vec![x, y, z], vec![f0, f1])
}

#[cfg(test)]
mod tests {

    use super::*;
    const TOL: f64 = 1e-10;
    const FD_TOL: f64 = 1e-6; // central differences: O(h²) with h = 1e-6

    fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs().max(1.0),
            "{ctx}: got {actual}, expected {expected}"
        );
    }

    /// Structural invariants every eval result must satisfy.
    fn check_csc(m: &SparseStorage<f64>) {
        assert_eq!(m.col_ptrs.len(), m.num_cols() + 1);
        assert_eq!(m.col_ptrs[0], 0);
        assert_eq!(*m.col_ptrs.last().unwrap() as usize, m.values().len());
        assert_eq!(m.row_idx.len(), m.values().len());
        for col in 0..m.num_cols() {
            let s = m.col_ptrs[col] as usize;
            let e = m.col_ptrs[col + 1] as usize;
            assert!(s <= e, "col_ptrs not monotone at col {col}");
            let rows = &m.row_idx[s..e];
            assert!(
                rows.windows(2).all(|w| w[0] < w[1]),
                "rows not strictly sorted in col {col}"
            );
            assert!(rows.iter().all(|&r| (r as usize) < m.num_rows()));
        }
    }

    // ---------- exact-value tests ----------

    #[test]
    fn single_output_analytic() {
        // z = x²y + x·sin(y)
        // ∂z/∂x = 2xy + sin(y);  ∂z/∂y = x² + x·cos(y)
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * x * y + x * y.sin();

        let jac = Jacobian::new(&tape, &[x, y], &[z]);
        let (xv, yv) = (1.5, 2.3);
        let j = jac.eval(&[xv, yv]);
        check_csc(&j);

        assert_eq!((j.num_rows(), j.num_cols()), (1, 2));
        assert_close(j.get(0, 0), 2.0 * xv * yv + yv.sin(), TOL, "dz/dx");
        assert_close(j.get(0, 1), xv * xv + xv * yv.cos(), TOL, "dz/dy");
    }

    #[test]
    fn multi_output_analytic() {
        // Your main() case, checked against hand derivatives.
        // z = x²y + x·sin(y)          w = y + xy + y·cos(x)
        // ∂w/∂x = y - y·sin(x);       ∂w/∂y = 1 + x + cos(x)
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * x * y + x * y.sin();
        let w = y + x * y + y * x.cos();

        let jac = Jacobian::new(&tape, &[x, y], &[z, w]);
        let (xv, yv) = (1.5, 2.3);
        let j = jac.eval(&[xv, yv]);
        check_csc(&j);

        assert_eq!((j.num_rows(), j.num_cols()), (2, 2));
        assert_close(j.get(0, 0), 2.0 * xv * yv + yv.sin(), TOL, "dz/dx");
        assert_close(j.get(0, 1), xv * xv + xv * yv.cos(), TOL, "dz/dy");
        assert_close(j.get(1, 0), yv - yv * xv.sin(), TOL, "dw/dx");
        assert_close(j.get(1, 1), 1.0 + xv + xv.cos(), TOL, "dw/dy");
    }

    #[test]
    fn identity_map() {
        // outputs = inputs → J = I
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = tape.param();

        let jac = Jacobian::new(&tape, &[x, y, z], &[x, y, z]);
        let j = jac.eval(&[3.0, -1.0, 0.5]);
        check_csc(&j);

        assert_eq!(j.nnz(), 3);
        for i in 0..3 {
            for k in 0..3 {
                let expect = if i == k { 1.0 } else { 0.0 };
                assert_close(j.get(i, k), expect, TOL, &format!("J[{i}][{k}]"));
            }
        }
    }

    #[test]
    fn division_and_neg() {
        // z = -x / y:  ∂z/∂x = -1/y,  ∂z/∂y = x/y²
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = -x / y;

        let jac = Jacobian::new(&tape, &[x, y], &[z]);
        let (xv, yv) = (2.0, 4.0);
        let j = jac.eval(&[xv, yv]);
        check_csc(&j);

        assert_close(j.get(0, 0), -1.0 / yv, TOL, "dz/dx");
        assert_close(j.get(0, 1), xv / (yv * yv), TOL, "dz/dy");
    }

    #[test]
    fn scalar_var_mixed_ops() {
        // Exercise f64-op-Var and Var-op-f64 paths.
        // z = 3x + x·2 - 1/x:  dz/dx = 5 + 1/x²
        let tape = Tape::new();
        let x = tape.param();
        let z = 3.0 * x + x * 2.0 - 1.0 / x;

        let jac = Jacobian::new(&tape, &[x], &[z]);
        let xv = 0.7;
        let j = jac.eval(&[xv]);
        assert_close(j.get(0, 0), 5.0 + 1.0 / (xv * xv), TOL, "dz/dx");
    }

    #[test]
    fn shared_subexpression() {
        // s = x·y used by both outputs — fan-out through one node.
        // z = s + s;  w = s·s
        // ∂z/∂x = 2y, ∂z/∂y = 2x, ∂w/∂x = 2xy², ∂w/∂y = 2x²y
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let s = x * y;
        let z = s + s;
        let w = s * s;

        let jac = Jacobian::new(&tape, &[x, y], &[z, w]);
        let (xv, yv) = (1.2, -0.8);
        let j = jac.eval(&[xv, yv]);
        check_csc(&j);

        assert_close(j.get(0, 0), 2.0 * yv, TOL, "dz/dx");
        assert_close(j.get(0, 1), 2.0 * xv, TOL, "dz/dy");
        assert_close(j.get(1, 0), 2.0 * xv * yv * yv, TOL, "dw/dx");
        assert_close(j.get(1, 1), 2.0 * xv * xv * yv, TOL, "dw/dy");
    }

    // ---------- structure tests ----------

    #[test]
    fn block_sparsity() {
        // z depends only on x; w only on y → off-diagonal entries absent.
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * x;
        let w = y.sin();

        let jac = Jacobian::new(&tape, &[x, y], &[z, w]);
        let j = jac.eval(&[2.0, 1.0]);
        check_csc(&j);

        assert_eq!(j.nnz(), 2, "cross terms should be structurally absent");
        assert_close(j.get(0, 0), 4.0, TOL, "dz/dx");
        assert_close(j.get(1, 1), 1.0f64.cos(), TOL, "dw/dy");
        assert_eq!(j.get(0, 1), 0.0);
        assert_eq!(j.get(1, 0), 0.0);
    }

    #[test]
    fn unused_input_gets_empty_column() {
        // y listed as input but unused: column must exist and be empty.
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * x;

        let jac = Jacobian::new(&tape, &[x, y], &[z]);
        let j = jac.eval(&[3.0, 99.0]);
        check_csc(&j);

        assert_eq!(j.num_cols(), 2);
        assert_eq!(j.col_ptrs[1], j.col_ptrs[2], "column for y must be empty");
        assert_close(j.get(0, 0), 6.0, TOL, "dz/dx");
    }

    #[test]
    fn constant_output_row() {
        // One output is a constant: its row contributes no entries.
        let tape = Tape::new();
        let x = tape.param();
        let c = tape.constant(7.0);
        let z = x * x;

        let jac = Jacobian::new(&tape, &[x], &[z, c]);
        let j = jac.eval(&[2.0]);
        check_csc(&j);

        assert_eq!(j.num_rows(), 2);
        assert_close(j.get(0, 0), 4.0, TOL, "dz/dx");
        assert_eq!(j.get(1, 0), 0.0, "constant row must be zero");
    }

    #[test]
    fn output_is_an_input() {
        // An output that *is* a parameter: ∂x/∂x = 1, ∂x/∂y = 0.
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * y;

        let jac = Jacobian::new(&tape, &[x, y], &[x, z]);
        let j = jac.eval(&[3.0, 5.0]);
        check_csc(&j);

        assert_close(j.get(0, 0), 1.0, TOL, "dx/dx");
        assert_eq!(j.get(0, 1), 0.0, "dx/dy");
        assert_close(j.get(1, 0), 5.0, TOL, "dz/dx");
        assert_close(j.get(1, 1), 3.0, TOL, "dz/dy");
    }

    // ---------- finite-difference oracle ----------

    /// Central-difference Jacobian of the primal function.
    fn fd_jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], m: usize) -> Vec<Vec<f64>> {
        let h = 1e-6;
        let n = x.len();
        let mut jac = vec![vec![0.0; n]; m];
        let mut xp = x.to_vec();
        for jcol in 0..n {
            xp[jcol] = x[jcol] + h;
            let fp = f(&xp);
            xp[jcol] = x[jcol] - h;
            let fm = f(&xp);
            xp[jcol] = x[jcol];
            for irow in 0..m {
                jac[irow][jcol] = (fp[irow] - fm[irow]) / (2.0 * h);
            }
        }
        jac
    }

    #[test]
    fn matches_finite_differences() {
        // Same math as plain f64, for the oracle.
        let primal = |v: &[f64]| -> Vec<f64> {
            let (x, y, z) = (v[0], v[1], v[2]);
            vec![
                (x * y).sin() * z + x / (y + 3.0),
                (x + y * z).cos() - z * z * x,
            ]
        };

        let tape = Tape::new();
        let (inputs, outputs) = build(&tape);
        let jac = Jacobian::new(&tape, &inputs, &outputs);

        for point in [[0.5, 1.0, -2.0], [1.5, 2.3, 0.7], [-1.0, 0.1, 3.0]] {
            let j = jac.eval(&point);
            check_csc(&j);
            let fd = fd_jacobian(&primal, &point, 2);
            for i in 0..2 {
                for k in 0..3 {
                    assert_close(
                        j.get(i, k),
                        fd[i][k],
                        FD_TOL,
                        &format!("J[{i}][{k}] at {point:?}"),
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_eval_is_consistent() {
        // Same point twice → identical results (catches state leaking
        // between sweeps, e.g. a seed entry not reset to 0).
        let tape = Tape::new();
        let x = tape.param();
        let y = tape.param();
        let z = x * y.sin() + y * x.cos();

        let jac = Jacobian::new(&tape, &[x, y], &[z]);
        let a = jac.eval(&[1.1, 2.2]);
        let b = jac.eval(&[1.1, 2.2]);
        assert_eq!(a.values(), b.values());
        assert_eq!(a.row_idx, b.row_idx);
        assert_eq!(a.col_ptrs, b.col_ptrs);
    }
}
