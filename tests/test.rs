#[cfg(test)]
mod tests {
    use autodiff::{
        context::Context,
        grad::gradient,
        interpreter::eval,
        jacobian::{hessian, jacobian},
        tape::Tape,
        var::VarId,
    };
    use index_vec::IndexVec;

    use super::*;

    fn eval1(tape: &Tape, xs: &[f64]) -> Vec<f64> {
        let inputs: IndexVec<VarId, f64> = xs.iter().copied().collect();
        let mut out = Vec::new();
        eval(tape, &inputs, &mut out);
        out
    }

    #[test]
    fn grad_matches_closed_form_and_fd() {
        let ctx = Context::new();
        let x = ctx.var();
        let y = ctx.var();
        let r2 = &x * &x + &y * &y;
        let f = r2.sqrt().sin() / (1.0 + r2.sqrt());

        let fwd = ctx.lower(&[&f]);
        let g = gradient(&fwd, 0, ctx.n_vars());

        let p = [3.0, 4.0];
        let out = eval1(&g, &p); // [f, df/dx, df/dy]

        // closed form: df/dx = (x/r) * (cos r (1+r) - sin r)/(1+r)^2, r = 5
        let r: f64 = 5.0;
        let gp = (r.cos() * (1.0 + r) - r.sin()) / (1.0 + r).powi(2);
        assert!((out[1] - 0.6 * gp).abs() < 1e-12);
        assert!((out[2] - 0.8 * gp).abs() < 1e-12);

        // finite differences
        let h = 1e-6;
        for i in 0..2 {
            let (mut lo, mut hi) = (p, p);
            lo[i] -= h;
            hi[i] += h;
            let fd = (eval1(&fwd, &hi)[0] - eval1(&fwd, &lo)[0]) / (2.0 * h);
            assert!(
                (out[1 + i] - fd).abs() < 1e-6,
                "var {i}: {} vs {}",
                out[1 + i],
                fd
            );
        }
    }

    #[test]
    fn jacobian_matches_closed_form_and_fd() {
        // f0 = x^2 * y, f1 = sin(x*y) — two outputs, two inputs.
        let ctx = Context::new();
        let x = ctx.var();
        let y = ctx.var();
        let f0 = &x * &x * &y;
        let f1 = (&x * &y).sin();

        let fwd = ctx.lower(&[&f0, &f1]);
        let n = ctx.n_vars();
        let j = jacobian(&fwd, n);

        let p = [1.3, 0.7];
        let out = eval1(&j, &p); // [f0, f1, df0/dx, df0/dy, df1/dx, df1/dy]
        let (x, y) = (p[0], p[1]);

        // closed form
        let expected = [
            2.0 * x * y,       // df0/dx
            x * x,             // df0/dy
            y * (x * y).cos(), // df1/dx
            x * (x * y).cos(), // df1/dy
        ];
        for (k, &e) in expected.iter().enumerate() {
            assert!(
                (out[2 + k] - e).abs() < 1e-12,
                "entry {k}: {} vs {e}",
                out[2 + k]
            );
        }

        // finite differences against every entry, both outputs
        let h = 1e-6;
        for out_i in 0..2 {
            for var_j in 0..2 {
                let (mut lo, mut hi) = (p, p);
                lo[var_j] -= h;
                hi[var_j] += h;
                let fd = (eval1(&fwd, &hi)[out_i] - eval1(&fwd, &lo)[out_i]) / (2.0 * h);
                let got = out[2 + out_i * 2 + var_j];
                assert!(
                    (got - fd).abs() < 1e-6,
                    "J[{out_i}][{var_j}]: {got} vs {fd}"
                );
            }
        }
    }

    #[test]
    fn hessian_matches_closed_form_and_fd() {
        // f = x^2 * y + sin(x*y), scalar, two inputs.
        let ctx = Context::new();
        let x = ctx.var();
        let y = ctx.var();
        let f = &x * &x * &y + (&x * &y).sin();

        let fwd = ctx.lower(&[&f]);
        let n = ctx.n_vars();
        let h_tape = hessian(&fwd, 0, n);

        let p = [1.3, 0.7];
        let out = eval1(&h_tape, &p); // [f, df/dx, df/dy, Hxx, Hxy, Hyx, Hyy]
        let (x, y) = (p[0], p[1]);
        let c = (x * y).cos();
        let s = (x * y).sin();

        // closed form Hessian
        let hxx = 2.0 * y - y * y * s;
        let hxy = 2.0 * x + c - x * y * s;
        let hyy = -x * x * s;
        let expected = [hxx, hxy, hxy, hyy];
        for (k, &e) in expected.iter().enumerate() {
            assert!(
                (out[3 + k] - e).abs() < 1e-10,
                "H entry {k}: {} vs {e}",
                out[3 + k]
            );
        }

        // symmetry: Hxy == Hyx exactly (same recomputed expression up to fold)
        assert!((out[4] - out[5]).abs() < 1e-12);

        // finite differences on the gradient (from the same Hessian tape's
        // df/dx, df/dy outputs) for each second partial.
        let h = 1e-5;
        for a in 0..2 {
            for b in 0..2 {
                let (mut lo, mut hi) = (p, p);
                lo[b] -= h;
                hi[b] += h;
                // gradient component a lives at output index 1 + a
                let fd = (eval1(&h_tape, &hi)[1 + a] - eval1(&h_tape, &lo)[1 + a]) / (2.0 * h);
                let got = out[3 + a * 2 + b];
                assert!((got - fd).abs() < 1e-5, "H[{a}][{b}]: {got} vs {fd}");
            }
        }
    }
}
