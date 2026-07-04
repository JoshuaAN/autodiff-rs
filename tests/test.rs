#[cfg(test)]
mod tests {
    use index_vec::IndexVec;

use super::*;

    fn eval1(tape: &Tape, xs: &[f64]) -> Vec<f64> {
        let inputs: IndexVec<VarId, f64> = xs.iter().copied().collect();
        let mut out = Vec::new();
        crate::interp::eval(tape, &inputs, &mut out);
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
        g.validate();

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
            assert!((out[1 + i] - fd).abs() < 1e-6, "var {i}: {} vs {}", out[1 + i], fd);
        }
    }
}
