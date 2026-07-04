//! Port of Sleipnir's autodiff/gradient_test.cpp.
//!
//! Sleipnir's `Gradient(f, x)` differentiates w.r.t. a single variable; here
//! we lower `f`, build the full gradient tape once, and index the component
//! for the variable under test. Sleipnir tests that touch functionality this
//! crate doesn't have are handled as follows:
//!
//! - `tan`, `sinh`/`cosh`/`tanh`, `log10`, `cbrt`, `hypot`, `sign` are
//!   composed from supported ops (`sin`/`cos`, `exp`, `ln`, `powf`, `sqrt`,
//!   `abs`).
//! - `asin`, `acos`, `atan`, `atan2`, `erf` aren't expressible and are
//!   omitted.
//! - Unary plus doesn't exist in Rust; that case reduces to the trivial-copy
//!   check in `trivial_case`.
//! - Differentiation w.r.t. an *intermediate* expression (e.g.
//!   `Gradient(pow(x, y), y)` where `y = 2a`) isn't supported — this crate
//!   only differentiates w.r.t. input variables — so those checks are
//!   omitted.
//! - Where Sleipnir defines a value at a kink (`abs` at 0, `min(x, x)`) and
//!   this crate documents NaN, we assert NaN.

use autodiff::{
    context::{Context, Expr},
    grad::gradient,
    interpreter::eval,
    tape::Tape,
    var::VarId,
};
use index_vec::IndexVec;

fn eval_tape(tape: &Tape, xs: &[f64]) -> Vec<f64> {
    let inputs: IndexVec<VarId, f64> = xs.iter().copied().collect();
    let mut out = Vec::new();
    eval(tape, &inputs, &mut out);
    out
}

/// Lower `f`, differentiate it w.r.t. every variable, and evaluate at `xs`.
/// Returns `(f, [df/dv0, df/dv1, ...])`.
fn value_and_grad(ctx: &Context, f: &Expr, xs: &[f64]) -> (f64, Vec<f64>) {
    let fwd = ctx.lower(&[f]);
    let g = gradient(&fwd, 0, ctx.n_vars());
    let out = eval_tape(&g, xs);
    (out[0], out[1..].to_vec())
}

/// Gradient tape for `f` (outputs `[f, df/dv0, ...]`), for tests that
/// re-evaluate the same tape at several points.
fn grad_tape(ctx: &Context, f: &Expr) -> Tape {
    gradient(&ctx.lower(&[f]), 0, ctx.n_vars())
}

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual} vs {expected}"
    );
}

#[test]
fn trivial_case() {
    let ctx = Context::new();
    let a = ctx.var();
    let _b = ctx.var();
    // `Variable c = a` copies the handle, not the node.
    let c = a.clone();

    let p = [10.0, 20.0];
    let (_, g) = value_and_grad(&ctx, &a, &p);
    assert_close(g[0], 1.0); // da/da
    assert_close(g[1], 0.0); // da/db

    let (_, g) = value_and_grad(&ctx, &c, &p);
    assert_close(g[0], 1.0); // dc/da
    assert_close(g[1], 0.0); // dc/db
}

#[test]
fn unary_minus() {
    let ctx = Context::new();
    let a = ctx.var();
    let c = -&a;

    let p = [10.0];
    let (v, g) = value_and_grad(&ctx, &c, &p);
    assert_close(v, -10.0);
    assert_close(g[0], -1.0);
}

#[test]
fn identical_variables() {
    let ctx = Context::new();
    let a = ctx.var();
    // `Variable x = a` aliases the same node, so c = a² + a.
    let x = a.clone();
    let c = &a * &a + &x;

    let p = [10.0];
    let (v, g) = value_and_grad(&ctx, &c, &p);
    assert_close(v, 10.0 * 10.0 + 10.0);
    // dc/da = 2a + 1
    assert_close(g[0], 2.0 * 10.0 + 1.0);
}

#[test]
fn elementary() {
    let ctx = Context::new();
    let a = ctx.var();
    let b = ctx.var();

    let c = -2.0 * &a;
    let (_, g) = value_and_grad(&ctx, &c, &[1.0, 2.0]);
    assert_close(g[0], -2.0);

    let c = &a / 3.0;
    let (_, g) = value_and_grad(&ctx, &c, &[1.0, 2.0]);
    assert_close(g[0], 1.0 / 3.0);

    let p = [100.0, 200.0];

    let c = &a + &b;
    let (_, g) = value_and_grad(&ctx, &c, &p);
    assert_close(g[0], 1.0);
    assert_close(g[1], 1.0);

    let c = &a - &b;
    let (_, g) = value_and_grad(&ctx, &c, &p);
    assert_close(g[0], 1.0);
    assert_close(g[1], -1.0);

    let c = -&a + &b;
    let (_, g) = value_and_grad(&ctx, &c, &p);
    assert_close(g[0], -1.0);
    assert_close(g[1], 1.0);

    let c = &a + 1.0;
    let (_, g) = value_and_grad(&ctx, &c, &p);
    assert_close(g[0], 1.0);
}

#[test]
fn comparison() {
    // Sleipnir compares Variable values with ==/</etc.; expressions here have
    // no value until evaluated, so compare evaluated outputs instead.
    let ctx = Context::new();
    let a = ctx.var();
    let b = ctx.var();

    let p = [10.0, 200.0];

    let exprs = [
        &a / &a * &a, // == a
        &a - &a,      // == 0
        &a + &a,      // == 2a
        &a - &a + &a, // == a
    ];
    let tape = ctx.lower(&[&exprs[0], &exprs[1], &exprs[2], &exprs[3], &a, &b]);
    let out = eval_tape(&tape, &p);
    let (av, bv) = (out[4], out[5]);

    assert_close(out[0], av); // a/a*a == a
    assert!(out[1] != av && out[1] < av); // a-a < a
    assert!(out[2] > av && av < out[2]); // a+a > a
    assert_close(out[3], av); // a-a+a == a
    assert!(av != bv && av < bv && bv > av);
}

#[test]
fn trigonometry() {
    let ctx = Context::new();
    let x = ctx.var();
    let p = [0.5];
    let xv: f64 = 0.5;

    // sin(x)
    let (v, g) = value_and_grad(&ctx, &x.sin(), &p);
    assert_close(v, xv.sin());
    assert_close(g[0], xv.cos());

    // cos(x)
    let (v, g) = value_and_grad(&ctx, &x.cos(), &p);
    assert_close(v, xv.cos());
    assert_close(g[0], -xv.sin());

    // tan(x) = sin(x)/cos(x)
    let tan = x.sin() / x.cos();
    let (v, g) = value_and_grad(&ctx, &tan, &p);
    assert_close(v, xv.tan());
    assert_close(g[0], 1.0 / (xv.cos() * xv.cos()));

    // asin/acos/atan aren't expressible with this crate's ops — omitted.
}

#[test]
fn hyperbolic() {
    let ctx = Context::new();
    let x = ctx.var();
    let p = [1.0];
    let xv: f64 = 1.0;

    // sinh(x) = (eˣ − e⁻ˣ)/2
    let sinh = (x.exp() - (-&x).exp()) / 2.0;
    let (v, g) = value_and_grad(&ctx, &sinh, &p);
    assert_close(v, xv.sinh());
    assert_close(g[0], xv.cosh());

    // cosh(x) = (eˣ + e⁻ˣ)/2
    let cosh = (x.exp() + (-&x).exp()) / 2.0;
    let (v, g) = value_and_grad(&ctx, &cosh, &p);
    assert_close(v, xv.cosh());
    assert_close(g[0], xv.sinh());

    // tanh(x) = sinh(x)/cosh(x)
    let tanh = &sinh / &cosh;
    let (v, g) = value_and_grad(&ctx, &tanh, &p);
    assert_close(v, xv.tanh());
    assert_close(g[0], 1.0 / (xv.cosh() * xv.cosh()));
}

#[test]
fn exponential() {
    let ctx = Context::new();
    let x = ctx.var();
    let p = [1.0];
    let xv: f64 = 1.0;

    // log(x)
    let (v, g) = value_and_grad(&ctx, &x.ln(), &p);
    assert_close(v, xv.ln());
    assert_close(g[0], 1.0 / xv);

    // log10(x) = ln(x)/ln(10)
    let log10 = x.ln() / 10f64.ln();
    let (v, g) = value_and_grad(&ctx, &log10, &p);
    assert_close(v, xv.log10());
    assert_close(g[0], 1.0 / (10f64.ln() * xv));

    // exp(x)
    let (v, g) = value_and_grad(&ctx, &x.exp(), &p);
    assert_close(v, xv.exp());
    assert_close(g[0], xv.exp());
}

#[test]
fn power() {
    let ctx = Context::new();
    let x = ctx.var();
    let a = ctx.var();
    let y = 2.0 * &a; // y(a)

    let p = [1.0, 2.0];
    let (xv, av): (f64, f64) = (1.0, 2.0);
    let yv = 2.0 * av;

    // sqrt(x)
    let (v, g) = value_and_grad(&ctx, &x.sqrt(), &p);
    assert_close(v, xv.sqrt());
    assert_close(g[0], 0.5 / xv.sqrt());

    // sqrt(a)
    let (v, g) = value_and_grad(&ctx, &a.sqrt(), &p);
    assert_close(v, av.sqrt());
    assert_close(g[1], 0.5 / av.sqrt());

    // cbrt(x) = x^(1/3)
    let third = ctx.constant(1.0 / 3.0);
    let (v, g) = value_and_grad(&ctx, &x.powf(&third), &p);
    assert_close(v, xv.cbrt());
    assert_close(g[0], 1.0 / (3.0 * xv.cbrt() * xv.cbrt()));

    // cbrt(a)
    let (v, g) = value_and_grad(&ctx, &a.powf(&third), &p);
    assert_close(v, av.cbrt());
    assert_close(g[1], 1.0 / (3.0 * av.cbrt() * av.cbrt()));

    // x²
    let (v, g) = value_and_grad(&ctx, &x.powi(2), &p);
    assert_close(v, xv.powi(2));
    assert_close(g[0], 2.0 * xv);

    // 2ˣ
    let two = ctx.constant(2.0);
    let (v, g) = value_and_grad(&ctx, &two.powf(&x), &p);
    assert_close(v, 2f64.powf(xv));
    assert_close(g[0], 2f64.ln() * 2f64.powf(xv));

    // xˣ
    let (v, g) = value_and_grad(&ctx, &x.powf(&x), &p);
    assert_close(v, xv.powf(xv));
    assert_close(g[0], (xv.ln() + 1.0) * xv.powf(xv));

    // y(a) = 2a
    let (v, g) = value_and_grad(&ctx, &y, &p);
    assert_close(v, yv);
    assert_close(g[1], 2.0);

    // xʸ(x): d/dx = y/x · xʸ
    let (v, g) = value_and_grad(&ctx, &x.powf(&y), &p);
    assert_close(v, xv.powf(yv));
    assert_close(g[0], yv / xv * xv.powf(yv));
    // xʸ(a): d/da = xʸ · ln(x) · dy/da = xʸ · ln(x) · 2
    assert_close(g[1], xv.powf(yv) * xv.ln() * 2.0);

    // Sleipnir's Gradient(pow(x, y), y) — w.r.t. the intermediate y — has no
    // equivalent here; only input variables are differentiable.
}

#[test]
fn abs() {
    let ctx = Context::new();
    let x = ctx.var();
    let g = grad_tape(&ctx, &x.abs());

    let out = eval_tape(&g, &[1.0]);
    assert_close(out[0], 1.0);
    assert_close(out[1], 1.0);

    let out = eval_tape(&g, &[-1.0]);
    assert_close(out[0], 1.0);
    assert_close(out[1], -1.0);

    // Sleipnir defines d|x|/dx = 0 at x = 0; this crate documents NaN
    // (sign is computed as x/|x|).
    let out = eval_tape(&g, &[0.0]);
    assert_close(out[0], 0.0);
    assert!(out[1].is_nan());
}

// atan2 isn't expressible with this crate's ops — test omitted.

#[test]
fn hypot() {
    let ctx = Context::new();
    let x = ctx.var();
    let y = ctx.var();

    let hypot2 = |a: &Expr, b: &Expr| (a * a + b * b).sqrt();

    // hypot(x, 2)
    let p = [1.8, 1.5];
    let xv: f64 = 1.8;
    let c2 = ctx.constant(2.0);
    let (v, g) = value_and_grad(&ctx, &hypot2(&x, &c2), &p);
    assert_close(v, xv.hypot(2.0));
    assert_close(g[0], xv / xv.hypot(2.0));

    // hypot(2, y)
    let yv: f64 = 1.5;
    let (v, g) = value_and_grad(&ctx, &hypot2(&c2, &y), &p);
    assert_close(v, 2f64.hypot(yv));
    assert_close(g[1], yv / 2f64.hypot(yv));

    // hypot(x, y)
    let p = [1.3, 2.3];
    let (xv, yv): (f64, f64) = (1.3, 2.3);
    let (v, g) = value_and_grad(&ctx, &hypot2(&x, &y), &p);
    assert_close(v, xv.hypot(yv));
    assert_close(g[0], xv / xv.hypot(yv));
    assert_close(g[1], yv / xv.hypot(yv));

    // hypot(2x, 3y)
    let h = hypot2(&(2.0 * &x), &(3.0 * &y));
    let (v, g) = value_and_grad(&ctx, &h, &p);
    assert_close(v, (2.0 * xv).hypot(3.0 * yv));
    assert_close(g[0], 4.0 * xv / (2.0 * xv).hypot(3.0 * yv));
    assert_close(g[1], 9.0 * yv / (2.0 * xv).hypot(3.0 * yv));

    // hypot(x, y, z)
    let z = ctx.var();
    let p = [1.3, 2.3, 3.3];
    let zv: f64 = 3.3;
    let h3 = (&x * &x + &y * &y + &z * &z).sqrt();
    let norm = (xv * xv + yv * yv + zv * zv).sqrt();
    let (v, g) = value_and_grad(&ctx, &h3, &p);
    assert_close(v, norm);
    assert_close(g[0], xv / norm);
    assert_close(g[1], yv / norm);
    assert_close(g[2], zv / norm);
}

#[test]
fn max() {
    let ctx = Context::new();
    let x = ctx.var();
    let x2 = &x * &x;
    let x3 = &x * &x * &x;

    let p = [2.0];
    let xv: f64 = 2.0;
    let dx3 = 3.0 * xv * xv; // d(x³)/dx

    // lhs < rhs
    let (v, g) = value_and_grad(&ctx, &x2.max(&x3), &p);
    assert_close(v, xv.powi(3));
    assert_close(g[0], dx3);

    // lhs > rhs
    let (v, g) = value_and_grad(&ctx, &x3.max(&x2), &p);
    assert_close(v, xv.powi(3));
    assert_close(g[0], dx3);

    // lhs == rhs: Sleipnir defines the gradient as 1; this crate documents
    // NaN at ties (sign(a−b) is computed as (a−b)/|a−b|).
    let (v, g) = value_and_grad(&ctx, &x.max(&x), &p);
    assert_close(v, xv);
    assert!(g[0].is_nan());
}

#[test]
fn min() {
    let ctx = Context::new();
    let x = ctx.var();
    let x2 = &x * &x;
    let x3 = &x * &x * &x;

    let p = [2.0];
    let xv: f64 = 2.0;
    let dx2 = 2.0 * xv; // d(x²)/dx

    // lhs < rhs
    let (v, g) = value_and_grad(&ctx, &x2.min(&x3), &p);
    assert_close(v, xv.powi(2));
    assert_close(g[0], dx2);

    // lhs > rhs
    let (v, g) = value_and_grad(&ctx, &x3.min(&x2), &p);
    assert_close(v, xv.powi(2));
    assert_close(g[0], dx2);

    // lhs == rhs: NaN at ties, unlike Sleipnir's 1 (see `max`).
    let (v, g) = value_and_grad(&ctx, &x.min(&x), &p);
    assert_close(v, xv);
    assert!(g[0].is_nan());
}

#[test]
fn miscellaneous() {
    let ctx = Context::new();
    let x = ctx.var();

    // dx/dx
    let (v, g) = value_and_grad(&ctx, &x, &[3.0]);
    assert_close(v, 3.0);
    assert_close(g[0], 1.0);

    // erf isn't expressible with this crate's ops — omitted.
}

#[test]
fn variable_reuse() {
    let ctx = Context::new();
    let a = ctx.var();
    let b = ctx.var();
    let x = &a * &b;

    // Build the gradient tape once, re-evaluate as b changes.
    let g = grad_tape(&ctx, &x);

    let out = eval_tape(&g, &[10.0, 20.0]);
    assert_close(out[1], 20.0); // dx/da = b

    let out = eval_tape(&g, &[10.0, 10.0]);
    assert_close(out[1], 10.0);
}

#[test]
fn sign() {
    // sign(x) = x/|x|
    let ctx = Context::new();
    let x = ctx.var();
    let s = &x / x.abs();
    let g = grad_tape(&ctx, &s);

    // sign(1)
    let out = eval_tape(&g, &[1.0]);
    assert_close(out[0], 1.0);
    assert_close(out[1], 0.0);

    // sign(-1)
    let out = eval_tape(&g, &[-1.0]);
    assert_close(out[0], -1.0);
    assert_close(out[1], 0.0);

    // sign(0): Sleipnir defines value 0 and gradient 0; x/|x| is 0/0 = NaN.
    let out = eval_tape(&g, &[0.0]);
    assert!(out[0].is_nan());
    assert!(out[1].is_nan());
}

#[test]
fn non_scalar() {
    let ctx = Context::new();
    let x0 = ctx.var();
    let x1 = ctx.var();
    let x2 = ctx.var();

    // y = x₀ + 3x₁ − 5x₂
    //
    // dy/dx = [1  3  −5]
    let y = &x0 + 3.0 * &x1 - 5.0 * &x2;

    let (_, g) = value_and_grad(&ctx, &y, &[1.0, 2.0, 3.0]);
    assert_eq!(g.len(), 3);
    assert_close(g[0], 1.0);
    assert_close(g[1], 3.0);
    assert_close(g[2], -5.0);
}
