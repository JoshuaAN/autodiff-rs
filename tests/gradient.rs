use autodiff::{expression::Expression, grad::gradient, tape::Tape, var::VarId};

#[test]
fn eval_and_gradient() {
    let x_id = VarId::from(0);
    let y_id = VarId::from(1);
    let x = Expression::variable(x_id);
    let y = Expression::variable(y_id);

    // f(x, y) = x*y + x^2
    let f = &x * &y + &x * &x;
    let tape = Tape::from_exprs(&[f], &[x, y]);

    let (xv, yv) = (1.5, 3.2);

    // Value: f(1.5, 3.2) = 4.8 + 2.25 = 7.05
    let eval = tape.eval(&[xv, yv]);
    assert_eq!(eval.len(), 1);
    assert!((eval[0] - (xv * yv + xv * xv)).abs() < 1e-12);

    // Gradient: df/dx = y + 2x = 6.2, df/dy = x = 1.5
    let grad_tape = gradient(&tape, tape.outputs[0]);
    let grads = grad_tape.eval(&[xv, yv]);
    assert_eq!(grads.len(), 2);
    assert!((grads[0] - (yv + 2.0 * xv)).abs() < 1e-12);
    assert!((grads[1] - xv).abs() < 1e-12);
}