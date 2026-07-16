use ipopt::Nlp;

#[test]
fn rosenbrock() {
    let sol = Nlp::new(2)
        .objective(|x| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        })
        .gradient(|x, grad| {
            let b = x[1] - x[0] * x[0];
            grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
            grad[1] = 200.0 * b;
        })
        .hessian(vec![(0, 0), (1, 0), (1, 1)], |x, sigma, _lambda, v| {
            v[0] = sigma * (2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0]);
            v[1] = sigma * (-400.0 * x[0]);
            v[2] = sigma * 200.0;
        })
        .num_option("tol", 1e-9)
        .int_option("print_level", 5)
        .str_option("sb", "yes")
        .solve(&[-1.2, 1.0])
        .unwrap();

    assert!(sol.is_success());
    assert!((sol.x[0] - 1.0).abs() < 1e-6);
    assert!((sol.x[1] - 1.0).abs() < 1e-6);
}

/// Rosenbrock constrained to the unit disk:
///
///     min  (1 - x)^2 + 100 (y - x^2)^2
///     s.t. x^2 + y^2 <= 1
///
/// The unconstrained optimum (1, 1) lies outside the disk (norm^2 = 2), so
/// the constraint is active at the solution. Known result (e.g. the classic
/// MATLAB fmincon example): x* ~= (0.7864, 0.6177), f* ~= 0.0457.
///
/// Unlike the unconstrained test, this exercises the code paths that
/// Rosenbrock alone never touches: eval_g, eval_jac_g, the lambda argument
/// of eval_h, and the mult_g output.
#[test]
fn rosenbrock_unit_disk() {
    let sol = Nlp::new(2)
        // one constraint: -inf < x^2 + y^2 <= 1
        .constraint_bounds(vec![-1e19], vec![1.0])
        .objective(|x| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        })
        .gradient(|x, grad| {
            let b = x[1] - x[0] * x[0];
            grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
            grad[1] = 200.0 * b;
        })
        .constraints(|x, g| {
            g[0] = x[0] * x[0] + x[1] * x[1];
        })
        // Jacobian of g: [dg/dx, dg/dy] = [2x, 2y] — dense 1x2 row.
        .jacobian(vec![(0, 0), (0, 1)], |x, v| {
            v[0] = 2.0 * x[0];
            v[1] = 2.0 * x[1];
        })
        // Hessian of the Lagrangian: sigma * H(f) + lambda[0] * H(g),
        // where H(g) = 2 * I. Lower triangle, same pattern as before.
        .hessian(vec![(0, 0), (1, 0), (1, 1)], |x, sigma, lambda, v| {
            v[0] = sigma * (2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0]) + lambda[0] * 2.0;
            v[1] = sigma * (-400.0 * x[0]);
            v[2] = sigma * 200.0 + lambda[0] * 2.0;
        })
        .num_option("tol", 1e-9)
        .int_option("print_level", 5)
        .str_option("sb", "yes")
        // Uncomment while developing: IPOPT finite-difference-checks the
        // gradient, Jacobian, AND the lambda term of the Hessian.
        // .str_option("derivative_test", "second-order")
        .solve(&[0.0, 0.0])
        .unwrap();

    assert!(sol.is_success(), "status was {:?}", sol.status);

    // Known constrained optimum.
    assert!((sol.x[0] - 0.7864).abs() < 1e-3, "x = {:?}", sol.x);
    assert!((sol.x[1] - 0.6177).abs() < 1e-3, "x = {:?}", sol.x);
    assert!((sol.obj - 0.0457).abs() < 1e-3, "obj = {}", sol.obj);

    // The constraint is active: g(x*) sits on the boundary...
    assert!((sol.g[0] - 1.0).abs() < 1e-6, "g = {:?}", sol.g);
    // ...so its multiplier is strictly positive (active upper bound in
    // IPOPT's sign convention, Lagrangian = f + lambda' g).
    assert!(sol.mult_g[0] > 1e-6, "mult_g = {:?}", sol.mult_g);
}
