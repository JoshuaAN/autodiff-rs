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
