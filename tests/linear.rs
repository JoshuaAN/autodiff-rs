use nearly::assert_nearly;
use solver::{NlpSolver, constraint};

#[test]
fn maximize() {
    let solver = NlpSolver::new();

    let x = solver.decision_variable();
    let y = solver.decision_variable();
    solver.set_initial(x, 1.0);
    solver.set_initial(y, 1.0);

    solver.maximize(50.0 * x + 40.0 * y);

    solver.subject_to(constraint!(x + 1.5 * y <= 750.0));
    solver.subject_to(constraint!(2.0 * x + 3.0 * y <= 1500.0));
    solver.subject_to(constraint!(2.0 * x + y <= 1000.0));
    solver.subject_to(constraint!(x >= 0.0));
    solver.subject_to(constraint!(y >= 0.0));

    let x_id = x.id();
    let y_id = y.id();

    let sol = solver.solve().expect("solve failed");

    assert!(sol.is_success());
    assert!((sol.value(x_id) - 375.0) < 1e-6);
    assert!((sol.value(y_id) - 250.0) < 1e-6);
}
