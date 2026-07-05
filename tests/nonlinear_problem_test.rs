use autodiff::expression::pow;
use nearly::assert_nearly;
use optimization::{constraint, problem::Problem};

#[test]
fn problem_quartic() {
  let mut problem = Problem::new();

  let mut x = problem.decision_variable();
  x.set_value(20.0);

  problem.minimize(pow(&x, 4.0));

  problem.subject_to(constraint!(&x >= 1.0));

  problem.solve();

  assert_nearly!(x.value() == 1.0, eps = 1e-6);
}