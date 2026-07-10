use ir::function::Function;

pub trait FunctionEvaluator {
  fn from_ir(func: &Function) -> Self;

  fn evaluate(params: &[f64]) -> &[f64];
}