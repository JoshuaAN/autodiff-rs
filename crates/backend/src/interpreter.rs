use ir::function::Function;

use crate::backend::FunctionEvaluator;

/// Computes functions in an interpreter. This is slower than JIT compilation, but is
/// cross-platform.
struct Interpreter {

}

impl FunctionEvaluator for Interpreter {
  fn from_ir(func: &Function) -> Self {
    todo!("Implement from_ir for Interpreter")
  }

  fn evaluate(params: &[f64]) -> &[f64] {
    todo!("Implement evaluate for Interpreter")
  }
}