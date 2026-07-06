use crate::expression::Expression;

pub struct ExpressionMatrix {
  storage: Vec<Expression>,
  rows: u32,
  cols: u32,
}