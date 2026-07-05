use std::{ops::{Add, Div, Mul, Sub}, rc::Rc};

use crate::var::VarId;

use super::op::{BinaryOp, UnaryOp};

#[derive(Clone, Debug)]
pub enum Node {
    Constant(f64),
    Variable(VarId),
    Unary(UnaryOp, Expression),
    Binary(BinaryOp, Expression, Expression),
}

#[derive(Clone, Debug)]
pub struct Expression {
    node: Rc<Node>,
    value: f64,
}

impl Expression {
    pub fn constant(v: f64) -> Expression {
        Expression { node: Rc::new(Node::Constant(v)), value: 0.0 }
    }
    
    pub fn variable(id: VarId) -> Expression {
        Expression { node: Rc::new(Node::Variable(id)), value: 0.0 }
    }

    pub fn unary(op: UnaryOp, a: Expression) -> Expression {
        Expression { node: Rc::new(Node::Unary(op, a)), value: 0.0 }
    }

    pub fn binary(op: BinaryOp, a: Expression, b: Expression) -> Expression {
        Expression { node: Rc::new(Node::Binary(op, a, b)), value: 0.0 }
    }

    pub fn node(&self) -> &Node {
        &self.node
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}

impl From<f64> for Expression {
    fn from(v: f64) -> Expression {
        Expression::constant(v)
    }
}
 
impl From<&Expression> for Expression {
    fn from(e: &Expression) -> Expression {
        e.clone()
    }
}

macro_rules! impl_binary_operator {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<R: Into<Expression>> $trait<R> for Expression {
            type Output = Expression;
            fn $method(self, rhs: R) -> Expression {
                Expression::binary($op, self, rhs.into())
            }
        }
        impl<R: Into<Expression>> $trait<R> for &Expression {
            type Output = Expression;
            fn $method(self, rhs: R) -> Expression {
                Expression::binary($op, self.clone(), rhs.into())
            }
        }
        impl $trait<Expression> for f64 {
            type Output = Expression;
            fn $method(self, rhs: Expression) -> Expression {
                Expression::binary($op, Expression::constant(self), rhs)
            }
        }
        impl $trait<&Expression> for f64 {
            type Output = Expression;
            fn $method(self, rhs: &Expression) -> Expression {
                Expression::binary($op, Expression::constant(self), rhs.clone())
            }
        }
    };
}
 
impl_binary_operator!(Add, add, BinaryOp::Add);
impl_binary_operator!(Sub, sub, BinaryOp::Sub);
impl_binary_operator!(Mul, mul, BinaryOp::Mul);
impl_binary_operator!(Div, div, BinaryOp::Div);

impl std::ops::Neg for &Expression {
    type Output = Expression;
    fn neg(self) -> Expression {
        Expression::unary(UnaryOp::Neg, self.clone())
    }
}
impl std::ops::Neg for Expression {
    type Output = Expression;
    fn neg(self) -> Expression {
        Expression::unary(UnaryOp::Neg, self)
    }
}

pub fn pow(base: impl Into<Expression>, exponent: impl Into<Expression>) -> Expression {
    Expression::binary(BinaryOp::Pow, base.into(), exponent.into())
}

pub fn min(base: impl Into<Expression>, exponent: impl Into<Expression>) -> Expression {
    Expression::binary(BinaryOp::Min, base.into(), exponent.into())
}

pub fn max(base: impl Into<Expression>, exponent: impl Into<Expression>) -> Expression {
    Expression::binary(BinaryOp::Max, base.into(), exponent.into())
}