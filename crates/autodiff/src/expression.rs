use std::{
    cell::Cell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::var::VarId;

use super::op::{BinaryOp, UnaryOp};

#[derive(Clone, Debug)]
pub enum Node {
    Constant(f64),
    Variable(VarId, Cell<f64>),
    Unary(UnaryOp, Expression),
    Binary(BinaryOp, Expression, Expression),
}

#[derive(Clone, Debug)]
pub struct Expression {
    node: Rc<Node>,
}

impl Expression {
    pub fn constant(v: f64) -> Expression {
        Expression {
            node: Rc::new(Node::Constant(v)),
        }
    }

    pub fn variable(id: VarId) -> Expression {
        Expression {
            node: Rc::new(Node::Variable(id, Cell::new(0.0))),
        }
    }

    pub fn unary(op: UnaryOp, a: Expression) -> Expression {
        Expression {
            node: Rc::new(Node::Unary(op, a)),
        }
    }

    pub fn binary(op: BinaryOp, a: Expression, b: Expression) -> Expression {
        Expression {
            node: Rc::new(Node::Binary(op, a, b)),
        }
    }

    pub fn node(&self) -> &Node {
        &self.node
    }

    pub fn as_ptr(&self) -> *const Node {
        Rc::as_ptr(&self.node)
    }

    pub fn value(&self) -> f64 {
        match self.node() {
            Node::Variable(_, v) => v.get(),
            n => panic!("value() called on non-variable expression: {n:?}"),
        }
    }

    pub fn set_value(&self, value: f64) {
        match self.node() {
            Node::Variable(_, v) => v.set(value),
            n => panic!("set_value() called on non-variable expression: {n:?}"),
        }
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

impl Drop for Expression {
    fn drop(&mut self) {
        // Fast paths: shared node (someone else keeps it alive; no recursion
        // happens from our drop) or leaf (no children to recurse into).
        if Rc::strong_count(&self.node) > 1
            || matches!(&*self.node, Node::Constant(_) | Node::Variable(..))
        {
            return;
        }

        // Sole owner of an interior node: dismantle iteratively. Children get
        // their nodes swapped for a shared leaf before they drop, so their
        // own Drop takes the fast path.
        thread_local! {
            static LEAF: Rc<Node> = Rc::new(Node::Constant(0.0));
        }
        let leaf = LEAF.with(Rc::clone);

        let mut stack = vec![std::mem::replace(&mut self.node, Rc::clone(&leaf))];
        while let Some(rc) = stack.pop() {
            // If another owner exists, dropping our Rc just decrements — fine.
            if let Ok(node) = Rc::try_unwrap(rc) {
                match node {
                    Node::Unary(_, mut a) => {
                        stack.push(std::mem::replace(&mut a.node, Rc::clone(&leaf)));
                    }
                    Node::Binary(_, mut a, mut b) => {
                        stack.push(std::mem::replace(&mut a.node, Rc::clone(&leaf)));
                        stack.push(std::mem::replace(&mut b.node, Rc::clone(&leaf)));
                    }
                    Node::Constant(_) | Node::Variable(..) => {}
                }
            }
        }
    }
}
