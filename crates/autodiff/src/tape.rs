use std::cell::RefCell;

use index_vec::{IndexVec};

use crate::{
    bits::Bits64, function::Function, node::{Node, NodeId}, op::{BinaryOp, UnaryOp},
};

#[derive(Default)]
struct TapeInner {
    nodes: IndexVec<NodeId, Node>,
    num_params: u32,
}

impl TapeInner {
    fn param(&mut self) -> NodeId {
        let data = Node::Param(self.num_params);
        self.num_params += 1;
        self.push(data)
    }

    fn constant(&mut self, x: f64) -> NodeId {
        self.push(Node::Constant(Bits64::from_f64(x)))
    }

    fn unary(&mut self, op: UnaryOp, v: NodeId) -> NodeId {
        self.push(Node::Unary(op, v))
    }

    fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.push(Node::Binary(op, lhs, rhs))
    }

    fn push(&mut self, data: Node) -> NodeId {
        self.nodes.push(data)
    }
}

pub struct Tape {
    inner: RefCell<TapeInner>,
}

#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    value: NodeId,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            inner: RefCell::new(TapeInner::default()),
        }
    }

    pub fn param(&self) -> Var<'_> {
        let value = self.inner.borrow_mut().param();
        Var { tape: self, value }
    }

    pub fn constant(&self, x: f64) -> Var<'_> {
        let value = self.inner.borrow_mut().constant(x);
        Var { tape: self, value }
    }

    #[inline]
    fn check_owns(&self, v: Var<'_>, what: &str) {
        assert!(
            std::ptr::eq(v.tape, self),
            "{what}: Var belongs to a different tape"
        );
    }

    pub fn freeze(&self, inputs: &[Var<'_>], outputs: &[Var<'_>]) -> Function {
        todo!("implement freeze")
    }
}

impl<'t> Var<'t> {
    fn push_unary(self, op: UnaryOp) -> Var<'t> {
        let value = self.tape.inner.borrow_mut().unary(op, self.value);
        Var {
            tape: self.tape,
            value,
        }
    }

    fn push_binary(self, op: BinaryOp, rhs: Var<'t>) -> Var<'t> {
        self.tape.check_owns(rhs, "binary op");
        let value = self
            .tape
            .inner
            .borrow_mut()
            .binary(op, self.value, rhs.value);
        Var {
            tape: self.tape,
            value,
        }
    }

    pub fn id(&self) -> NodeId {
        self.value
    }
}

macro_rules! impl_unary_fns {
    ($($method:ident => $op:expr),* $(,)?) => {
        impl<'t> Var<'t> {
            $(
                pub fn $method(self) -> Var<'t> {
                    self.push_unary($op)
                }
            )*
        }
    };
}

impl_unary_fns! {
    sin => UnaryOp::Sin,
    cos => UnaryOp::Cos,
    neg => UnaryOp::Neg,
}

impl<'t> std::ops::Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Var<'t> {
        Var::neg(self)
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        // Var op Var
        impl<'t> std::ops::$trait for Var<'t> {
            type Output = Var<'t>;
            fn $method(self, rhs: Var<'t>) -> Var<'t> {
                self.push_binary($op, rhs)
            }
        }

        // Var op f64
        impl<'t> std::ops::$trait<f64> for Var<'t> {
            type Output = Var<'t>;
            fn $method(self, rhs: f64) -> Var<'t> {
                let rhs = self.tape.constant(rhs);
                self.push_binary($op, rhs)
            }
        }

        // f64 op Var
        impl<'t> std::ops::$trait<Var<'t>> for f64 {
            type Output = Var<'t>;
            fn $method(self, rhs: Var<'t>) -> Var<'t> {
                let lhs = rhs.tape.constant(self);
                lhs.push_binary($op, rhs)
            }
        }
    };
}

impl_binary_op!(Add, add, BinaryOp::Add);
impl_binary_op!(Sub, sub, BinaryOp::Sub);
impl_binary_op!(Mul, mul, BinaryOp::Mul);
impl_binary_op!(Div, div, BinaryOp::Div);
