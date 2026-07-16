use std::fmt::{self, Display};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Sin,
    Cos,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Sin => write!(f, "sin"),
            UnaryOp::Cos => write!(f, "cos"),
        };
        Ok(())
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
        };
        Ok(())
    }
}
