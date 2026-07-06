#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
}

impl UnaryOp {
    #[inline]
    pub fn apply(self, a: f64) -> f64 {
        match self {
            UnaryOp::Neg => -a,
            UnaryOp::Sqrt => a.sqrt(),
            UnaryOp::Exp => a.exp(),
            UnaryOp::Ln => a.ln(),
            UnaryOp::Sin => a.sin(),
            UnaryOp::Cos => a.cos(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }
}
