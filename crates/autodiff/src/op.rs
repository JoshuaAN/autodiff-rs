#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Abs,
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
            UnaryOp::Abs => a.abs(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Pow,
}

impl BinaryOp {
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Mod => a % b,
            BinaryOp::Min => a.min(b),
            BinaryOp::Max => a.max(b),
            BinaryOp::Pow => a.powf(b),
        }
    }
}
