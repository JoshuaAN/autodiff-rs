use std::fmt;

use index_vec::{IndexVec, define_index_type};

use crate::{
    bits::Bits64,
    op::{BinaryOp, UnaryOp},
};

define_index_type! { pub struct Inst = u32; }

pub enum InstructionData {
    /// Function parameter.
    Param(u32),

    /// Constant value.
    Constant(Bits64),

    /// Unary operation on the result of a previous instruction.
    Unary(UnaryOp, Inst),

    /// Binary operation on the results of two previous instructions.
    Binary(BinaryOp, Inst, Inst),
}

pub struct Function {
    /// Program instructions.
    pub insts: IndexVec<Inst, InstructionData>,

    /// Number of parameters.
    pub num_params: u32,

    /// Return values for the function.
    pub returns: Vec<Inst>,
}

impl Function {
    pub fn forward(&self) -> Function {
        use InstructionData as I;
        let n = self.num_params;

        let mut insts: IndexVec<Inst, I> = IndexVec::new();
        // For each original instruction: (primal Inst, tangent Inst).
        // A `None` tangent is a structural zero — never materialized.
        let mut map: IndexVec<Inst, (Inst, Option<Inst>)> =
            IndexVec::with_capacity(self.insts.len());

        fn un(insts: &mut IndexVec<Inst, I>, op: UnaryOp, a: Inst) -> Inst {
            insts.push(I::Unary(op, a))
        }
        fn bin(insts: &mut IndexVec<Inst, I>, op: BinaryOp, a: Inst, b: Inst) -> Inst {
            insts.push(I::Binary(op, a, b))
        }

        for inst in self.insts.iter() {
            let entry = match *inst {
                I::Param(p) => {
                    let primal = insts.push(I::Param(p));
                    let tangent = insts.push(I::Param(n + p));
                    (primal, Some(tangent))
                }
                I::Constant(bits) => (insts.push(I::Constant(bits)), None),
                I::Unary(op, a) => {
                    let (pa, ta) = map[a];
                    let primal = un(&mut insts, op, pa);
                    let tangent = ta.map(|ta| match op {
                        // d sin(a) = cos(a) * da
                        UnaryOp::Sin => {
                            let c = un(&mut insts, UnaryOp::Cos, pa);
                            bin(&mut insts, BinaryOp::Mul, c, ta)
                        }
                        // d cos(a) = -(sin(a) * da)
                        UnaryOp::Cos => {
                            let s = un(&mut insts, UnaryOp::Sin, pa);
                            let m = bin(&mut insts, BinaryOp::Mul, s, ta);
                            un(&mut insts, UnaryOp::Neg, m)
                        }
                        UnaryOp::Neg => un(&mut insts, UnaryOp::Neg, ta),
                    });
                    (primal, tangent)
                }
                I::Binary(op, a, b) => {
                    let (pa, ta) = map[a];
                    let (pb, tb) = map[b];
                    let primal = bin(&mut insts, op, pa, pb);
                    let tangent = match op {
                        BinaryOp::Add => match (ta, tb) {
                            (Some(x), Some(y)) => Some(bin(&mut insts, BinaryOp::Add, x, y)),
                            (Some(x), None) => Some(x),
                            (None, Some(y)) => Some(y),
                            (None, None) => None,
                        },
                        BinaryOp::Sub => match (ta, tb) {
                            (Some(x), Some(y)) => Some(bin(&mut insts, BinaryOp::Sub, x, y)),
                            (Some(x), None) => Some(x),
                            (None, Some(y)) => Some(un(&mut insts, UnaryOp::Neg, y)),
                            (None, None) => None,
                        },
                        // d(a*b) = da*b + a*db
                        BinaryOp::Mul => {
                            let l = ta.map(|t| bin(&mut insts, BinaryOp::Mul, t, pb));
                            let r = tb.map(|t| bin(&mut insts, BinaryOp::Mul, pa, t));
                            match (l, r) {
                                (Some(x), Some(y)) => Some(bin(&mut insts, BinaryOp::Add, x, y)),
                                (x, None) => x,
                                (None, y) => y,
                            }
                        }
                        // y = a/b  =>  dy = (da - y*db) / b   (reuses the primal!)
                        BinaryOp::Div => {
                            let num = match (ta, tb) {
                                (Some(da), Some(db)) => {
                                    let ydb = bin(&mut insts, BinaryOp::Mul, primal, db);
                                    Some(bin(&mut insts, BinaryOp::Sub, da, ydb))
                                }
                                (Some(da), None) => Some(da),
                                (None, Some(db)) => {
                                    let ydb = bin(&mut insts, BinaryOp::Mul, primal, db);
                                    Some(un(&mut insts, UnaryOp::Neg, ydb))
                                }
                                (None, None) => None,
                            };
                            num.map(|nu| bin(&mut insts, BinaryOp::Div, nu, pb))
                        }
                    };
                    (primal, tangent)
                }
            };
            map.push(entry);
        }

        // Returns: primals, then tangents (a lazily-created zero for
        // outputs that don't depend on any input).
        let mut zero: Option<Inst> = None;
        let mut returns = Vec::with_capacity(self.returns.len() * 2);
        for &r in &self.returns {
            returns.push(map[r].0);
        }
        for &r in &self.returns {
            let t = match map[r].1 {
                Some(t) => t,
                None => *zero.get_or_insert_with(|| insts.push(I::Constant(Bits64::from_f64(0.0)))),
            };
            returns.push(t);
        }

        let f = Function {
            insts,
            num_params: 2 * n,
            returns,
        };
        f.verify();
        f
    }

    pub fn backward(&self) -> Function {
      todo!("Implement backward for Function")
    }

    #[cfg(debug_assertions)]
    pub fn verify(&self) {
        for (i, inst) in self.insts.iter_enumerated() {
            match *inst {
                InstructionData::Param(p) => debug_assert!(p < self.num_params),
                InstructionData::Unary(_, a) => debug_assert!(a < i),
                InstructionData::Binary(_, a, b) => debug_assert!(a < i && b < i),
                InstructionData::Constant(_) => {}
            }
        }
        for &r in &self.returns {
            debug_assert!(r.index() < self.insts.len());
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fn({} params):", self.num_params)?;
        for (i, inst) in self.insts.iter_enumerated() {
            write!(f, "  %{} = ", i.index())?;
            match *inst {
                InstructionData::Param(p) => writeln!(f, "param {p}")?,
                InstructionData::Constant(bits) => writeln!(f, "const {}", bits.to_f64())?,
                InstructionData::Unary(op, a) => writeln!(f, "{op} %{}", a.index())?,
                InstructionData::Binary(op, a, b) => {
                    writeln!(f, "%{} {op} %{}", a.index(), b.index())?
                }
            }
        }
        for r in &self.returns {
            writeln!(f, "  ret %{}", r.index())?;
        }
        Ok(())
    }
}
