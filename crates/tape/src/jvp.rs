use index_vec::IndexVec;

use crate::{
    op::{BinaryOp, UnaryOp},
    tape::{Inst, InstructionData, Tape, Value, ValueData},
};

pub fn jvp(tape: &Tape) -> Tape {
    let n = tape.num_params();
    debug_assert!(
        2 * n <= u16::MAX as usize + 1,
        "too many params for u16 indices"
    );

    let mut out = Tape::default();
    for _ in 0..(2 * n) {
        out.param();
    }

    // Maps old instruction --> old value it defines
    let mut result_of: IndexVec<Inst, Option<Value>> =
        IndexVec::from_vec(vec![None; tape.insts.len()]);
    for (v, &data) in tape.values.iter_enumerated() {
        if let ValueData::Result(i) = data {
            result_of[i] = Some(v);
        }
    }

    // Maps old primal value ---> (new primal, new tangent)
    let mut env: IndexVec<Value, Option<(Value, Option<Value>)>> = IndexVec::new();
    env.resize(tape.values.len(), None);
    let new_params: Vec<Value> = out.args().to_vec();
    let (primals, tangents) = new_params.split_at(n);
    for ((&old, &p), &t) in tape.args().iter().zip(primals).zip(tangents) {
        env[old] = Some((p, Some(t)));
    }

    for &inst in tape.layout() {
        let (p, t) = match tape.inst_data(inst) {
            // d(c) = 0 — symbolic, no tangent instruction emitted.
            InstructionData::Constant(c) => (out.constant(c.to_f64()), None),

            InstructionData::Unary(op, a) => {
                let (pa, ta) = env[a].expect("operand defined before use");
                match op {
                    // d(-a) = -da
                    UnaryOp::Neg => {
                        let p = out.unary(UnaryOp::Neg, pa);
                        let t = ta.map(|ta| out.unary(UnaryOp::Neg, ta));
                        (p, t)
                    }
                    // d(sin a) = cos(a) . da
                    UnaryOp::Sin => {
                        let p = out.unary(UnaryOp::Sin, pa);
                        let t = ta.map(|ta| {
                            let c = out.unary(UnaryOp::Cos, pa);
                            out.binary(BinaryOp::Mul, c, ta)
                        });
                        (p, t)
                    }
                    // d(cos a) = -(sin(a) . da)
                    UnaryOp::Cos => {
                        let p = out.unary(UnaryOp::Cos, pa);
                        let t = ta.map(|ta| {
                            let s = out.unary(UnaryOp::Sin, pa);
                            let m = out.binary(BinaryOp::Mul, s, ta);
                            out.unary(UnaryOp::Neg, m)
                        });
                        (p, t)
                    }
                }
            }

            InstructionData::Binary(op, a, b) => {
                let (pa, ta) = env[a].expect("operand defined before use");
                let (pb, tb) = env[b].expect("operand defined before use");
                match op {
                    // d(a + b) = da + db
                    BinaryOp::Add => {
                        let p = out.binary(BinaryOp::Add, pa, pb);
                        let t = match (ta, tb) {
                            (None, None) => None,
                            (Some(ta), None) => Some(ta),
                            (None, Some(tb)) => Some(tb),
                            (Some(ta), Some(tb)) => Some(out.binary(BinaryOp::Add, ta, tb)),
                        };
                        (p, t)
                    }
                    // d(a - b) = da - db
                    BinaryOp::Sub => {
                        let p = out.binary(BinaryOp::Sub, pa, pb);
                        let t = match (ta, tb) {
                            (None, None) => None,
                            (Some(ta), None) => Some(ta),
                            (None, Some(tb)) => Some(out.unary(UnaryOp::Neg, tb)),
                            (Some(ta), Some(tb)) => Some(out.binary(BinaryOp::Sub, ta, tb)),
                        };
                        (p, t)
                    }
                    // d(a * b) = b . da + a . db
                    BinaryOp::Mul => {
                        let p = out.binary(BinaryOp::Mul, pa, pb);
                        let t = match (ta, tb) {
                            (None, None) => None,
                            (Some(ta), None) => Some(out.binary(BinaryOp::Mul, pb, ta)),
                            (None, Some(tb)) => Some(out.binary(BinaryOp::Mul, pa, tb)),
                            (Some(ta), Some(tb)) => {
                                let l = out.binary(BinaryOp::Mul, pb, ta);
                                let r = out.binary(BinaryOp::Mul, pa, tb);
                                Some(out.binary(BinaryOp::Add, l, r))
                            }
                        };
                        (p, t)
                    }
                    // d(a / b) = (da - (a/b) . db) / b, reusing the primal
                    // quotient: one division, and better numerics than the
                    // textbook (da.b - a.db)/b^2.
                    BinaryOp::Div => {
                        let p = out.binary(BinaryOp::Div, pa, pb);
                        let t = match (ta, tb) {
                            (None, None) => None,
                            (Some(ta), None) => Some(out.binary(BinaryOp::Div, ta, pb)),
                            (None, Some(tb)) => {
                                let u = out.binary(BinaryOp::Mul, p, tb);
                                let nu = out.unary(UnaryOp::Neg, u);
                                Some(out.binary(BinaryOp::Div, nu, pb))
                            }
                            (Some(ta), Some(tb)) => {
                                let u = out.binary(BinaryOp::Mul, p, tb);
                                let d = out.binary(BinaryOp::Sub, ta, u);
                                Some(out.binary(BinaryOp::Div, d, pb))
                            }
                        };
                        (p, t)
                    }
                }
            }
        };
        let old = result_of[inst].expect("instruction defines a value");
        env[old] = Some((p, t));
    }

    // Returns primal outputs first, then tangent outputs.
    for &r in &tape.returns {
        let (p, _) = env[r].expect("return value defined");
        out.returns.push(p);
    }
    let mut zero: Option<Value> = None;
    for &r in &tape.returns {
        let (_, t) = env[r].expect("return value defined");
        let t = t.unwrap_or_else(|| *zero.get_or_insert_with(|| out.constant(0.0)));
        out.returns.push(t);
    }

    out
}
