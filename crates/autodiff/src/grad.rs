use index_vec::IndexVec;

use crate::{
    op::{BinaryOp, UnaryOp},
    tape::{Instr, Slot, Tape},
    var::VarId,
};

/// Builds a tape incrementally, with just enough peephole folding to keep
/// the gradient output from drowning in `x * 1.0` junk.
pub(crate) struct TapeBuilder {
    pub(crate) insts: IndexVec<Slot, Instr>,
}

impl TapeBuilder {
    pub(crate) fn constant(&mut self, v: f64) -> Slot {
        self.insts.push(Instr::Const(v))
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, a: Slot) -> Slot {
        if let Instr::Const(x) = self.insts[a] {
            return self.constant(op.apply(x));
        }
        self.insts.push(Instr::Unary(op, a))
    }

    pub(crate) fn binary(&mut self, op: BinaryOp, a: Slot, b: Slot) -> Slot {
        use Instr::Const;
        match (op, self.insts[a], self.insts[b]) {
            (_, Const(x), Const(y)) => return self.constant(op.apply(x, y)),
            (BinaryOp::Mul, Const(x), _) if x == 1.0 => return b,
            (BinaryOp::Mul, _, Const(y)) if y == 1.0 => return a,
            (BinaryOp::Add, Const(x), _) if x == 0.0 => return b,
            (BinaryOp::Add, _, Const(y)) if y == 0.0 => return a,
            _ => {}
        }
        self.insts.push(Instr::Binary(op, a, b))
    }
}

fn accumulate(
    tb: &mut TapeBuilder,
    adj: &mut IndexVec<Slot, Option<Slot>>,
    target: Slot,
    contrib: Slot,
) {
    adj[target] = Some(match adj[target] {
        None => contrib,
        Some(prev) => tb.binary(BinaryOp::Add, prev, contrib),
    });
}

/// One reverse-mode sweep over `forward`, seeding `d(seed)/d(seed) = 1`.
///
/// New derivative instructions are appended to `tb` (whose prefix must be
/// `forward`, so every forward Slot stays valid). Returns `adj`, where
/// `adj[s]` is the slot holding `d(seed)/d(s)`, or `None` if that partial is
/// structurally zero.
///
/// The sweep reads instructions and primals only from `forward`, never from
/// the derivative slots it appends — so it can be called repeatedly against
/// the same `forward` (once per output/row) to build a Jacobian or Hessian.
pub(crate) fn reverse_sweep(
    tb: &mut TapeBuilder,
    forward: &IndexVec<Slot, Instr>,
    seed: Slot,
) -> IndexVec<Slot, Option<Slot>> {
    // adj[s] = slot currently holding d(seed)/d(s), or None if zero so far.
    let mut adj: IndexVec<Slot, Option<Slot>> = index_vec::index_vec![None; forward.len()];

    let one = tb.constant(1.0);
    adj[seed] = Some(one);

    for (slot, &inst) in forward.iter_enumerated().rev() {
        let Some(d) = adj[slot] else { continue };

        match inst {
            Instr::Const(_) | Instr::Input(_) => {}

            Instr::Unary(op, a) => {
                let contrib = match op {
                    // d(-a) = -d
                    UnaryOp::Neg => tb.unary(UnaryOp::Neg, d),
                    // y = sqrt(a): dy/da = 1/(2y)  — reuses primal y
                    UnaryOp::Sqrt => {
                        let two = tb.constant(2.0);
                        let t = tb.binary(BinaryOp::Mul, two, slot);
                        tb.binary(BinaryOp::Div, d, t)
                    }
                    // y = exp(a): dy/da = y
                    UnaryOp::Exp => tb.binary(BinaryOp::Mul, d, slot),
                    // d(ln a) = d / a
                    UnaryOp::Ln => tb.binary(BinaryOp::Div, d, a),
                    UnaryOp::Sin => {
                        let c = tb.unary(UnaryOp::Cos, a);
                        tb.binary(BinaryOp::Mul, d, c)
                    }
                    UnaryOp::Cos => {
                        let s = tb.unary(UnaryOp::Sin, a);
                        let t = tb.binary(BinaryOp::Mul, d, s);
                        tb.unary(UnaryOp::Neg, t)
                    }
                    // d|a| = sign(a) = a/|a|  — reuses primal; NaN at a == 0
                    UnaryOp::Abs => {
                        let s = tb.binary(BinaryOp::Div, a, slot);
                        tb.binary(BinaryOp::Mul, d, s)
                    }
                };
                accumulate(tb, &mut adj, a, contrib);
            }

            Instr::Binary(op, lhs, rhs) => match op {
                BinaryOp::Add => {
                    accumulate(tb, &mut adj, lhs, d);
                    accumulate(tb, &mut adj, rhs, d);
                }
                BinaryOp::Sub => {
                    accumulate(tb, &mut adj, lhs, d);
                    let nd = tb.unary(UnaryOp::Neg, d);
                    accumulate(tb, &mut adj, rhs, nd);
                }
                BinaryOp::Mul => {
                    let ca = tb.binary(BinaryOp::Mul, d, rhs);
                    accumulate(tb, &mut adj, lhs, ca);
                    let cb = tb.binary(BinaryOp::Mul, d, lhs);
                    accumulate(tb, &mut adj, rhs, cb);
                }
                BinaryOp::Div => {
                    // y = a/b: dy/da = 1/b, dy/db = -y/b  — reuses primal y
                    let ca = tb.binary(BinaryOp::Div, d, rhs);
                    accumulate(tb, &mut adj, lhs, ca);
                    let t = tb.binary(BinaryOp::Mul, d, slot);
                    let t = tb.binary(BinaryOp::Div, t, rhs);
                    let cb = tb.unary(UnaryOp::Neg, t);
                    accumulate(tb, &mut adj, rhs, cb);
                }
                BinaryOp::Mod => {
                    // y = a - trunc(a/b)*b: dy/da = 1, dy/db = -trunc(a/b).
                    // Recover trunc(a/b) as (a - y)/b from the primal.
                    accumulate(tb, &mut adj, lhs, d);
                    let t = tb.binary(BinaryOp::Sub, lhs, slot);
                    let q = tb.binary(BinaryOp::Div, t, rhs);
                    let t = tb.binary(BinaryOp::Mul, d, q);
                    let cb = tb.unary(UnaryOp::Neg, t);
                    accumulate(tb, &mut adj, rhs, cb);
                }
                BinaryOp::Min | BinaryOp::Max => {
                    // Route d to whichever operand was selected, via
                    // s = sign(a-b) = (a-b)/|a-b|:
                    //   min: wa = (1-s)/2, wb = (1+s)/2   (max: swapped)
                    // NaN at a == b — see note below.
                    let diff = tb.binary(BinaryOp::Sub, lhs, rhs);
                    let ad = tb.unary(UnaryOp::Abs, diff);
                    let s = tb.binary(BinaryOp::Div, diff, ad);
                    let one = tb.constant(1.0);
                    let half = tb.constant(0.5);
                    let (wa, wb) = if op == BinaryOp::Min {
                        (
                            tb.binary(BinaryOp::Sub, one, s),
                            tb.binary(BinaryOp::Add, one, s),
                        )
                    } else {
                        (
                            tb.binary(BinaryOp::Add, one, s),
                            tb.binary(BinaryOp::Sub, one, s),
                        )
                    };
                    let wa = tb.binary(BinaryOp::Mul, half, wa);
                    let wb = tb.binary(BinaryOp::Mul, half, wb);
                    let ca = tb.binary(BinaryOp::Mul, d, wa);
                    accumulate(tb, &mut adj, lhs, ca);
                    let cb = tb.binary(BinaryOp::Mul, d, wb);
                    accumulate(tb, &mut adj, rhs, cb);
                }
                BinaryOp::Pow => {
                    // y = a^b: dy/da = b*y/a, dy/db = y*ln(a)  — reuses y
                    let t = tb.binary(BinaryOp::Mul, rhs, slot);
                    let t = tb.binary(BinaryOp::Div, t, lhs);
                    let ca = tb.binary(BinaryOp::Mul, d, t);
                    accumulate(tb, &mut adj, lhs, ca);
                    let l = tb.unary(UnaryOp::Ln, lhs);
                    let t = tb.binary(BinaryOp::Mul, slot, l);
                    let cb = tb.binary(BinaryOp::Mul, d, t);
                    accumulate(tb, &mut adj, rhs, cb);
                }
            },
        }
    }

    adj
}

/// Slot holding each `VarId`'s value in `insts`, indexed by `VarId`.
/// Inputs never emitted stay `None`.
pub(crate) fn input_slots(
    insts: &IndexVec<Slot, Instr>,
    n_vars: u32,
) -> IndexVec<VarId, Option<Slot>> {
    let mut input_slot: IndexVec<VarId, Option<Slot>> =
        index_vec::index_vec![None; n_vars as usize];
    for (slot, &inst) in insts.iter_enumerated() {
        if let Instr::Input(v) = inst {
            input_slot[v] = Some(slot);
        }
    }
    input_slot
}

/// Push one derivative row — `adj` reduced to input order — onto `outputs`.
/// Inputs the seed doesn't depend on contribute an explicit shared zero.
pub(crate) fn push_input_adjoints(
    tb: &mut TapeBuilder,
    outputs: &mut Vec<Slot>,
    input_slot: &IndexVec<VarId, Option<Slot>>,
    adj: &IndexVec<Slot, Option<Slot>>,
    zero: &mut Option<Slot>,
) {
    for v in input_slot.indices() {
        let g = match input_slot[v].and_then(|s| adj[s]) {
            Some(g) => g,
            None => *zero.get_or_insert_with(|| tb.constant(0.0)),
        };
        outputs.push(g);
    }
}

/// Reverse-mode differentiation of `tape.outputs[output_idx]` w.r.t. all
/// `n_vars` inputs (get n_vars from `ctx.n_vars()`).
///
/// The result keeps all original outputs and appends the gradient:
///   outputs = [original outputs..., df/dv0, df/dv1, ..., df/dv(n-1)]
pub fn gradient(tape: &Tape, output_idx: usize, n_vars: u32) -> Tape {
    // Forward instructions become the prefix of the new tape, so every
    // forward Slot remains valid and refers to the same value.
    let mut tb = TapeBuilder {
        insts: tape.insts.clone(),
    };

    let adj = reverse_sweep(&mut tb, &tape.insts, tape.outputs[output_idx]);

    let input_slot = input_slots(&tape.insts, n_vars);
    let mut outputs = tape.outputs.clone();
    let mut zero: Option<Slot> = None;
    push_input_adjoints(&mut tb, &mut outputs, &input_slot, &adj, &mut zero);

    Tape {
        insts: tb.insts,
        outputs,
    }
}
