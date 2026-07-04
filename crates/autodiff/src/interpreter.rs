use index_vec::IndexVec;

use crate::{
    tape::{Instr, Slot, Tape},
    var::VarId,
};

pub fn eval(tape: &Tape, inputs: &IndexVec<VarId, f64>, out: &mut Vec<f64>) {
    let mut vals: IndexVec<Slot, f64> = index_vec::index_vec![0.0; tape.insts.len()];

    for (slot, inst) in tape.insts.iter_enumerated() {
        vals[slot] = match inst {
            Instr::Const(c) => *c,
            Instr::Input(v) => inputs[*v],
            Instr::Unary(op, a) => op.apply(vals[*a]),
            Instr::Binary(op, a, b) => op.apply(vals[*a], vals[*b]),
        };
    }
    out.clear();
    out.extend(tape.outputs.iter().map(|&s| vals[s]));
}
