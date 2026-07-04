use index_vec::IndexVec;

use crate::{context::Context, interpreter::eval, tape::{Instr, Tape}, var::VarId};

pub mod node;
pub mod interpreter;
pub mod op;
pub mod tape;
pub mod var;
pub mod context;
pub mod grad;

use std::fmt;

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (slot, inst) in self.insts.iter_enumerated() {
            match *inst {
                Instr::Const(c)         => writeln!(f, "{:?} = const {}", slot, c)?,
                Instr::Input(v)         => writeln!(f, "{:?} = input {:?}", slot, v)?,
                Instr::Unary(op, a)     => writeln!(f, "{:?} = {:?} {:?}", slot, op, a)?,
                Instr::Binary(op, a, b) => writeln!(f, "{:?} = {:?} {:?} {:?}", slot, op, a, b)?,
            }
        }
        writeln!(f, "outputs: {:?}", self.outputs)
    }
}

fn main() {
    let ctx = Context::new();
    let x = ctx.var();
    let y = ctx.var();

    let r2 = &x * &x + &y * &y;
    let f = r2.sqrt().sin() / (1.0 + r2.sqrt());

    let tape = ctx.lower(&[&f]);

    println!("--- tape ({} insts) ---", tape.insts.len());
    println!("{}", tape);

    // evaluate at (3, 4): r2 = 25, sqrt = 5
    let inputs: IndexVec<VarId, f64> = index_vec::index_vec![3.0, 4.0];
    let mut out = Vec::new();
    eval(&tape, &inputs, &mut out);

    let expected = (5.0f64).sin() / (1.0 + 5.0);
    println!("got      = {}", out[0]);
    println!("expected = {}", expected);
    assert_eq!(out[0], expected, "mismatch");
    println!("ok!");
}
