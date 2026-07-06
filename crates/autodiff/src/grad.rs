use index_vec::IndexVec;

use crate::{op::{BinaryOp, UnaryOp}, tape::{Instr::{self, Const}, InstrId, Slot, Tape}};

struct TapeBuilder {
  num_inputs: u32,
   instrs: IndexVec<InstrId, Instr>,
}

impl TapeBuilder {
  fn push(&mut self, i: Instr) -> Slot {
    self.instrs.push(i);
    Slot::from(self.num_inputs as usize + self.instrs.len() - 1)
  }

  fn constant(&mut self, c: f64) -> Slot { self.push(Instr::Const(c)) }

  fn unary(&mut self, op: UnaryOp, a: Slot) -> Slot { self.push(Instr::Unary(op, a)) }
  
  fn binary(&mut self, op: BinaryOp, a: Slot, b: Slot) -> Slot { self.push(Instr::Binary(op, a, b)) }
}

pub fn gradient(tape: &Tape, output: Slot) -> Tape {
  let mut builder = TapeBuilder {
    num_inputs: tape.num_inputs,
    instrs: tape.instrs.clone(),
  };

  let mut adjoint: Vec<Option<Slot>> = vec![None; tape.num_slots()];
  adjoint[output.index()] = Some(builder.push(Const(1.0)));

  for (i, instr) in tape.instrs.iter_enumerated().rev() {
    let slot = tape.slot_of(i);
    let Some(adj) = adjoint[slot.index()] else { continue };

    let contribute = |e: &mut TapeBuilder, adjoint: &mut Vec<Option<Slot>>, target: Slot, c: Slot| {
        adjoint[target.index()] = Some(match adjoint[target.index()] {
            None => c,
            Some(prev) => e.binary(BinaryOp::Add, prev, c),
        });
    };

    match *instr {
      Instr::Const(_) => {}
      Instr::Unary(op, x) => {
        let c = match op {
          UnaryOp::Neg  => builder.unary(UnaryOp::Neg, adj),
          UnaryOp::Sin  => { let c = builder.unary(UnaryOp::Cos, x); builder.binary(BinaryOp::Mul, adj, c) }
          UnaryOp::Cos  => { let s = builder.unary(UnaryOp::Sin, x); let m = builder.binary(BinaryOp::Mul, adj, s); builder.unary(UnaryOp::Neg, m) }
          UnaryOp::Exp  => builder.binary(BinaryOp::Mul, adj, slot),
          UnaryOp::Ln   => builder.binary(BinaryOp::Div, adj, x),
          UnaryOp::Sqrt => {
              let two = builder.constant(2.0);
              let d = builder.binary(BinaryOp::Mul, two, slot);
              builder.binary(BinaryOp::Div, adj, d)
          }
        };
        contribute(&mut builder, &mut adjoint, x, c);
      }
      Instr::Binary(op, a, b) => {
        let (ca, cb) = match op {
          BinaryOp::Add => (adj, adj),
          BinaryOp::Sub => (adj, builder.unary(UnaryOp::Neg, adj)),
          BinaryOp::Mul => (builder.binary(BinaryOp::Mul, adj, b), builder.binary(BinaryOp::Mul, adj, a)),
          BinaryOp::Div => {
              // d/da = adj / b;  d/db = -adj * (a/b) / b = -adj * out / b
              let da = builder.binary(BinaryOp::Div, adj, b);
              let t = builder.binary(BinaryOp::Mul, adj, slot);
              let t = builder.binary(BinaryOp::Div, t, b);
              (da, builder.unary(UnaryOp::Neg, t))
          }
        };
        contribute(&mut builder, &mut adjoint, a, ca);
        contribute(&mut builder, &mut adjoint, b, cb);
      }
    }
  }

  let zero = builder.constant(0.0);
  let outputs = (0..tape.num_inputs)
      .map(|i| adjoint[i as usize].unwrap_or(zero))
      .collect();

  Tape { num_inputs: tape.num_inputs, instrs: builder.instrs, outputs }
}