use std::convert::Infallible;

use index_vec::IndexVec;

use crate::{backend::{Backend, CompiledFunction}, op::{BinaryOp, UnaryOp}, tape::{Inst, InstructionData, Tape, Value, ValueData}};

#[inline]
pub fn apply_unary(op: UnaryOp, a: f64) -> f64 {
    match op {
        UnaryOp::Neg => -a,
        UnaryOp::Sin => a.sin(),
        UnaryOp::Cos => a.cos(),
    }
}
 
#[inline]
pub fn apply_binary(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
    }
}

#[derive(Default)]
pub struct Interpreter;
 
pub struct InterpretedFunc {
    tape: Tape,
    /// Result of each instruction, reused across calls.
    buf: IndexVec<Inst, f64>,
}
 
impl Backend for Interpreter {
    type Func = InterpretedFunc;
    type Error = Infallible;
 
    fn compile(&mut self, tape: &Tape) -> Result<InterpretedFunc, Infallible> {
        Ok(InterpretedFunc {
            buf: IndexVec::from_vec(vec![0.0; tape.insts.len()]),
            tape: tape.clone(),
        })
    }
}
 
#[inline]
fn operand(tape: &Tape, args: &[f64], buf: &IndexVec<Inst, f64>, v: Value) -> f64 {
    match tape.values[v] {
        ValueData::Param(i) => args[i as usize],
        ValueData::Result(i) => buf[i],
    }
}
 
impl CompiledFunction for InterpretedFunc {
    fn num_params(&self) -> usize {
        self.tape.num_params()
    }
 
    fn num_returns(&self) -> usize {
        self.tape.returns.len()
    }
 
    fn call(&mut self, args: &[f64], returns: &mut [f64]) {
        assert_eq!(args.len(), self.tape.num_params(), "wrong number of args");
        assert_eq!(returns.len(), self.tape.returns.len(), "wrong number of returns");
 
        let tape = &self.tape;
        for &inst in tape.layout() {
            let x = match tape.inst_data(inst) {
                InstructionData::Constant(c) => c.to_f64(),
                InstructionData::Unary(op, a) => {
                    apply_unary(op, operand(tape, args, &self.buf, a))
                }
                InstructionData::Binary(op, a, b) => apply_binary(
                    op,
                    operand(tape, args, &self.buf, a),
                    operand(tape, args, &self.buf, b),
                ),
            };
            self.buf[inst] = x;
        }
 
        for (out, &r) in returns.iter_mut().zip(&tape.returns) {
            *out = operand(tape, args, &self.buf, r);
        }
    }
}
