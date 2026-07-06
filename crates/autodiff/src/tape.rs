use std::{collections::HashMap, fmt};

use index_vec::{IndexVec, define_index_type};

use crate::{
    expression::{Expression, Node}, op::{BinaryOp, UnaryOp}, var::VarId,
};

define_index_type! { pub struct InstrId = u32; }
define_index_type! { pub struct Slot = u32; }

#[derive(Clone, Copy, Debug)]
pub enum Instr {
    Const(f64),
    Unary(UnaryOp, Slot),
    Binary(BinaryOp, Slot, Slot),
}

pub struct Tape {
    /// Number of function parameters.
    pub num_inputs: u32,

    /// Instructions used in the tape. Notes:
    ///   - Slots 0..num_inputs are reserved for the input parameters.
    ///   - instrs[i] represents slot num_inputs + i.
    pub instrs: IndexVec<InstrId, Instr>,

    /// List of slots which correspond to outputs in the tape.
    pub outputs: Vec<Slot>,
}

impl Tape {
    pub fn from_exprs(exprs: &[Expression], inputs: &[Expression]) -> Tape {
        let num_inputs = inputs.len() as u32;

        let mut var_slot: HashMap<VarId, Slot> = HashMap::with_capacity(inputs.len());
        for (i, e) in inputs.iter().enumerate() {
            let Node::Variable(v) = e.node() else {
                panic!("input {i} is not a variable expression: {:?}", e.node());
            };
            let prev = var_slot.insert(*v, Slot::from(i));
            assert!(prev.is_none(), "duplicate input variable {v:?}");
        }

        let mut instrs: IndexVec<InstrId, Instr> = IndexVec::new();
        let mut memo: HashMap<*const Node, Slot> = HashMap::new();

        let mut push_instr = |instrs: &mut IndexVec<InstrId, Instr>, i: Instr| -> Slot {
            instrs.push(i);
            Slot::from(num_inputs as usize + instrs.len() - 1)
        };

        enum Task<'a> {
            Visit(&'a Expression),
            Emit(&'a Expression),
        }

        let mut stack: Vec<Task> = exprs.iter().rev().map(Task::Visit).collect();

        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(e) => {
                    if memo.contains_key(&e.as_ptr()) {
                        continue;
                    }
                    match e.node() {
                        Node::Variable(v) => {
                            let slot = *var_slot
                                .get(v)
                                .unwrap_or_else(|| panic!("variable {v:?} not in inputs"));
                            memo.insert(e.as_ptr(), slot);
                        }
                        Node::Constant(c) => {
                            let s = push_instr(&mut instrs, Instr::Const(*c));
                            memo.insert(e.as_ptr(), s);
                        }
                        Node::Unary(_, a) => {
                            stack.push(Task::Emit(e));
                            stack.push(Task::Visit(a));
                        }
                        Node::Binary(_, a, b) => {
                            stack.push(Task::Emit(e));
                            stack.push(Task::Visit(a));
                            stack.push(Task::Visit(b));
                        }
                    }
                }
                Task::Emit(e) => {
                    if memo.contains_key(&e.as_ptr()) {
                        continue;
                    }
                    let s = match e.node() {
                        Node::Unary(op, a) => {
                            let a = memo[&a.as_ptr()];
                            push_instr(&mut instrs, Instr::Unary(*op, a))
                        }
                        Node::Binary(op, a, b) => {
                            let a = memo[&a.as_ptr()];
                            let b = memo[&b.as_ptr()];
                            push_instr(&mut instrs, Instr::Binary(*op, a, b))
                        }
                        _ => unreachable!("leaves handled in Visit"),
                    };
                    memo.insert(e.as_ptr(), s);
                }
            }
        }

        let outputs = exprs.iter().map(|e| memo[&e.as_ptr()]).collect();
        Tape { num_inputs, instrs, outputs }
    }

    pub fn num_slots(&self) -> usize {
        self.num_inputs as usize + self.instrs.len()
    }

    pub fn slot_of(&self, i: InstrId) -> Slot {
        Slot::from((self.num_inputs + i.raw()) as usize)
    }

    pub fn eval(&self, args: &[f64]) -> Vec<f64> {
        assert_eq!(
            args.len(),
            self.num_inputs as usize,
            "expected {} inputs, got {}",
            self.num_inputs,
            args.len()
        );

        let mut vals: Vec<f64> = Vec::with_capacity(self.num_slots());
        vals.extend_from_slice(args);

        for instr in &self.instrs {
            let v = match *instr {
                Instr::Const(c) => c,
                Instr::Unary(op, a) => op.apply(vals[a.index()]),
                Instr::Binary(op, a, b) => op.apply(vals[a.index()], vals[b.index()]),
            };
            vals.push(v);
        }

        self.outputs.iter().map(|&s| vals[s.index()]).collect()
    }
}

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (slot, instr) in self.instrs.iter_enumerated() {
            match *instr {
                Instr::Const(c) => writeln!(f, "{:?} = const {}", slot, c)?,
                Instr::Unary(op, a) => writeln!(f, "{:?} = {:?} {:?}", slot, op, a)?,
                Instr::Binary(op, a, b) => writeln!(f, "{:?} = {:?} {:?} {:?}", slot, op, a, b)?,
            }
        }
        writeln!(f, "outputs: {:?}", self.outputs)
    }
}
