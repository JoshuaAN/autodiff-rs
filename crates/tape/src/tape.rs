use std::cell::RefCell;

use index_vec::{IndexVec, define_index_type, index_vec};

use crate::{
    bits::Bits64,
    function::{Function, Inst, InstructionData},
    op::{BinaryOp, UnaryOp},
};

define_index_type! { pub struct Value = u32; }

pub enum ValueData {
    /// Input parameter, identified by a flat index in creation order.
    Param(u32),

    /// Constant, stored by bit pattern so `Eq`/`Hash` are well-defined.
    Constant(Bits64),

    /// Unary expression, e.g. `x <- sin(a)`.
    Unary(UnaryOp, Value),

    /// Binary expression, e.g. `x <- a + b`.
    Binary(BinaryOp, Value, Value),
}

#[derive(Default)]
struct TapeInner {
    nodes: IndexVec<Value, ValueData>,
    num_params: u32,
}

impl TapeInner {
    fn param(&mut self) -> Value {
        let data = ValueData::Param(self.num_params);
        self.num_params += 1;
        self.push(data)
    }

    fn constant(&mut self, x: f64) -> Value {
        self.push(ValueData::Constant(Bits64::from_f64(x)))
    }

    fn unary(&mut self, op: UnaryOp, v: Value) -> Value {
        self.push(ValueData::Unary(op, v))
    }

    fn binary(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Value {
        self.push(ValueData::Binary(op, lhs, rhs))
    }

    fn push(&mut self, data: ValueData) -> Value {
        self.nodes.push(data)
    }
}

pub struct Tape {
    inner: RefCell<TapeInner>,
}

#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    value: Value,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            inner: RefCell::new(TapeInner::default()),
        }
    }

    pub fn param(&self) -> Var<'_> {
        let value = self.inner.borrow_mut().param();
        Var { tape: self, value }
    }

    pub fn constant(&self, x: f64) -> Var<'_> {
        let value = self.inner.borrow_mut().constant(x);
        Var { tape: self, value }
    }

    #[inline]
    fn check_owns(&self, v: Var<'_>, what: &str) {
        assert!(
            std::ptr::eq(v.tape, self),
            "{what}: Var belongs to a different tape"
        );
    }

    pub fn compile(&self, inputs: &[Var<'_>], outputs: &[Var<'_>]) -> Function {
        for v in inputs {
            self.check_owns(*v, "compile input");
        }
        for v in outputs {
            self.check_owns(*v, "compile output");
        }

        let inner = self.inner.borrow();
        let nodes = &inner.nodes;

        // --- Reachability: backward sweep over the (topo-ordered) tape. ---
        let mut live: IndexVec<Value, bool> = index_vec![false; nodes.len()];
        for v in outputs {
            live[v.value] = true;
        }
        for idx in (0..nodes.len()).rev() {
            let idx = Value::from_usize(idx);
            if !live[idx] {
                continue;
            }
            match nodes[idx] {
                ValueData::Unary(_, a) => live[a] = true,
                ValueData::Binary(_, a, b) => {
                    live[a] = true;
                    live[b] = true;
                }
                ValueData::Param(_) | ValueData::Constant(_) => {}
            }
        }

        // --- Emit: forward pass, renumbering tape Values into dense Insts. ---
        let mut insts: IndexVec<Inst, InstructionData> = IndexVec::new();
        let mut remap: IndexVec<Value, Option<Inst>> = index_vec![None; nodes.len()];

        // Parameters first, numbered by their position in `inputs` — the
        // function's calling convention is what the caller declared, so even
        // unused inputs get a slot.
        for (i, v) in inputs.iter().enumerate() {
            if !matches!(nodes[v.value], ValueData::Param(_)) {
                panic!("compile: input {i} is not a parameter");
            }
            if remap[v.value].is_some() {
                panic!("compile: input {i} was already listed");
            }
            remap[v.value] = Some(insts.push(InstructionData::Param(i as u32)));
        }

        for (idx, node) in nodes.iter_enumerated() {
            if !live[idx] || remap[idx].is_some() {
                continue; // dead, or an already-emitted input param
            }
            let data = match *node {
                ValueData::Param(_) => {
                    panic!("compile: outputs depend on a parameter not listed in inputs")
                }
                ValueData::Constant(bits) => InstructionData::Constant(bits),
                // remap[..] is Some here because the tape is topologically
                // ordered and liveness was propagated to all operands.
                ValueData::Unary(op, a) => InstructionData::Unary(op, remap[a].unwrap()),
                ValueData::Binary(op, a, b) => {
                    InstructionData::Binary(op, remap[a].unwrap(), remap[b].unwrap())
                }
            };
            remap[idx] = Some(insts.push(data));
        }

        let returns = outputs.iter().map(|v| remap[v.value].unwrap()).collect();

        Function {
            insts,
            num_params: inputs.len() as u32,
            returns,
        }
    }
}

impl<'t> Var<'t> {
    fn push_unary(self, op: UnaryOp) -> Var<'t> {
        let value = self.tape.inner.borrow_mut().unary(op, self.value);
        Var {
            tape: self.tape,
            value,
        }
    }

    fn push_binary(self, op: BinaryOp, rhs: Var<'t>) -> Var<'t> {
        self.tape.check_owns(rhs, "binary op");
        let value = self
            .tape
            .inner
            .borrow_mut()
            .binary(op, self.value, rhs.value);
        Var {
            tape: self.tape,
            value,
        }
    }
}

macro_rules! impl_unary_fns {
    ($($method:ident => $op:expr),* $(,)?) => {
        impl<'t> Var<'t> {
            $(
                pub fn $method(self) -> Var<'t> {
                    self.push_unary($op)
                }
            )*
        }
    };
}

impl_unary_fns! {
    sin => UnaryOp::Sin,
    cos => UnaryOp::Cos,
    neg => UnaryOp::Neg,
}

impl<'t> std::ops::Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Var<'t> {
        Var::neg(self)
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        // Var op Var
        impl<'t> std::ops::$trait for Var<'t> {
            type Output = Var<'t>;
            fn $method(self, rhs: Var<'t>) -> Var<'t> {
                self.push_binary($op, rhs)
            }
        }

        // Var op f64
        impl<'t> std::ops::$trait<f64> for Var<'t> {
            type Output = Var<'t>;
            fn $method(self, rhs: f64) -> Var<'t> {
                let rhs = self.tape.constant(rhs);
                self.push_binary($op, rhs)
            }
        }

        // f64 op Var
        impl<'t> std::ops::$trait<Var<'t>> for f64 {
            type Output = Var<'t>;
            fn $method(self, rhs: Var<'t>) -> Var<'t> {
                let lhs = rhs.tape.constant(self);
                lhs.push_binary($op, rhs)
            }
        }
    };
}

impl_binary_op!(Add, add, BinaryOp::Add);
impl_binary_op!(Sub, sub, BinaryOp::Sub);
impl_binary_op!(Mul, mul, BinaryOp::Mul);
impl_binary_op!(Div, div, BinaryOp::Div);
