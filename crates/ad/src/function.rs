use index_vec::{IndexVec, define_index_type};

use crate::sparsity::Sparsity;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Sin,
    Cos,
    Sqrt,
    Exp,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

define_index_type! { pub struct Value = u32; }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueData {
    Input { input: u32, nonzero: u32 },
    Constant(f64),
    Unary(UnaryOp, Value),
    Binary(BinaryOp, Value, Value),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Output {
    pub sparsity: Sparsity,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    OperandOutOfOrder {
        at: Value,
        operand: Value,
    },
    InputIndexOutOfRange {
        at: Value,
        input: u32,
    },
    NonzeroOutOfRange {
        at: Value,
        input: u32,
        nonzero: u32,
    },
    OutputValueOutOfRange {
        output: usize,
        position: usize,
        value: Value,
    },
    OutputLengthMismatch {
        output: usize,
        expected: usize,
        got: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// Sparsity pattern of inputs.
    sparsity_in: Vec<Sparsity>,

    /// Tape.
    tape: IndexVec<Value, ValueData>,

    /// Function outputs.
    outputs: Vec<Output>,
}

impl Function {
    pub fn new(
        sparsity_in: Vec<Sparsity>,
        tape: IndexVec<Value, ValueData>,
        outputs: Vec<Output>,
    ) -> Result<Function, VerifyError> {
        let f = Function {
            sparsity_in,
            tape,
            outputs,
        };
        f.verify()?;
        Ok(f)
    }

    pub fn num_inputs(&self) -> usize {
        self.sparsity_in.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn input_nonzero_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.num_inputs() + 1);
        let mut sum = 0;
        offsets.push(0);
        for s in &self.sparsity_in {
            sum += s.nnz();
            offsets.push(sum);
        }
        offsets
    }

    /// Computes the global sparsity pattern of the Jacobian for the nonzero output
    /// variables with respect to the nonzero input variables.
    ///
    /// The nonzero inputs and outputs are flattened in column major and concatenated.
    pub fn jacobian_sparsity(&self) -> Sparsity {
        let num_rows = self.outputs.iter().map(|o| o.sparsity.nnz()).sum();
        let num_cols = self.sparsity_in.iter().map(|s| s.nnz()).sum();
        let mut columns: Vec<Vec<u32>> = vec![Vec::new(); num_cols];
        let col_offsets = self.input_nonzero_offsets();

        let mut dep: IndexVec<Value, u64> = IndexVec::with_capacity(self.tape.len());
        dep.resize(self.tape.len(), 0);

        for block in (0..num_cols).step_by(64) {
            for (value, data) in self.tape.iter_enumerated() {
                let b = match *data {
                    ValueData::Constant(..) => 0u64,
                    ValueData::Input { input, nonzero } => {
                        let j = col_offsets[input as usize] + nonzero as usize;
                        if block <= j && j < block + 64 {
                            1u64 << (j - block)
                        } else {
                            0u64
                        }
                    }
                    ValueData::Unary(.., v) => dep[v],
                    ValueData::Binary(.., lhs, rhs) => dep[lhs] | dep[rhs],
                };
                dep[value] = b;
            }
            let mut row = 0u32;
            for out in &self.outputs {
                for &v in &out.values {
                    let mut bits = dep[v];
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        columns[block + bit].push(row);
                        bits &= bits - 1 // Clears lowest nonzero bit.
                    }
                    row += 1;
                }
            }
        }

        Sparsity::from_columns(num_rows, num_cols, columns)
    }

    /// Computes the gradient of a scalar output function using a single sweep of reverse
    /// mode automatic differentation.
    pub fn gradient(&self) -> Function {
        assert_eq!(
            self.num_outputs(),
            1,
            "Function must only have one output to compute
        gradient"
        );
        assert_eq!(
            self.outputs[0].sparsity.nnz(),
            1,
            "Gradient is only defined for scalar
        outputs"
        );

        let mut b = FunctionBuilder::new(self.sparsity_in.clone());
        let map = b.replay(self);

        let one = b.constant(1.0);
        let seeds = vec![vec![Some(one)]];
        let adjoints = self.reverse(&mut b, &map, &seeds);

        let outputs = adjoints
            .into_iter()
            .zip(&self.sparsity_in)
            .map(|(adj, sparsity)| b.gather(sparsity.clone(), adj))
            .collect();
        b.finish(outputs)
    }

    pub fn jacobian(&self) -> Function {
        todo!("Implement Function::jacobian")
    }

    /// Computes the Hessian using forward over reverse automatic differentation.
    pub fn hessian(&self) -> Function {
        self.gradient().jacobian_forward()
    }

    /// Computes the Jacobian using forward mode automatic differentation (one sweep per
    /// input variable).
    pub fn jacobian_forward(&self) -> Function {
        let mut b = FunctionBuilder::new(self.sparsity_in.clone());
        let map = b.replay(self);
        let one = b.constant(1.0);

        todo!("Implement Function::jacobian_forward")
    }

    /// Computes the Jacobian using reverse mode automatic differentation (one sweep per
    /// output variable).
    pub fn jacobian_reverse(&self) -> Function {
        todo!("Implement Function::jacobian_reverse")
    }

    /// The forward mode autodiff primitive which gradient, Jacobian, and Hessian
    /// computations are built on.
    ///
    /// Given tangent seeds for every input nonzero, emit the tangent program into the
    /// function builder, and return the tangent of each output nonzero.
    fn forward(
        &self,
        b: &mut FunctionBuilder,
        map: &IndexVec<Value, Value>,
        seeds: &[Vec<Option<Value>>],
    ) -> Vec<Vec<Option<Value>>> {
        assert_eq!(seeds.len(), self.num_inputs());
        for (i, s) in seeds.iter().enumerate() {
            assert_eq!(s.len(), self.sparsity_in[i].nnz())
        }

        let mut tangent: IndexVec<Value, Option<Value>> = IndexVec::with_capacity(self.tape.len());

        for (old, data) in self.tape.iter_enumerated() {
            let t = match *data {
                ValueData::Input { input, nonzero } => seeds[input as usize][nonzero as usize],
                ValueData::Constant(x) => None,
                ValueData::Unary(op, x) => match tangent[x] {
                    Some(dx) => Some(chain_unary(b, op, map[x], map[old], dx)),
                    None => None,
                },
                ValueData::Binary(op, lhs, rhs) => {
                    let (l, r, v) = (map[lhs], map[rhs], map[old]);
                    let l_dot = match tangent[lhs] {
                        Some(s) => Some(chain_binary_lhs(b, op, l, r, v, s)),
                        None => None,
                    };
                    let r_dot = match tangent[rhs] {
                        Some(s) => Some(chain_binary_rhs(b, op, l, r, v, s)),
                        None => None,
                    };
                    b.add_opt(l_dot, r_dot)
                }
            };
            tangent.push(t);
        }

        self.outputs
            .iter()
            .map(|output| output.values.iter().map(|v| tangent[*v]).collect())
            .collect()
    }

    /// The reverse mode autodiff primitive which gradient, Jacobian, and Hessian
    /// computations are built on.
    ///
    /// Given adjoint seeds for every output nonzero, emit the adjoint program into the
    /// function builder, and return the adjoint of each input nonzero.
    fn reverse(
        &self,
        b: &mut FunctionBuilder,
        map: &IndexVec<Value, Value>,
        seeds: &[Vec<Option<Value>>],
    ) -> Vec<Vec<Option<Value>>> {
        todo!("Implement function::forward")
    }

    fn verify(&self) -> Result<(), VerifyError> {
        // Tape must be topologically sorted.
        for (value, data) in self.tape.iter_enumerated() {
            match data {
                ValueData::Input { .. } => (),
                ValueData::Constant(..) => (),
                ValueData::Unary(.., operand) => {
                    if *operand >= value {
                        return Err(VerifyError::OperandOutOfOrder {
                            at: value,
                            operand: *operand,
                        });
                    }
                }
                ValueData::Binary(.., lhs, rhs) => {
                    for &operand in [lhs, rhs] {
                        if operand >= value {
                            return Err(VerifyError::OperandOutOfOrder { at: value, operand });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Computes `s * d f(x)/dx` for a unary op with primal operand `x` and result `y`.
fn chain_unary(b: &mut FunctionBuilder, op: UnaryOp, x: Value, y: Value, s: Value) -> Value {
    match op {
        UnaryOp::Neg => b.neg(s),
        UnaryOp::Sin => {
            let cx = b.cos(x);
            b.mul(s, cx)
        }
        UnaryOp::Cos => {
            let sx = b.sin(x);
            let ns = b.neg(sx);
            b.mul(s, ns)
        }
        UnaryOp::Sqrt => {
            let two = b.constant(2.0);
            let ty = b.mul(two, y);
            b.div(s, ty)
        }
        UnaryOp::Exp => b.mul(s, y),
        UnaryOp::Log => b.div(s, x),
    }
}

fn chain_binary_lhs(
    b: &mut FunctionBuilder,
    op: BinaryOp,
    l: Value,
    r: Value,
    v: Value,
    s: Value,
) -> Value {
    match op {
        BinaryOp::Add => s,
        BinaryOp::Sub => s,
        BinaryOp::Mul => b.mul(s, r),
        BinaryOp::Div => b.div(s, r),
    }
}

fn chain_binary_rhs(
    b: &mut FunctionBuilder,
    op: BinaryOp,
    l: Value,
    r: Value,
    v: Value,
    s: Value,
) -> Value {
    match op {
        BinaryOp::Add => s,
        BinaryOp::Sub => b.neg(s),
        BinaryOp::Mul => b.mul(s, l),
        BinaryOp::Div => {
            let m = b.mul(s, v);
            let d = b.div(m, r);
            b.neg(d)
        }
    }
}

struct FunctionBuilder {
    sparsity_in: Vec<Sparsity>,
    tape: IndexVec<Value, ValueData>,
}

impl FunctionBuilder {
    pub fn new(sparsity_in: Vec<Sparsity>) -> Self {
        FunctionBuilder {
            sparsity_in,
            tape: IndexVec::new(),
        }
    }

    pub fn input(&mut self, input: u32, nonzero: u32) -> Value {
        self.tape.push(ValueData::Input { input, nonzero })
    }

    pub fn constant(&mut self, x: f64) -> Value {
        self.tape.push(ValueData::Constant(x))
    }

    pub fn unary(&mut self, op: UnaryOp, operand: Value) -> Value {
        self.tape.push(ValueData::Unary(op, operand))
    }

    pub fn binary(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Value {
        self.tape.push(ValueData::Binary(op, lhs, rhs))
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        self.binary(BinaryOp::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.binary(BinaryOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.binary(BinaryOp::Mul, lhs, rhs)
    }

    pub fn div(&mut self, lhs: Value, rhs: Value) -> Value {
        self.binary(BinaryOp::Div, lhs, rhs)
    }

    pub fn neg(&mut self, v: Value) -> Value {
        self.unary(UnaryOp::Neg, v)
    }

    pub fn sin(&mut self, v: Value) -> Value {
        self.unary(UnaryOp::Sin, v)
    }

    pub fn cos(&mut self, v: Value) -> Value {
        self.unary(UnaryOp::Cos, v)
    }

    pub fn add_opt(&mut self, lhs: Option<Value>, rhs: Option<Value>) -> Option<Value> {
        match (lhs, rhs) {
            (Some(l), Some(r)) => Some(self.add(l, r)),
            (None, Some(r)) => Some(r),
            (Some(l), None) => Some(l),
            (None, None) => None,
        }
    }

    pub fn replay(&mut self, f: &Function) -> IndexVec<Value, Value> {
        assert_eq!(self.sparsity_in, f.sparsity_in);

        let mut map = IndexVec::with_capacity(f.tape.len());
        for data in f.tape.iter() {
            let v = match *data {
                ValueData::Input { input, nonzero } => self.input(input, nonzero),
                ValueData::Constant(x) => self.constant(x),
                ValueData::Unary(op, operand) => self.unary(op, map[operand]),
                ValueData::Binary(op, lhs, rhs) => self.binary(op, map[lhs], map[rhs]),
            };
            map.push(v);
        }
        map
    }

    pub fn gather(&mut self, sparsity: Sparsity, values: Vec<Option<Value>>) -> Output {
        assert_eq!(values.len(), sparsity.nnz());
        let values = values
            .into_iter()
            .map(|v| v.unwrap_or_else(|| self.constant(0.0)))
            .collect();
        Output { sparsity, values }
    }

    pub fn finish(self, outputs: Vec<Output>) -> Function {
        Function::new(self.sparsity_in, self.tape, outputs)
            .expect("FunctionBuilder produced invalid function")
    }
}
