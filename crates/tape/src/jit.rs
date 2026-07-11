//! Cranelift JIT backend: compiles a tape to straight-line native code.
//!
//! Add to Cargo.toml — ALL FOUR crates are required, with matching versions
//! (verified against 0.107.2; newer versions occasionally rename things,
//! e.g. `finalize_definitions` gaining a `Result`):
//!
//! ```toml
//! [dependencies]
//! cranelift = "0.107"
//! cranelift-jit = "0.107"
//! cranelift-module = "0.107"
//! cranelift-native = "0.107"
//! ```
//!
//! Kernel ABI: `extern "C" fn(args: *const f64, returns: *mut f64,
//! scratch: *mut f64)`. Structure and constants are baked in at compile
//! time; the evaluation point is a runtime argument, so one compilation
//! serves every input.
//!
//! # Chunking
//!
//! Large tapes are split into multiple functions of at most `chunk_size`
//! instructions, called in sequence; values that cross a chunk boundary are
//! spilled to per-value slots in the scratch buffer. This bounds two things
//! that otherwise blow up on giant single-block functions:
//!
//! - **Code size per function.** AArch64 PC-relative references (constant
//!   pool literals etc.) reach only +/-1MB; multi-MB single functions can
//!   trip label-range assertions in the machine buffer. x86-64 is unaffected
//!   (RIP-relative reaches +/-2GB), which is why the same tape can compile
//!   on one machine and panic on another.
//! - **Register-allocation time**, which grows superlinearly with the
//!   number of simultaneously live values in one function.
//!
//! `sin`/`cos` are emitted as calls to registered symbols that wrap
//! `f64::sin`/`f64::cos` — the *same* functions the interpreter uses, and
//! Cranelift performs no IEEE-unsafe reassociation, so JIT and interpreter
//! results are bitwise identical (the tests assert exactly that).

use cranelift::codegen::isa::OwnedTargetIsa;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, ModuleError};
use index_vec::IndexVec;

use crate::{
    backend::{Backend, CompiledFunction},
    op::{BinaryOp, UnaryOp},
    tape::{Inst, InstructionData, Tape, Value as TapeValue, ValueData},
};

/// Symbols registered with the JIT for transcendentals. Using our own
/// wrappers (rather than resolving libm's `sin` from the process) guarantees
/// bitwise agreement with the interpreter.
extern "C" fn jit_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn jit_cos(x: f64) -> f64 {
    x.cos()
}

type KernelFn = unsafe extern "C" fn(*const f64, *mut f64, *mut f64);

/// Default maximum instructions per compiled function. At typical code
/// densities this keeps each function far below AArch64's +/-1MB
/// PC-relative range, and keeps per-function regalloc cheap.
const DEFAULT_CHUNK_SIZE: usize = 16_384;

#[derive(Debug)]
pub enum JitError {
    /// Host ISA construction or flag configuration failed.
    Isa(String),
    /// Cranelift module-level failure (declaration, definition, finalize).
    Module(ModuleError),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Isa(e) => write!(f, "ISA setup failed: {e}"),
            JitError::Module(e) => write!(f, "cranelift module error: {e}"),
        }
    }
}

impl std::error::Error for JitError {}

impl From<ModuleError> for JitError {
    fn from(e: ModuleError) -> Self {
        JitError::Module(e)
    }
}

/// JIT backend. Holds the (expensive-to-construct) host ISA; each compiled
/// tape gets its own `JITModule` so kernels have independent lifetimes.
pub struct Cranelift {
    isa: OwnedTargetIsa,
    chunk_size: usize,
}

impl Cranelift {
    /// Detects the host CPU (enabling its features, e.g. FMA on x86 with
    /// AVX2) and configures codegena for speed.
    pub fn new() -> Result<Self, JitError> {
        let mut flags = settings::builder();
        for (k, v) in [
            ("use_colocated_libcalls", "false"),
            ("is_pic", "false"),
            ("opt_level", "speed"),
        ] {
            flags
                .set(k, v)
                .map_err(|e| JitError::Isa(format!("setting {k}: {e}")))?;
        }
        let isa = cranelift_native::builder()
            .map_err(|e| JitError::Isa(e.to_string()))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| JitError::Isa(e.to_string()))?;
        Ok(Self {
            isa,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }

    /// Overrides the maximum instructions per compiled function. Mostly for
    /// tests (tiny sizes exercise boundary spilling) and tuning.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        self.chunk_size = chunk_size;
        self
    }
}

impl Backend for Cranelift {
    type Func = JitFunction;
    type Error = JitError;

    fn compile(&mut self, tape: &Tape) -> Result<JitFunction, JitError> {
        let layout = tape.layout();
        let n_chunks = (layout.len().div_ceil(self.chunk_size)).max(1);
        let last_chunk = (n_chunks - 1) as u32;

        // ------------------------------------------------------------------
        // Plan: which chunk defines each instruction, and which results
        // cross a chunk boundary (used by a later chunk, or returned from a
        // non-final chunk). Crossers get a slot in the scratch buffer.
        // ------------------------------------------------------------------
        let mut chunk_of: IndexVec<Inst, u32> =
            IndexVec::from_vec(vec![u32::MAX; tape.insts.len()]);
        for (pos, &inst) in layout.iter().enumerate() {
            chunk_of[inst] = (pos / self.chunk_size) as u32;
        }

        let mut slot: IndexVec<Inst, Option<u32>> =
            IndexVec::from_vec(vec![None; tape.insts.len()]);
        let mut n_slots: u32 = 0;
        {
            let mut mark = |v: TapeValue, c: u32| {
                if let ValueData::Result(i) = tape.values[v] {
                    if chunk_of[i] != c && slot[i].is_none() {
                        slot[i] = Some(n_slots);
                        n_slots += 1;
                    }
                }
            };
            for (pos, &inst) in layout.iter().enumerate() {
                let c = (pos / self.chunk_size) as u32;
                match tape.inst_data(inst) {
                    InstructionData::Constant(_) => {}
                    InstructionData::Unary(_, a) => mark(a, c),
                    InstructionData::Binary(_, a, b) => {
                        mark(a, c);
                        mark(b, c);
                    }
                }
            }
            // Returns are all written by the last chunk.
            for &r in &tape.returns {
                mark(r, last_chunk);
            }
        }

        // ------------------------------------------------------------------
        // Codegen: one function per chunk, all in one module.
        // ------------------------------------------------------------------
        let mut jb =
            JITBuilder::with_isa(self.isa.clone(), cranelift_module::default_libcall_names());
        jb.symbol("jit_sin", jit_sin as *const u8);
        jb.symbol("jit_cos", jit_cos as *const u8);
        let mut module = JITModule::new(jb);

        let ptr_ty = module.target_config().pointer_type();

        // Kernel signature: (args: *const f64, returns: *mut f64,
        // scratch: *mut f64) — identical for every chunk.
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));

        // f64 -> f64, for the sin/cos imports.
        let mut unary_sig = module.make_signature();
        unary_sig.params.push(AbiParam::new(types::F64));
        unary_sig.returns.push(AbiParam::new(types::F64));

        let sin_id = module.declare_function("jit_sin", Linkage::Import, &unary_sig)?;
        let cos_id = module.declare_function("jit_cos", Linkage::Import, &unary_sig)?;
        let func_ids: Vec<FuncId> = (0..n_chunks)
            .map(|c| module.declare_function(&format!("tape_kernel_{c}"), Linkage::Export, &sig))
            .collect::<Result<_, _>>()?;

        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();

        // Per-instruction cache of the current chunk's SSA value, tagged
        // with a generation (chunk index + 1) so stale entries from earlier
        // chunks — which are ir::Values of a *different function* — are
        // never reused. Allocated once, never cleared.
        let mut local: IndexVec<Inst, Option<(u32, Value)>> =
            IndexVec::from_vec(vec![None; tape.insts.len()]);

        // The caller (JitFunction::call) asserts slice lengths and owns the
        // scratch buffer, so these accesses cannot trap and are 8-byte
        // aligned.
        let mem = MemFlags::trusted();

        for c in 0..n_chunks {
            let gena = c as u32 + 1;
            ctx.func.signature = sig.clone();
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

            let sin_ref = module.declare_func_in_func(sin_id, b.func);
            let cos_ref = module.declare_func_in_func(cos_id, b.func);

            // Each chunk is pure straight-line code: one block, no branches.
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);

            let args_ptr = b.block_params(entry)[0];
            let rets_ptr = b.block_params(entry)[1];
            let scratch_ptr = b.block_params(entry)[2];

            // Load every param up front; Cranelift drops unused ones.
            let params: Vec<Value> = (0..tape.num_params())
                .map(|i| b.ins().load(types::F64, mem, args_ptr, (i * 8) as i32))
                .collect();

            // Operand resolution: params from the arg loads; same-chunk
            // results from the cache; earlier-chunk results loaded (once)
            // from their scratch slot.
            fn resolve(
                b: &mut FunctionBuilder,
                tape: &Tape,
                slot: &IndexVec<Inst, Option<u32>>,
                local: &mut IndexVec<Inst, Option<(u32, Value)>>,
                params: &[Value],
                scratch_ptr: Value,
                mem: MemFlags,
                gena: u32,
                v: TapeValue,
            ) -> Value {
                match tape.values[v] {
                    ValueData::Param(i) => params[i as usize],
                    ValueData::Result(i) => {
                        if let Some((g, val)) = local[i] {
                            if g == gena {
                                return val;
                            }
                        }
                        let s = slot[i].expect("cross-chunk operand without a scratch slot");
                        let off =
                            i32::try_from(s as u64 * 8).expect("scratch offset overflows i32");
                        let val = b.ins().load(types::F64, mem, scratch_ptr, off);
                        local[i] = Some((gena, val));
                        val
                    }
                }
            }

            macro_rules! get {
                ($b:expr, $v:expr) => {
                    resolve($b, tape, &slot, &mut local, &params, scratch_ptr, mem, gena, $v)
                };
            }

            let lo = c * self.chunk_size;
            let hi = layout.len().min(lo + self.chunk_size);
            for &inst in &layout[lo..hi] {
                let x = match tape.inst_data(inst) {
                    InstructionData::Constant(cst) => b.ins().f64const(cst.to_f64()),
                    InstructionData::Unary(op, a) => {
                        let a = get!(&mut b, a);
                        match op {
                            UnaryOp::Neg => b.ins().fneg(a),
                            UnaryOp::Sin => {
                                let call = b.ins().call(sin_ref, &[a]);
                                b.inst_results(call)[0]
                            }
                            UnaryOp::Cos => {
                                let call = b.ins().call(cos_ref, &[a]);
                                b.inst_results(call)[0]
                            }
                        }
                    }
                    InstructionData::Binary(op, av, bv) => {
                        let l = get!(&mut b, av);
                        let r = get!(&mut b, bv);
                        match op {
                            BinaryOp::Add => b.ins().fadd(l, r),
                            BinaryOp::Sub => b.ins().fsub(l, r),
                            BinaryOp::Mul => b.ins().fmul(l, r),
                            BinaryOp::Div => b.ins().fdiv(l, r),
                        }
                    }
                };
                local[inst] = Some((gena, x));
                // Boundary-crossing values are spilled to their scratch slot
                // at definition; within the chunk the SSA value keeps being
                // used directly.
                if let Some(s) = slot[inst] {
                    let off = i32::try_from(s as u64 * 8).expect("scratch offset overflows i32");
                    b.ins().store(mem, x, scratch_ptr, off);
                }
            }

            // The last chunk writes every return (loading earlier-chunk
            // values from scratch as needed; params resolve naturally).
            if c == n_chunks - 1 {
                for (j, &r) in tape.returns.iter().enumerate() {
                    let v = get!(&mut b, r);
                    b.ins().store(mem, v, rets_ptr, (j * 8) as i32);
                }
            }

            b.ins().return_(&[]);
            b.finalize();

            module.define_function(func_ids[c], &mut ctx)?;
            module.clear_context(&mut ctx);
        }

        module.finalize_definitions()?;

        let chunks: Vec<KernelFn> = func_ids
            .iter()
            .map(|&id| {
                let code = module.get_finalized_function(id);
                // SAFETY: `code` points to a finalized function with exactly
                // the KernelFn signature we declared, and `JitFunction`
                // keeps the module (and thus the executable mapping) alive
                // as long as the pointers.
                unsafe { std::mem::transmute::<*const u8, KernelFn>(code) }
            })
            .collect();

        Ok(JitFunction {
            module: Some(module),
            chunks,
            scratch: vec![0.0; n_slots as usize],
            n_params: tape.num_params(),
            n_returns: tape.returns.len(),
        })
    }
}

/// A compiled kernel: one or more chunk functions called in sequence,
/// communicating through the owned scratch buffer. Owns the `JITModule`
/// whose mapping backs the pointers; the mapping is freed exactly once, on
/// drop, after which the pointers are never used.
pub struct JitFunction {
    module: Option<JITModule>,
    chunks: Vec<KernelFn>,
    scratch: Vec<f64>,
    n_params: usize,
    n_returns: usize,
}

impl CompiledFunction for JitFunction {
    fn num_params(&self) -> usize {
        self.n_params
    }

    fn num_returns(&self) -> usize {
        self.n_returns
    }

    fn call(&mut self, args: &[f64], returns: &mut [f64]) {
        assert_eq!(args.len(), self.n_params, "wrong number of args");
        assert_eq!(returns.len(), self.n_returns, "wrong number of returns");
        let scratch = self.scratch.as_mut_ptr();
        for &f in &self.chunks {
            // SAFETY: lengths asserted above; each chunk reads at most
            // n_params f64s from `args`, writes at most n_returns f64s to
            // `returns`, and reads/writes scratch slots within
            // `self.scratch` (slots were allocated to cover every spill);
            // the code mapping is alive because `self.module` is Some until
            // drop. Chunks run in layout order, so every scratch slot read
            // was written by an earlier chunk in this same call.
            unsafe { f(args.as_ptr(), returns.as_mut_ptr(), scratch) }
        }
    }
}

impl Drop for JitFunction {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: the chunk pointers are unreachable after drop; nothing
            // else holds pointers into this module's memory.
            unsafe { module.free_memory() };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: the interpreter is the oracle; results must be bitwise identical.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;

    /// f(x, y) = ( cos(-(sin(x*y)/y + 0.5)) - x , sin(x*y)/y )
    /// — exercises every op, constant, multi-use values, multiple returns.
    fn sample_tape() -> Tape {
        let mut t = Tape::default();
        let x = t.param();
        let y = t.param();
        let xy = t.binary(BinaryOp::Mul, x, y);
        let s = t.unary(UnaryOp::Sin, xy);
        let q = t.binary(BinaryOp::Div, s, y);
        let c = t.constant(0.5);
        let z = t.binary(BinaryOp::Add, q, c);
        let n = t.unary(UnaryOp::Neg, z);
        let co = t.unary(UnaryOp::Cos, n);
        let d = t.binary(BinaryOp::Sub, co, x);
        t.returns.push(d);
        t.returns.push(q);
        t
    }

    fn assert_backends_agree(tape: &Tape, jit: &mut JitFunction, argsets: &[Vec<f64>]) {
        let mut interp = Interpreter.compile(tape).unwrap();
        for args in argsets {
            let mut a = vec![0.0; tape.returns.len()];
            let mut b = vec![0.0; tape.returns.len()];
            interp.call(args, &mut a);
            jit.call(args, &mut b);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.to_bits(), y.to_bits(), "args {args:?}: {x} vs {y}");
            }
        }
    }

    fn argsets() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![0.3, -1.7],
            vec![-4.2, 0.001],
            vec![1e10, 3.5],
        ]
    }

    #[test]
    fn jit_matches_interpreter_bitwise() {
        let tape = sample_tape();
        let mut jit = Cranelift::new().unwrap().compile(&tape).unwrap();
        assert_backends_agree(&tape, &mut jit, &argsets());
    }

    /// Tiny chunk sizes force nearly every operand across a boundary,
    /// torturing the scratch spill/reload path.
    #[test]
    fn chunked_matches_interpreter_bitwise() {
        let tape = sample_tape();
        for chunk_size in [1, 2, 3, 5] {
            let mut jit = Cranelift::new()
                .unwrap()
                .with_chunk_size(chunk_size)
                .compile(&tape)
                .unwrap();
            assert_eq!(jit.chunks.len(), tape.layout().len().div_ceil(chunk_size));
            assert_backends_agree(&tape, &mut jit, &argsets());
        }
    }

    /// A long dependent chain crossing many default-size... (small here)
    /// chunks, with a value from the FIRST chunk used in the LAST.
    #[test]
    fn long_range_cross_chunk_use() {
        let mut t = Tape::default();
        let x = t.param();
        let early = t.binary(BinaryOp::Mul, x, x); // defined in chunk 0
        let mut acc = x;
        for _ in 0..100 {
            let s = t.unary(UnaryOp::Sin, acc);
            acc = t.binary(BinaryOp::Add, s, x);
        }
        let out = t.binary(BinaryOp::Mul, acc, early); // uses chunk-0 value
        t.returns.push(out);

        let mut jit = Cranelift::new()
            .unwrap()
            .with_chunk_size(16)
            .compile(&t)
            .unwrap();
        assert_backends_agree(&t, &mut jit, &[vec![0.7], vec![-2.3]]);
    }

    #[test]
    fn constant_only_tape() {
        let mut t = Tape::default();
        let c = t.constant(42.5);
        t.returns.push(c);

        let mut jit = Cranelift::new().unwrap().compile(&t).unwrap();
        let mut out = [0.0];
        jit.call(&[], &mut out);
        assert_eq!(out[0], 42.5);
    }

    /// Param-passthrough: empty layout, returns reference params directly.
    #[test]
    fn param_passthrough() {
        let mut t = Tape::default();
        let x = t.param();
        let y = t.param();
        t.returns.push(y);
        t.returns.push(x);

        let mut jit = Cranelift::new().unwrap().compile(&t).unwrap();
        let mut out = [0.0; 2];
        jit.call(&[1.5, 2.5], &mut out);
        assert_eq!(out, [2.5, 1.5]);
    }
}