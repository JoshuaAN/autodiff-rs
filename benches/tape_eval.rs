//! Stress benchmark for tape evaluation throughput, over both backends.
//!
//! Place at `benches/tape_eval.rs` and add to Cargo.toml:
//!
//! ```toml
//! [dev-dependencies]
//! criterion = "0.5"
//!
//! [[bench]]
//! name = "tape_eval"
//! harness = false
//! ```
//!
//! Run with `cargo bench --bench tape_eval`. JIT compile time per tape is
//! printed to stderr (it is setup cost, not part of the measured eval).
//! NOTE: JIT compilation of the 1M-instruction tapes takes tens of seconds
//! (regalloc over one giant block); trim SIZES while iterating.
//!
//! Every tape is compiled by BOTH backends and their outputs cross-checked
//! bitwise before timing, so this doubles as a property test of the JIT
//! against the interpreter oracle on large random tapes.
//!
//! The generator evaluates the tape concretely (at the same args the
//! benchmark will use) *while building it*, and rejects any candidate
//! instruction whose result leaves [1e-100, 1e100]. This is the only sound
//! way to prevent overflow and denormals: multiplicative "squaring chains"
//! double a value's exponent per step (10 steps take 2 to 2^1024), so at
//! stress-test sizes every seed eventually diverges without rejection, and
//! static bound-tracking can't help because `sin` destroys lower bounds.

use std::hint::black_box;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use index_vec::IndexVec;

use tape::{
    backend::{Backend, CompiledFunction},
    interpreter::{apply_binary, apply_unary, Interpreter},
    jit::Cranelift,
    op::{BinaryOp, UnaryOp},
    tape::{Inst, InstructionData, Tape, Value, ValueData},
};

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64) — no `rand` dependency, reproducible seeds.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in [1.0, 2.0): safely away from zero and denormals.
    fn f64(&mut self) -> f64 {
        f64::from_bits(0x3FF0_0000_0000_0000 | (self.next() >> 12))
    }

    /// Picks an index into the last `window` defined values. Small windows
    /// give cache-friendly operand access (like tapes traced from real
    /// programs); `usize::MAX` gives uniformly scattered operands.
    fn pick(&mut self, len: usize, window: usize) -> usize {
        let lo = len.saturating_sub(window);
        lo + self.below(len - lo)
    }
}

// ---------------------------------------------------------------------------
// Random tape generation with concrete rejection sampling.
// ---------------------------------------------------------------------------

/// Values are kept inside [1e-100, 1e100] (or exactly zero). The margins
/// guarantee that no single operation on in-range operands can jump past the
/// check into denormal (>= 1e-100 * 1e-100 = 1e-200) or non-finite
/// (<= 1e100 * 1e100 = 1e200) territory undetected — every excursion is
/// caught and rejected at the instruction that would cause it.
#[inline]
fn ok(x: f64) -> bool {
    x.is_finite() && (x == 0.0 || (1e-100..=1e100).contains(&x.abs()))
}

/// Rejected candidates are replaced by `sin(a)`, which is always in-range —
/// except in the astronomically rare case that sin(a) is itself subnormal,
/// where `cos(a)` (near +/-1 exactly when sin is near 0) is used instead.
fn fallback(tape: &mut Tape, a: Value, ca: f64) -> (Value, f64) {
    let s = ca.sin();
    if ok(s) {
        (tape.unary(UnaryOp::Sin, a), s)
    } else {
        (tape.unary(UnaryOp::Cos, a), ca.cos())
    }
}

fn random_tape(target_insts: usize, window: usize, seed: u64, args: &[f64]) -> Tape {
    let mut rng = Rng(seed);
    let mut tape = Tape::default();

    // The value pool and, in lockstep, each value's concrete result at `args`.
    let mut vals: Vec<Value> = (0..args.len()).map(|_| tape.param()).collect();
    let mut conc: Vec<f64> = args.to_vec();

    // Shared constant for the guarded-division pattern.
    let two = tape.constant(2.0);

    let mut rejected = 0usize;

    while tape.layout().len() < target_insts {
        let ia = rng.pick(vals.len(), window);
        let ib = rng.pick(vals.len(), window);
        let (a, ca) = (vals[ia], conc[ia]);
        let (b, cb) = (vals[ib], conc[ib]);

        let (v, x) = match rng.below(100) {
            roll @ 0..=59 => {
                let op = match roll {
                    0..=24 => BinaryOp::Add,
                    25..=39 => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                let cand = apply_binary(op, ca, cb);
                if ok(cand) {
                    (tape.binary(op, a, b), cand)
                } else {
                    rejected += 1;
                    fallback(&mut tape, a, ca)
                }
            }
            // Guarded division: a / (sin(b) + 2). The denominator lies in
            // [1, 3], so the quotient is bounded by |a|; a raw random
            // denominator would spray infinities.
            60..=67 => {
                let s = cb.sin();
                let q = ca / (s + 2.0);
                if ok(s) && ok(q) {
                    let sv = tape.unary(UnaryOp::Sin, b);
                    let dv = tape.binary(BinaryOp::Add, sv, two);
                    (tape.binary(BinaryOp::Div, a, dv), q)
                } else {
                    rejected += 1;
                    fallback(&mut tape, a, ca)
                }
            }
            roll @ 68..=93 => {
                let op = match roll {
                    68..=79 => UnaryOp::Sin,
                    80..=87 => UnaryOp::Cos,
                    _ => UnaryOp::Neg,
                };
                let cand = apply_unary(op, ca);
                if ok(cand) {
                    (tape.unary(op, a), cand)
                } else {
                    rejected += 1;
                    fallback(&mut tape, a, ca)
                }
            }
            _ => {
                let c = rng.f64();
                (tape.constant(c), c)
            }
        };

        vals.push(v);
        conc.push(x);
    }

    if rejected > 0 {
        eprintln!(
            "[random_tape] {rejected}/{} candidates rejected and replaced with sin/cos",
            tape.layout().len()
        );
    }

    for &v in vals.iter().rev().take(4) {
        tape.returns.push(v);
    }
    tape
}

/// Safety net: one evaluation at the generation args, then a scan of the
/// ENTIRE instruction buffer (not just the returns) for non-finite values
/// and denormals. Subnormal arithmetic is 10-100x slower on x86 and silently
/// poisons timing comparisons. This walks the tape itself (rather than going
/// through a backend) precisely because it needs every intermediate, which
/// the `CompiledFunction` ABI deliberately doesn't expose.
fn validate(tape: &Tape, args: &[f64]) {
    let mut buf: IndexVec<Inst, f64> = IndexVec::from_vec(vec![0.0; tape.insts.len()]);
    let get = |buf: &IndexVec<Inst, f64>, v: Value| match tape.values[v] {
        ValueData::Param(i) => args[i as usize],
        ValueData::Result(i) => buf[i],
    };
    for &inst in tape.layout() {
        let x = match tape.inst_data(inst) {
            InstructionData::Constant(c) => c.to_f64(),
            InstructionData::Unary(op, a) => apply_unary(op, get(&buf, a)),
            InstructionData::Binary(op, a, b) => apply_binary(op, get(&buf, a), get(&buf, b)),
        };
        buf[inst] = x;
    }

    let non_finite = buf.iter().filter(|x| !x.is_finite()).count();
    let denormal = buf.iter().filter(|x| x.is_subnormal()).count();
    assert_eq!(non_finite, 0, "generator invariant violated: {non_finite} non-finite values");
    assert_eq!(denormal, 0, "generator invariant violated: {denormal} denormal values");
}

// ---------------------------------------------------------------------------
// Benchmark.
// ---------------------------------------------------------------------------

fn bench_eval(c: &mut Criterion) {
    const N_PARAMS: usize = 16;
    const SEED: u64 = 0xC0FFEE;
    // Trim while iterating: JIT compilation of the 1M tapes takes tens of
    // seconds (one-time setup cost per tape, printed to stderr).
    const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000, 4_000_000];

    let mut interp_backend = Interpreter;
    let mut jit_backend = Cranelift::new().expect("host ISA setup failed");

    let mut group = c.benchmark_group("tape_eval");

    for &size in SIZES {
        for &(label, window) in &[("local", 256usize), ("scattered", usize::MAX)] {
            // Args are chosen FIRST: the generator evaluates at these exact
            // args while building, so benchmark-time evaluation is guaranteed
            // to stay in range.
            let mut rng = Rng(SEED ^ 1);
            let args: Vec<f64> = (0..N_PARAMS).map(|_| rng.f64()).collect();

            let tape = random_tape(size, window, SEED, &args);
            validate(&tape, &args);

            let mut interp = interp_backend.compile(&tape).unwrap();
            let t0 = Instant::now();
            let mut jit = jit_backend.compile(&tape).expect("jit compile failed");
            eprintln!("[jit] {label}/{size}: compiled in {:?}", t0.elapsed());

            // Cross-check: the interpreter is the oracle; the JIT must agree
            // bitwise. This makes the benchmark double as a property test.
            let mut r_interp = vec![0.0; tape.returns.len()];
            let mut r_jit = vec![0.0; tape.returns.len()];
            interp.call(&args, &mut r_interp);
            jit.call(&args, &mut r_jit);
            for (i, (x, y)) in r_interp.iter().zip(&r_jit).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "backend mismatch at return {i} ({label}/{size}): {x} vs {y}"
                );
            }

            // Throughput in instructions/sec: comparable across sizes and
            // between backends.
            group.throughput(Throughput::Elements(tape.layout().len() as u64));

            // Buffers allocated once: we measure evaluation, not allocation.
            let mut returns = vec![0.0; tape.returns.len()];

            group.bench_function(BenchmarkId::new(format!("interp_{label}"), size), |b| {
                b.iter(|| {
                    interp.call(black_box(&args), &mut returns);
                    black_box(&returns);
                });
            });
            group.bench_function(BenchmarkId::new(format!("jit_{label}"), size), |b| {
                b.iter(|| {
                    jit.call(black_box(&args), &mut returns);
                    black_box(&returns);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(bench, bench_eval);
criterion_main!(bench);