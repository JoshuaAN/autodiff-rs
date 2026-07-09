//! Pretty-printing for the IR.
//!
//! Two audiences, one printer:
//!   - reading a program        -> names, aligned columns, a `return` line
//!   - debugging a *transform*  -> arena indices, detached instructions,
//!                                 liveness, the things the semantics hides
//!
//! Options are opt-in via a builder, so the default `{}` stays quiet:
//!
//!     println!("{}", f.print().named("grad"));
//!     println!("{}", f.print().compact().liveness().detached());

use std::collections::HashMap;
use std::fmt;

use crate::{
    function::Function,
    instruction::{Inst, InstructionData},
    op::{BinaryOp, UnaryOp},
    value::Value,
};

// ---------------------------------------------------------------- opcodes

fn unary_name(o: UnaryOp) -> &'static str {
    match o {
        UnaryOp::Neg => "neg",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
    }
}
fn binary_name(o: BinaryOp) -> &'static str {
    match o {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Div => "div",
        BinaryOp::Mul => "mul",
    }
}

/// Constants round-trip: Rust's `{:?}` for f64 prints the shortest decimal that
/// parses back exactly. We append raw bits only when the decimal form is
/// ambiguous about the thing we actually hash on (-0.0, NaN payloads).
fn fmt_const(x: f64) -> String {
    let needs_bits = (x == 0.0 && x.is_sign_negative()) || !x.is_finite();
    if needs_bits {
        format!("{:?} /*0x{:016x}*/", x, x.to_bits())
    } else {
        format!("{:?}", x)
    }
}

// ---------------------------------------------------------------- printer

#[derive(Clone, Copy, Default)]
struct Opts {
    compact: bool,
    liveness: bool,
    detached: bool,
    types: bool,
}

pub struct Printer<'a> {
    f: &'a Function,
    name: &'a str,
    opts: Opts,
}

impl<'a> Printer<'a> {
    pub fn new(f: &'a Function) -> Self {
        Printer {
            f,
            name: "func",
            opts: Opts::default(),
        }
    }
    /// Rename instruction results by layout position instead of arena index.
    /// Stable output after DCE/GVN; loses the ability to cross-reference the arena.
    pub fn compact(mut self) -> Self {
        self.opts.compact = true;
        self
    }
    /// Annotate each line with how many values are live across it.
    pub fn liveness(mut self) -> Self {
        self.opts.liveness = true;
        self
    }
    /// List instructions that exist in the DFG but are not placed in the layout.
    pub fn detached(mut self) -> Self {
        self.opts.detached = true;
        self
    }
    pub fn types(mut self) -> Self {
        self.opts.types = true;
        self
    }
    pub fn named(mut self, n: &'a str) -> Self {
        self.name = n;
        self
    }

    fn positions(&self) -> HashMap<Inst, usize> {
        self.f
            .layout()
            .iter()
            .enumerate()
            .map(|(p, &i)| (i, p))
            .collect()
    }

    fn name_of(&self, v: Value, pos: &HashMap<Inst, usize>) -> String {
        match v {
            Value::Param(p) => format!("p{}", p),
            Value::Result(i) => {
                if self.opts.compact {
                    match pos.get(&i) {
                        Some(p) => format!("v{p}"),
                        None => format!("v?{}", i.index()),
                    }
                } else {
                    format!("v{}", i.index())
                }
            }
        }
    }

    /// live_across[p] = values defined at or before p whose last use is after p.
    fn live_counts(&self, pos: &HashMap<Inst, usize>) -> Vec<usize> {
        let n = self.f.layout().len();
        let mut last: HashMap<Inst, usize> = HashMap::new();
        for (p, &i) in self.f.layout().iter().enumerate() {
            for a in self.f.args(i) {
                if let Value::Result(j) = a {
                    last.insert(j, p);
                }
            }
        }
        for r in self.f.returns.iter() {
            if let Value::Result(j) = r {
                last.insert(*j, n);
            }
        }
        (0..n)
            .map(|p| {
                self.f
                    .layout()
                    .iter()
                    .filter(|&&i| pos[&i] <= p && last.get(&i).is_some_and(|&l| l > p))
                    .count()
            })
            .collect()
    }
}

impl fmt::Display for Printer<'_> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        let f = self.f;
        let pos = self.positions();
        let ty = if self.opts.types { ": f64" } else { "" };

        // signature
        let params: Vec<String> = (0..f.num_params).map(|k| format!("p{k}{ty}")).collect();
        let rets: Vec<String> = f.returns.iter().map(|v| self.name_of(*v, &pos)).collect();
        writeln!(
            w,
            "function %{}({}) -> {} x f64 {{",
            self.name,
            params.join(", "),
            f.returns.len()
        )?;

        // column widths: longest result name, so the `=` signs line up
        let dst_w = f
            .layout()
            .iter()
            .map(|&i| self.name_of(Value::Result(i), &pos).len())
            .max()
            .unwrap_or(2)
            .max(6); // 6 = width of "return"

        let live = if self.opts.liveness {
            self.live_counts(&pos)
        } else {
            vec![]
        };

        // two passes: render bodies, then pad the comment column
        let bodies: Vec<String> = f
            .layout()
            .iter()
            .map(|&i| match f.inst_data(i) {
                InstructionData::Constant(c) => format!("f64const {}", fmt_const(c.to_f64())),
                InstructionData::Unary(o, a) => {
                    format!("{:<8} {}", unary_name(o), self.name_of(a, &pos))
                }
                InstructionData::Binary(o, a, b) => format!(
                    "{:<8} {}, {}",
                    binary_name(o),
                    self.name_of(a, &pos),
                    self.name_of(b, &pos)
                ),
            })
            .collect();
        let body_w = bodies.iter().map(|b| b.len()).max().unwrap_or(0);

        for (p, &i) in f.layout().iter().enumerate() {
            let dst = self.name_of(Value::Result(i), &pos);
            if self.opts.liveness {
                writeln!(
                    w,
                    "    {:>dw$} = {:<bw$}   ; live {}",
                    dst,
                    bodies[p],
                    live[p],
                    dw = dst_w,
                    bw = body_w
                )?;
            } else {
                writeln!(w, "    {:>dw$} = {}", dst, bodies[p], dw = dst_w)?;
            }
        }

        // returns as a pseudo-instruction, so the exit is visible in the body
        writeln!(
            w,
            "    {:>dst_w$}   return   {}",
            "",
            rets.join(", "),
            dst_w = dst_w
        )?;
        write!(w, "}}")?;

        // instructions in the arena but not in the layout — the DFG/Layout split
        // made visible. Nothing else in the IR can show you these.
        if self.opts.detached {
            let dead: Vec<Inst> = f.detached().collect();
            if !dead.is_empty() {
                write!(
                    w,
                    "\n; detached ({} inst{} in the DFG, unplaced):",
                    dead.len(),
                    if dead.len() == 1 { "" } else { "s" }
                )?;
                for i in dead {
                    let body = match f.inst_data(i) {
                        InstructionData::Constant(c) => {
                            format!("f64const {}", fmt_const(c.to_f64()))
                        }
                        InstructionData::Unary(o, a) => {
                            format!("{} {}", unary_name(o), self.name_of(a, &pos))
                        }
                        InstructionData::Binary(o, a, b) => format!(
                            "{} {}, {}",
                            binary_name(o),
                            self.name_of(a, &pos),
                            self.name_of(b, &pos)
                        ),
                    };
                    write!(w, "\n;   v{} = {}", i.index(), body)?;
                }
            }
        }
        Ok(())
    }
}

/// `{}` on a Function is the quiet default.
impl fmt::Display for Function {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        Printer::new(self).fmt(w)
    }
}

/// A single instruction, for `dbg!` and error messages.
pub struct InstDisp<'a>(pub &'a Function, pub Inst);
impl fmt::Display for InstDisp<'_> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = Printer::new(self.0);
        let pos = p.positions();
        match self.0.inst_data(self.1) {
            InstructionData::Constant(c) => write!(
                w,
                "v{} = f64const {}",
                self.1.index(),
                fmt_const(c.to_f64())
            ),
            InstructionData::Unary(o, a) => write!(
                w,
                "v{} = {} {}",
                self.1.index(),
                unary_name(o),
                p.name_of(a, &pos)
            ),
            InstructionData::Binary(o, a, b) => write!(
                w,
                "v{} = {} {}, {}",
                self.1.index(),
                binary_name(o),
                p.name_of(a, &pos),
                p.name_of(b, &pos)
            ),
        }
    }
}
