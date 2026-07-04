//! Jacobians and Hessians, built on the reverse-mode sweep in [`crate::grad`].
//!
//! Both work the same way as [`gradient`](crate::grad::gradient): the forward
//! tape becomes the prefix of the result and every derivative is appended as
//! extra outputs, so a single [`eval`](crate::interpreter::eval) yields the
//! primal(s) and the whole matrix in one pass.

use crate::{
    grad::{TapeBuilder, gradient, input_slots, push_input_adjoints, reverse_sweep},
    tape::{Slot, Tape},
};

/// Reverse-mode Jacobian of every output w.r.t. all `n_vars` inputs.
///
/// One reverse sweep is run per output, all sharing the forward tape as a
/// common prefix. The result keeps the original outputs and appends the
/// Jacobian in row-major order (output-major, then input):
///   outputs = [f0..f(m-1), df0/dv0..df0/dv(n-1), df1/dv0.., ..., df(m-1)/..]
///
/// For a single-output tape this equals [`gradient`](crate::grad::gradient).
pub fn jacobian(tape: &Tape, n_vars: u32) -> Tape {
    let mut tb = TapeBuilder {
        insts: tape.insts.clone(),
    };

    let input_slot = input_slots(&tape.insts, n_vars);
    let mut outputs = tape.outputs.clone();
    let mut zero: Option<Slot> = None;

    for &out in &tape.outputs {
        let adj = reverse_sweep(&mut tb, &tape.insts, out);
        push_input_adjoints(&mut tb, &mut outputs, &input_slot, &adj, &mut zero);
    }

    Tape {
        insts: tb.insts,
        outputs,
    }
}

/// Reverse-over-reverse Hessian of the scalar `tape.outputs[output_idx]`
/// w.r.t. all `n_vars` inputs.
///
/// First the gradient tape is built; each of its `n_vars` gradient components
/// is then differentiated again w.r.t. every input. Because the second sweep
/// runs over the *full* first-derivative tape, it correctly differentiates
/// through the instructions the gradient itself introduced.
///
/// The result keeps the primal and gradient and appends the Hessian in
/// row-major order:
///   outputs = [f, df/dv0..df/dv(n-1),
///              d2f/dv0dv0, d2f/dv0dv1, ..., d2f/dv(n-1)dv(n-1)]
///
/// Assuming continuous second derivatives the Hessian is symmetric, so this
/// deliberately recomputes both triangles rather than trusting equality of
/// mixed partials at non-smooth points (e.g. `abs`, `min`/`max`).
pub fn hessian(tape: &Tape, output_idx: usize, n_vars: u32) -> Tape {
    // First-derivative tape: outputs = [orig outputs..., df/dv0..df/dv(n-1)].
    let g = gradient(tape, output_idx, n_vars);
    let n = n_vars as usize;
    // The gradient rows are the last `n` outputs `gradient` appended.
    let grad_start = g.outputs.len() - n;

    let mut tb = TapeBuilder {
        insts: g.insts.clone(),
    };

    let input_slot = input_slots(&g.insts, n_vars);
    let mut outputs = g.outputs.clone();
    let mut zero: Option<Slot> = None;

    for i in 0..n {
        let row = g.outputs[grad_start + i];
        let adj = reverse_sweep(&mut tb, &g.insts, row);
        push_input_adjoints(&mut tb, &mut outputs, &input_slot, &adj, &mut zero);
    }

    Tape {
        insts: tb.insts,
        outputs,
    }
}
