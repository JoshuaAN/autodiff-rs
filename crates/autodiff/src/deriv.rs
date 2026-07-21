//! Source-transformation automatic differentiation.
//!
//! Both entry points are graph-to-graph: they take a frozen [`Function`] and
//! return a *new* [`Function`] whose outputs are directional derivatives. The
//! result is an ordinary graph, so it can be JIT-compiled like anything else,
//! or fed back into these transforms (forward-over-reverse = Hessian).
//!
//! * [`forward`]  — Jacobian-vector products `J · s` (tangent mode).
//! * [`reverse`]  — vector-Jacobian products `wᵀ · J` (adjoint mode).
//! * [`gradient`] — convenience: [`reverse`] with a single unit weight.
//! * [`prune`]    — dead-code elimination; the transforms call it themselves.
//!
//! Multiple seed/weight directions are fused into ONE output graph that shares
//! a single copy of the primal computation, which is exactly what the coloring
//! layer wants: hand `forward` one seed vector per color and you get a single
//! compressed-Jacobian function.
//!
//! Invariants relied upon:
//! * The tape is in SSA/topological order: a node's operands always have
//!   smaller `NodeId`s than the node itself.
//! * `Node: Copy`, `Bits64::to_f64` exists, and `Function` exposes
//!   `nodes()`, `inputs()`, `outputs()`, and `from_parts(..)`.
//!
//! `Param` nodes that are *not* listed in `Function::inputs()` are treated as
//! constants (zero tangent, no adjoint output).

use std::collections::HashMap;

use index_vec::{IndexVec, index_vec};

use crate::{
    bits::Bits64,
    function::Function,
    node::{Node, NodeId},
    op::{BinaryOp, UnaryOp},
};

// ---------------------------------------------------------------------------
// Builder: emits into a new graph with peephole simplification
// ---------------------------------------------------------------------------

/// Emits nodes into a fresh graph, folding constants and algebraic identities
/// (`1 * x`, `x + 0`, `-(-x)`, ...) as it goes.
///
/// Structural zeros are handled *outside* the builder: derivative slots are
/// `Option<NodeId>` where `None` means "known zero", so zero branches are
/// never emitted at all. The folds here mop up what the seed constants
/// introduce (e.g. `1 · y  →  y`).
///
/// Note: `0 * x → 0` style folds are not IEEE-faithful when `x` is `inf`/`NaN`.
/// This is the standard trade-off in AD code generation (CasADi does the
/// same); document it and move on.
struct Builder {
    nodes: IndexVec<NodeId, Node>,
    /// Constant dedup: f64 bit pattern → existing node.
    consts: HashMap<u64, NodeId>,
}

impl Builder {
    /// Start from a verbatim copy of the primal graph. Because the copy
    /// preserves `NodeId`s, primal ids remain valid in the new graph — the
    /// derivative rules below lean on this to reference primal values
    /// (e.g. `cos(x)` for the tangent of `sin(x)`).
    fn from_primal(func: &Function) -> Self {
        let nodes = func.nodes().clone();
        let mut consts = HashMap::new();
        for (id, node) in nodes.iter_enumerated() {
            if let Node::Constant(b) = *node {
                consts.entry(b.to_f64().to_bits()).or_insert(id);
            }
        }
        Builder { nodes, consts }
    }

    fn constant(&mut self, x: f64) -> NodeId {
        let bits = x.to_bits();
        if let Some(&id) = self.consts.get(&bits) {
            return id;
        }
        let id = self.nodes.push(Node::Constant(Bits64::from_f64(x)));
        self.consts.insert(bits, id);
        id
    }

    fn const_val(&self, id: NodeId) -> Option<f64> {
        match self.nodes[id] {
            Node::Constant(b) => Some(b.to_f64()),
            _ => None,
        }
    }

    fn unary(&mut self, op: UnaryOp, a: NodeId) -> NodeId {
        if let Some(x) = self.const_val(a) {
            let v = match op {
                UnaryOp::Neg => -x,
                UnaryOp::Sin => x.sin(),
                UnaryOp::Cos => x.cos(),
            };
            return self.constant(v);
        }
        // -(-x) = x
        if op == UnaryOp::Neg {
            if let Node::Unary(UnaryOp::Neg, inner) = self.nodes[a] {
                return inner;
            }
        }
        self.nodes.push(Node::Unary(op, a))
    }

    fn binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        let (ca, cb) = (self.const_val(a), self.const_val(b));
        if let (Some(x), Some(y)) = (ca, cb) {
            let v = match op {
                BinaryOp::Add => x + y,
                BinaryOp::Sub => x - y,
                BinaryOp::Mul => x * y,
                BinaryOp::Div => x / y,
            };
            return self.constant(v);
        }
        match op {
            BinaryOp::Add => {
                if ca == Some(0.0) {
                    return b;
                }
                if cb == Some(0.0) {
                    return a;
                }
            }
            BinaryOp::Sub => {
                if cb == Some(0.0) {
                    return a;
                }
                if ca == Some(0.0) {
                    return self.unary(UnaryOp::Neg, b);
                }
            }
            BinaryOp::Mul => {
                if ca == Some(0.0) || cb == Some(0.0) {
                    return self.constant(0.0);
                }
                if ca == Some(1.0) {
                    return b;
                }
                if cb == Some(1.0) {
                    return a;
                }
                if ca == Some(-1.0) {
                    return self.unary(UnaryOp::Neg, b);
                }
                if cb == Some(-1.0) {
                    return self.unary(UnaryOp::Neg, a);
                }
            }
            BinaryOp::Div => {
                if cb == Some(1.0) {
                    return a;
                }
                if cb == Some(-1.0) {
                    return self.unary(UnaryOp::Neg, a);
                }
                if ca == Some(0.0) {
                    return self.constant(0.0);
                }
            }
        }
        self.nodes.push(Node::Binary(op, a, b))
    }

    /// `slot += v` where `None` is structural zero.
    fn acc(&mut self, slot: Option<NodeId>, v: NodeId) -> Option<NodeId> {
        Some(match slot {
            None => v,
            Some(s) => self.binary(BinaryOp::Add, s, v),
        })
    }

    /// `slot -= v` where `None` is structural zero.
    fn acc_neg(&mut self, slot: Option<NodeId>, v: NodeId) -> Option<NodeId> {
        Some(match slot {
            None => self.unary(UnaryOp::Neg, v),
            Some(s) => self.binary(BinaryOp::Sub, s, v),
        })
    }
}

// ---------------------------------------------------------------------------
// Helper cache: derivative subexpressions shared across directions
// ---------------------------------------------------------------------------

/// Per-primal-node helper expressions that depend only on primal values, so
/// they are emitted once and reused by every seed/weight direction:
/// `cos(x)` for a `Sin` node, `sin(x)` for a `Cos` node, `1/den` for a `Div`.
/// Keyed by the id of the node whose derivative rule needs the helper.
struct Helpers {
    slots: IndexVec<NodeId, Option<NodeId>>,
}

impl Helpers {
    fn new(num_primal_nodes: usize) -> Self {
        Helpers {
            slots: index_vec![None; num_primal_nodes],
        }
    }

    fn get(
        &mut self,
        b: &mut Builder,
        id: NodeId,
        make: impl FnOnce(&mut Builder) -> NodeId,
    ) -> NodeId {
        if let Some(h) = self.slots[id] {
            h
        } else {
            let h = make(b);
            self.slots[id] = Some(h);
            h
        }
    }
}

// ---------------------------------------------------------------------------
// Forward mode (tangent): J · s
// ---------------------------------------------------------------------------

/// Forward-mode transform. For each seed direction `s ∈ seeds` (one entry per
/// function input), the returned function computes the tangents `J · s` of
/// every output.
///
/// Output layout is direction-major:
/// `out[d * n_out + k]` = derivative of output `k` along `seeds[d]`.
///
/// For compressed Jacobians, pass one 0/1 seed vector per color; the primal
/// computation is shared across all directions, and constant folding prunes
/// every zero-tangent branch — this is where the sparsity savings show up in
/// the emitted graph.
pub fn forward(func: &Function, seeds: &[&[f64]]) -> Function {
    let n_in = func.inputs().len();
    let n_primal = func.nodes().len();
    for (d, s) in seeds.iter().enumerate() {
        assert_eq!(
            s.len(),
            n_in,
            "forward: seed {d} has length {}, expected {n_in} (one per input)",
            s.len()
        );
    }

    let mut b = Builder::from_primal(func);
    let mut helpers = Helpers::new(n_primal);
    let mut outputs = Vec::with_capacity(seeds.len() * func.outputs().len());

    for seed in seeds {
        // Tangent of every primal node under this seed; None = structural zero.
        let mut tan: IndexVec<NodeId, Option<NodeId>> = index_vec![None; n_primal];

        for (j, &param) in func.inputs().iter().enumerate() {
            if seed[j] != 0.0 {
                let c = b.constant(seed[j]);
                tan[param] = Some(c);
            }
        }

        for (id, node) in func.nodes().iter_enumerated() {
            match *node {
                // Params were seeded above; constants have zero tangent.
                Node::Param(_) | Node::Constant(_) => {}

                Node::Unary(op, a) => {
                    let da = tan[a];
                    tan[id] = match op {
                        // d(-x) = -dx
                        UnaryOp::Neg => da.map(|d| b.unary(UnaryOp::Neg, d)),
                        // d(sin x) = cos(x) · dx
                        UnaryOp::Sin => da.map(|d| {
                            let c = helpers.get(&mut b, id, |b| b.unary(UnaryOp::Cos, a));
                            b.binary(BinaryOp::Mul, c, d)
                        }),
                        // d(cos x) = -sin(x) · dx
                        UnaryOp::Cos => da.map(|d| {
                            let s = helpers.get(&mut b, id, |b| b.unary(UnaryOp::Sin, a));
                            let t = b.binary(BinaryOp::Mul, s, d);
                            b.unary(UnaryOp::Neg, t)
                        }),
                    };
                }

                Node::Binary(op, l, r) => {
                    let (dl, dr) = (tan[l], tan[r]);
                    tan[id] = match op {
                        BinaryOp::Add => match (dl, dr) {
                            (None, None) => None,
                            (Some(x), None) | (None, Some(x)) => Some(x),
                            (Some(x), Some(y)) => Some(b.binary(BinaryOp::Add, x, y)),
                        },
                        BinaryOp::Sub => match (dl, dr) {
                            (None, None) => None,
                            (Some(x), None) => Some(x),
                            (None, Some(y)) => Some(b.unary(UnaryOp::Neg, y)),
                            (Some(x), Some(y)) => Some(b.binary(BinaryOp::Sub, x, y)),
                        },
                        // d(l·r) = dl·r + l·dr
                        BinaryOp::Mul => {
                            let t1 = dl.map(|d| b.binary(BinaryOp::Mul, d, r));
                            let t2 = dr.map(|d| b.binary(BinaryOp::Mul, l, d));
                            match (t1, t2) {
                                (None, None) => None,
                                (Some(x), None) | (None, Some(x)) => Some(x),
                                (Some(x), Some(y)) => Some(b.binary(BinaryOp::Add, x, y)),
                            }
                        }
                        // y = l/r:  dy = (dl - y·dr) / r  =  (dl - y·dr) · (1/r)
                        // `id` IS the primal y, reused directly.
                        BinaryOp::Div => {
                            let num = match (dl, dr) {
                                (None, None) => None,
                                (Some(x), None) => Some(x),
                                (None, Some(y)) => {
                                    let t = b.binary(BinaryOp::Mul, id, y);
                                    Some(b.unary(UnaryOp::Neg, t))
                                }
                                (Some(x), Some(y)) => {
                                    let t = b.binary(BinaryOp::Mul, id, y);
                                    Some(b.binary(BinaryOp::Sub, x, t))
                                }
                            };
                            num.map(|n| {
                                let recip = helpers.get(&mut b, id, |b| {
                                    let one = b.constant(1.0);
                                    b.binary(BinaryOp::Div, one, r)
                                });
                                b.binary(BinaryOp::Mul, n, recip)
                            })
                        }
                    };
                }
            }
        }

        for &o in func.outputs() {
            let out = match tan[o] {
                Some(t) => t,
                None => b.constant(0.0),
            };
            outputs.push(out);
        }
    }

    let raw = Function::from_parts(b.nodes, func.inputs().to_vec(), outputs);
    prune(&raw)
}

// ---------------------------------------------------------------------------
// Reverse mode (adjoint): wᵀ · J
// ---------------------------------------------------------------------------

/// Reverse-mode transform. For each weight vector `w ∈ weights` (one entry
/// per function output), the returned function computes `wᵀ · J`, i.e. the
/// adjoint of every input.
///
/// Output layout is weight-major:
/// `out[d * n_in + j]` = ∂(wᵀf)/∂input_j for `weights[d]`.
///
/// For compressed (row-colored) Jacobians, pass one 0/1 weight vector per
/// color. For a plain gradient, use [`gradient`].
pub fn reverse(func: &Function, weights: &[&[f64]]) -> Function {
    let n_out = func.outputs().len();
    let n_primal = func.nodes().len();
    for (d, w) in weights.iter().enumerate() {
        assert_eq!(
            w.len(),
            n_out,
            "reverse: weight {d} has length {}, expected {n_out} (one per output)",
            w.len()
        );
    }

    let mut b = Builder::from_primal(func);
    let mut helpers = Helpers::new(n_primal);
    let mut outputs = Vec::with_capacity(weights.len() * func.inputs().len());

    for weight in weights {
        // Adjoint of every primal node; None = structural zero.
        let mut adj: IndexVec<NodeId, Option<NodeId>> = index_vec![None; n_primal];

        // Seed the outputs. `acc` handles the same node appearing as several
        // outputs (its adjoints simply add up).
        for (k, &o) in func.outputs().iter().enumerate() {
            if weight[k] != 0.0 {
                let c = b.constant(weight[k]);
                adj[o] = b.acc(adj[o], c);
            }
        }

        // Sweep parents-before-children. Because the tape is in SSA order
        // (operands have smaller ids), by the time we visit a node every
        // consumer has already deposited its contribution, so `adj[id]` is
        // final here.
        for i in (0..n_primal).rev() {
            let id = NodeId::from_usize(i);
            let Some(wbar) = adj[id] else { continue };

            match func.nodes()[id] {
                Node::Param(_) | Node::Constant(_) => {}

                Node::Unary(op, a) => match op {
                    // adj[x] -= w
                    UnaryOp::Neg => {
                        let cur = adj[a];
                        adj[a] = b.acc_neg(cur, wbar);
                    }
                    // adj[x] += w · cos(x)
                    UnaryOp::Sin => {
                        let c = helpers.get(&mut b, id, |b| b.unary(UnaryOp::Cos, a));
                        let t = b.binary(BinaryOp::Mul, wbar, c);
                        let cur = adj[a];
                        adj[a] = b.acc(cur, t);
                    }
                    // adj[x] -= w · sin(x)
                    UnaryOp::Cos => {
                        let s = helpers.get(&mut b, id, |b| b.unary(UnaryOp::Sin, a));
                        let t = b.binary(BinaryOp::Mul, wbar, s);
                        let cur = adj[a];
                        adj[a] = b.acc_neg(cur, t);
                    }
                },

                Node::Binary(op, l, r) => match op {
                    BinaryOp::Add => {
                        let cur = adj[l];
                        adj[l] = b.acc(cur, wbar);
                        let cur = adj[r];
                        adj[r] = b.acc(cur, wbar);
                    }
                    BinaryOp::Sub => {
                        let cur = adj[l];
                        adj[l] = b.acc(cur, wbar);
                        let cur = adj[r];
                        adj[r] = b.acc_neg(cur, wbar);
                    }
                    // adj[l] += w·r ; adj[r] += w·l
                    // (l == r, i.e. x·x, works: both accumulations land.)
                    BinaryOp::Mul => {
                        let t = b.binary(BinaryOp::Mul, wbar, r);
                        let cur = adj[l];
                        adj[l] = b.acc(cur, t);
                        let t = b.binary(BinaryOp::Mul, wbar, l);
                        let cur = adj[r];
                        adj[r] = b.acc(cur, t);
                    }
                    // y = l/r:  adj[l] += w/r ; adj[r] -= (w/r)·y
                    // `id` IS the primal y; 1/r is a shared helper.
                    BinaryOp::Div => {
                        let recip = helpers.get(&mut b, id, |b| {
                            let one = b.constant(1.0);
                            b.binary(BinaryOp::Div, one, r)
                        });
                        let w_over_r = b.binary(BinaryOp::Mul, wbar, recip);
                        let cur = adj[l];
                        adj[l] = b.acc(cur, w_over_r);
                        let t = b.binary(BinaryOp::Mul, w_over_r, id);
                        let cur = adj[r];
                        adj[r] = b.acc_neg(cur, t);
                    }
                },
            }
        }

        for &p in func.inputs() {
            let out = match adj[p] {
                Some(a) => a,
                None => b.constant(0.0),
            };
            outputs.push(out);
        }
    }

    let raw = Function::from_parts(b.nodes, func.inputs().to_vec(), outputs);
    prune(&raw)
}

/// Gradient of a scalar-valued function: a single reverse sweep with unit
/// weight. No coloring involved — do not route gradients through the
/// Jacobian machinery.
pub fn gradient(func: &Function) -> Function {
    assert_eq!(
        func.outputs().len(),
        1,
        "gradient: function must have exactly one output, got {}",
        func.outputs().len()
    );
    reverse(func, &[&[1.0]])
}

// ---------------------------------------------------------------------------
// Dead code elimination
// ---------------------------------------------------------------------------

/// Drop every node not reachable from the outputs, remapping `NodeId`s.
///
/// Input `Param` nodes are always kept (even if an output doesn't depend on
/// them) so the function's arity and input list survive intact.
pub fn prune(func: &Function) -> Function {
    let n = func.nodes().len();

    let mut live: IndexVec<NodeId, bool> = index_vec![false; n];
    for &o in func.outputs() {
        live[o] = true;
    }
    for &i in func.inputs() {
        live[i] = true;
    }
    // SSA order ⇒ a reverse sweep propagates liveness to all operands.
    for i in (0..n).rev() {
        let id = NodeId::from_usize(i);
        if !live[id] {
            continue;
        }
        match func.nodes()[id] {
            Node::Unary(_, a) => live[a] = true,
            Node::Binary(_, a, b) => {
                live[a] = true;
                live[b] = true;
            }
            Node::Param(_) | Node::Constant(_) => {}
        }
    }

    let mut remap: IndexVec<NodeId, Option<NodeId>> = index_vec![None; n];
    let mut nodes: IndexVec<NodeId, Node> = IndexVec::new();
    for (id, node) in func.nodes().iter_enumerated() {
        if !live[id] {
            continue;
        }
        let new_node = match *node {
            Node::Unary(op, a) => Node::Unary(op, remap[a].expect("operand pruned")),
            Node::Binary(op, a, b) => Node::Binary(
                op,
                remap[a].expect("operand pruned"),
                remap[b].expect("operand pruned"),
            ),
            other => other,
        };
        remap[id] = Some(nodes.push(new_node));
    }

    let inputs = func
        .inputs()
        .iter()
        .map(|&i| remap[i].expect("input pruned"))
        .collect();
    let outputs = func
        .outputs()
        .iter()
        .map(|&o| remap[o].expect("output pruned"))
        .collect();

    Function::from_parts(nodes, inputs, outputs)
}

// ---------------------------------------------------------------------------
// Tests (structural — numeric round-trips belong with Function::eval)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x, y) = x * y
    fn product() -> Function {
        let mut nodes: IndexVec<NodeId, Node> = IndexVec::new();
        let x = nodes.push(Node::Param(0));
        let y = nodes.push(Node::Param(1));
        let m = nodes.push(Node::Binary(BinaryOp::Mul, x, y));
        Function::from_parts(nodes, vec![x, y], vec![m])
    }

    #[test]
    fn forward_unit_seed_folds_to_operand() {
        // ∂(x·y)/∂x = y; the 1·y from the unit seed must fold to y itself.
        let f = product();
        let df = forward(&f, &[&[1.0, 0.0]]);
        assert_eq!(df.outputs().len(), 1);
        assert_eq!(df.outputs()[0], df.inputs()[1]);
    }

    #[test]
    fn forward_zero_seed_is_constant_zero() {
        let f = product();
        let df = forward(&f, &[&[0.0, 0.0]]);
        assert!(matches!(df.nodes()[df.outputs()[0]], Node::Constant(_)));
    }

    #[test]
    fn forward_multiple_seeds_layout() {
        let f = product();
        let df = forward(&f, &[&[1.0, 0.0], &[0.0, 1.0]]);
        // direction-major: [d0·out0, d1·out0]
        assert_eq!(df.outputs().len(), 2);
        assert_eq!(df.outputs()[0], df.inputs()[1]); // ∂f/∂x = y
        assert_eq!(df.outputs()[1], df.inputs()[0]); // ∂f/∂y = x
    }

    #[test]
    fn gradient_of_product() {
        let f = product();
        let g = gradient(&f);
        assert_eq!(g.outputs().len(), 2);
        assert_eq!(g.outputs()[0], g.inputs()[1]); // ∂f/∂x = y
        assert_eq!(g.outputs()[1], g.inputs()[0]); // ∂f/∂y = x
    }

    #[test]
    fn prune_drops_dead_primal() {
        // Gradient of x·y needs no Mul node at all after folding + DCE:
        // only the two params should survive.
        let f = product();
        let g = gradient(&f);
        assert_eq!(g.nodes().len(), 2);
    }
}
