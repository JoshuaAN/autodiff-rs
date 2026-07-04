use std::{cell::RefCell, collections::HashMap, ops, rc::Rc};

use index_vec::IndexVec;

use crate::{
    node::{Node, NodeId},
    op::{BinaryOp, UnaryOp},
    tape::{Instr, Slot, Tape},
    var::VarId,
};

/// TODO: rewrite/check later, this file is currently claude generaed

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

struct ContextInner {
    nodes: IndexVec<NodeId, Node>,
    n_vars: u32,
}

impl ContextInner {
    fn insert(&mut self, node: Node) -> NodeId {
       self.nodes.push(node)
    }

    fn lower(&self, roots: &[NodeId]) -> Tape {
        // Pass 1: mark reachable nodes, high -> low.
        let mut live: IndexVec<NodeId, bool> =
            index_vec::index_vec![false; self.nodes.len()];
        for r in roots {
            live[*r] = true;
        }
        for id in self.nodes.indices().rev() {
            if !live[id] {
                continue;
            }
            match self.nodes[id] {
                Node::Unary(_, a) => live[a] = true,
                Node::Binary(_, a, b) => {
                    live[a] = true;
                    live[b] = true;
                }
                Node::Constant(_) | Node::Variable(_) => {}
            }
        }

        // Pass 2: emit live nodes, low -> high.
        let mut slot_of: IndexVec<NodeId, Option<Slot>> =
            index_vec::index_vec![None; self.nodes.len()];
        let mut insts: IndexVec<Slot, Instr> = IndexVec::new();

        for (id, &node) in self.nodes.iter_enumerated() {
            if !live[id] {
                continue;
            }
            let inst = match node {
                Node::Constant(bits) => Instr::Const((bits)),
                Node::Variable(v) => Instr::Input(v),
                Node::Unary(op, a) => Instr::Unary(op, slot_of[a].unwrap()),
                Node::Binary(op, a, b) => {
                    Instr::Binary(op, slot_of[a].unwrap(), slot_of[b].unwrap())
                }
            };
            slot_of[id] = Some(insts.push(inst));
        }

        Tape {
            outputs: roots.iter().map(|r| slot_of[*r].unwrap()).collect(),
            insts,
        }
    }
}

// ---------------------------------------------------------------------------
// Public handles
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Context(Rc<RefCell<ContextInner>>);

#[derive(Clone)]
pub struct Expr {
    id: NodeId,
    ctx: Context,
}

impl Context {
    pub fn new() -> Self {
        Context(Rc::new(RefCell::new(ContextInner {
            nodes: IndexVec::new(),
            n_vars: 0,
        })))
    }

    pub fn var(&self) -> Expr {
        let mut inner = self.0.borrow_mut();
        let v = VarId::from_raw(inner.n_vars);
        inner.n_vars += 1;
        let id = inner.insert(Node::Variable(v));
        Expr { id, ctx: self.clone() }
    }

    pub fn constant(&self, v: f64) -> Expr {
        let id = self.0.borrow_mut().insert(Node::Constant(v));
        Expr { id, ctx: self.clone() }
    }

    pub fn n_vars(&self) -> u32 {
        self.0.borrow().n_vars
    }

    pub fn lower(&self, roots: &[&Expr]) -> Tape {
        let ids: Vec<NodeId> = roots
            .iter()
            .map(|r| {
                assert!(
                    Rc::ptr_eq(&self.0, &r.ctx.0),
                    "Expr belongs to a different Context"
                );
                r.id
            })
            .collect();
        self.0.borrow().lower(&ids)
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Expr construction helpers
// ---------------------------------------------------------------------------

impl Expr {
    fn unary(op: UnaryOp, a: &Expr) -> Expr {
        let id = a.ctx.0.borrow_mut().insert(Node::Unary(op, a.id));
        Expr { id, ctx: a.ctx.clone() }
    }

    fn binary(op: BinaryOp, a: &Expr, b: &Expr) -> Expr {
        assert!(
            Rc::ptr_eq(&a.ctx.0, &b.ctx.0),
            "cannot combine Exprs from different Contexts"
        );
        let id = a.ctx.0.borrow_mut().insert(Node::Binary(op, a.id, b.id));
        Expr { id, ctx: a.ctx.clone() }
    }

    /// Intern a constant in this Expr's context (for f64-mixed operators).
    fn lift(&self, v: f64) -> Expr {
        self.ctx.constant(v)
    }

    // ---- named math methods ----

    pub fn sqrt(&self) -> Expr { Expr::unary(UnaryOp::Sqrt, self) }
    pub fn exp(&self)  -> Expr { Expr::unary(UnaryOp::Exp,  self) }
    pub fn ln(&self)   -> Expr { Expr::unary(UnaryOp::Ln,   self) }
    pub fn sin(&self)  -> Expr { Expr::unary(UnaryOp::Sin,  self) }
    pub fn cos(&self)  -> Expr { Expr::unary(UnaryOp::Cos,  self) }
    pub fn abs(&self)  -> Expr { Expr::unary(UnaryOp::Abs,  self) }

    pub fn min(&self, other: &Expr) -> Expr { Expr::binary(BinaryOp::Min, self, other) }
    pub fn max(&self, other: &Expr) -> Expr { Expr::binary(BinaryOp::Max, self, other) }
    pub fn powf(&self, e: &Expr)    -> Expr { Expr::binary(BinaryOp::Pow, self, e) }

    /// Integer power by squaring; hash-consing dedups the intermediates.
    pub fn powi(&self, n: i32) -> Expr {
        match n {
            0 => self.lift(1.0),
            1 => self.clone(),
            n if n < 0 => 1.0 / self.powi(-n),
            n => {
                let half = self.powi(n / 2);
                let sq = &half * &half;
                if n % 2 == 0 { sq } else { &sq * self }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operator overloads
// ---------------------------------------------------------------------------

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:expr) => {
        impl ops::$trait<&Expr> for &Expr {
            type Output = Expr;
            fn $method(self, rhs: &Expr) -> Expr { Expr::binary($op, self, rhs) }
        }
        impl ops::$trait<Expr> for Expr {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr { Expr::binary($op, &self, &rhs) }
        }
        impl ops::$trait<&Expr> for Expr {
            type Output = Expr;
            fn $method(self, rhs: &Expr) -> Expr { Expr::binary($op, &self, rhs) }
        }
        impl ops::$trait<Expr> for &Expr {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr { Expr::binary($op, self, &rhs) }
        }
        impl ops::$trait<f64> for &Expr {
            type Output = Expr;
            fn $method(self, rhs: f64) -> Expr {
                let c = self.lift(rhs);
                Expr::binary($op, self, &c)
            }
        }
        impl ops::$trait<f64> for Expr {
            type Output = Expr;
            fn $method(self, rhs: f64) -> Expr { (&self).$method(rhs) }
        }
        impl ops::$trait<&Expr> for f64 {
            type Output = Expr;
            fn $method(self, rhs: &Expr) -> Expr {
                let c = rhs.lift(self);
                Expr::binary($op, &c, rhs)
            }
        }
        impl ops::$trait<Expr> for f64 {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr { self.$method(&rhs) }
        }
    };
}

impl_binop!(Add, add, BinaryOp::Add);
impl_binop!(Sub, sub, BinaryOp::Sub);
impl_binop!(Mul, mul, BinaryOp::Mul);
impl_binop!(Div, div, BinaryOp::Div);
impl_binop!(Rem, rem, BinaryOp::Mod);

impl ops::Neg for &Expr {
    type Output = Expr;
    fn neg(self) -> Expr { Expr::unary(UnaryOp::Neg, self) }
}
impl ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr { -&self }
}