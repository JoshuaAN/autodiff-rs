use std::cell::RefCell;
use std::collections::HashMap;

use autodiff::node::NodeId;
use autodiff::tape::{Tape, Var};

#[derive(Default)]
struct SolverState {
    decision_variables: Vec<NodeId>,
    parameters: Vec<NodeId>,
    parameter_values: Vec<f64>,
    objective: Option<NodeId>,
    constraints: Vec<Constraint>,
    initial: HashMap<NodeId, f64>,
}

pub struct NlpSolver {
    tape: Tape,
    state: RefCell<SolverState>,
}

impl Default for NlpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl NlpSolver {
    pub fn new() -> Self {
        NlpSolver {
            tape: Tape::new(),
            state: RefCell::new(SolverState::default()),
        }
    }

    pub fn decision_variable(&self) -> Var<'_> {
        let v = self.tape.param();
        self.state.borrow_mut().decision_variables.push(v.id());
        v
    }

    pub fn constant(&self, x: f64) -> Var<'_> {
        self.tape.constant(x)
    }

    pub fn minimize(&self, f: Var<'_>) {
        self.state.borrow_mut().objective = Some(f.id());
    }

    pub fn maximize(&self, f: Var<'_>) {
        self.minimize(-f);
    }

    pub fn set_initial(&self, v: Var<'_>, x: f64) {
        self.state.borrow_mut().initial.insert(v.id(), x);
    }

    pub fn subject_to(&self, c: Constraint) {
        self.state.borrow_mut().constraints.push(c);
    }

    pub fn solve(self) -> Result<Solution, SolveError> {
        todo!("implement solve")
    }
}

pub enum Constraint {
    Eq(NodeId, NodeId),
    Le(NodeId, NodeId),
    Bounded(NodeId, NodeId, NodeId),
}

pub trait Operand<'t> {
    fn to_node(&self, tape: &'t Tape) -> NodeId;
    fn tape(&self) -> Option<&'t Tape>;
}

impl<'t> Operand<'t> for Var<'t> {
    fn to_node(&self, _tape: &'t Tape) -> NodeId {
        self.id()
    }
    fn tape(&self) -> Option<&'t Tape> {
        Some(self.tape())
    }
}

impl<'t> Operand<'t> for f64 {
    fn to_node(&self, tape: &'t Tape) -> NodeId {
        tape.constant(*self).id()
    }
    fn tape(&self) -> Option<&'t Tape> {
        None
    }
}

pub mod cns {
    use super::*;

    fn resolve<'t>(a: &impl Operand<'t>, b: &impl Operand<'t>) -> &'t Tape {
        a.tape()
            .or_else(|| b.tape())
            .expect("constraint requires at least one Var operand")
    }

    pub fn le<'t>(lhs: impl Operand<'t>, rhs: impl Operand<'t>) -> Constraint {
        let t = resolve(&lhs, &rhs);
        Constraint::Le(lhs.to_node(t), rhs.to_node(t))
    }

    pub fn ge<'t>(lhs: impl Operand<'t>, rhs: impl Operand<'t>) -> Constraint {
        le(rhs, lhs) // normalize: a >= b  ⇒  b <= a
    }

    pub fn eq<'t>(lhs: impl Operand<'t>, rhs: impl Operand<'t>) -> Constraint {
        let t = resolve(&lhs, &rhs);
        Constraint::Eq(lhs.to_node(t), rhs.to_node(t))
    }

    pub fn bounded<'t>(lb: impl Operand<'t>, expr: Var<'t>, ub: impl Operand<'t>) -> Constraint {
        let t = expr.tape();
        Constraint::Bounded(lb.to_node(t), expr.id(), ub.to_node(t))
    }
}

#[macro_export]
macro_rules! constraint {
    ($($t:tt)+) => { $crate::__cns!(@scan [] $($t)+) };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __cns {
    (@scan [$($l:tt)+] <= $($r:tt)+) => { $crate::__cns!(@le [$($l)+] [] $($r)+) };
    (@scan [$($l:tt)+] >= $($r:tt)+) => { $crate::__cns!(@ge [$($l)+] [] $($r)+) };
    (@scan [$($l:tt)+] == $($r:tt)+) => { $crate::cns::eq(($($l)+), ($($r)+)) };
    (@scan [$($l:tt)*] $t:tt $($rest:tt)*) => { $crate::__cns!(@scan [$($l)* $t] $($rest)*) };
    (@scan [$($l:tt)+]) => { compile_error!("constraint! requires <=, >=, or ==") };

    (@le [$($l:tt)+] [$($m:tt)+] <= $($u:tt)+) => {
        $crate::cns::bounded(($($l)+), ($($m)+), ($($u)+))
    };
    (@le [$($l:tt)+] [$($m:tt)*] $t:tt $($rest:tt)*) => {
        $crate::__cns!(@le [$($l)+] [$($m)* $t] $($rest)*)
    };
    (@le [$($l:tt)+] [$($m:tt)+]) => { $crate::cns::le(($($l)+), ($($m)+)) };

    (@ge [$($l:tt)+] [$($m:tt)+] >= $($u:tt)+) => {
        $crate::cns::bounded(($($u)+), ($($m)+), ($($l)+))
    };
    (@ge [$($l:tt)+] [$($m:tt)*] $t:tt $($rest:tt)*) => {
        $crate::__cns!(@ge [$($l)+] [$($m)* $t] $($rest)*)
    };
    (@ge [$($l:tt)+] [$($m:tt)+]) => { $crate::cns::ge(($($l)+), ($($m)+)) };
}

#[derive(Debug)]
pub enum SolveError {
    NoObjective,
    Constraint {
        index: usize,
        kind: ConstraintErrorKind,
    },
}

#[derive(Debug)]
pub enum ConstraintErrorKind {
    /// No decision variables reachable from the constraint.
    NoDecisionVariables,
    /// Simplified to something unexpected after lowering.
    Degenerate,
    /// Constant bounds that can never hold (e.g. lb > ub, or 1 <= 0).
    InfeasibleConstant,
}

pub struct Solution {
    pub x: Vec<f64>,
    var_index: HashMap<NodeId, usize>,
    success: bool,
}

impl Solution {
    pub fn is_success(&self) -> bool {
        self.success
    }

    pub fn value(&self, v: NodeId) -> f64 {
        self.x[*self
            .var_index
            .get(&v)
            .expect("value(): not a decision variable of this problem")]
    }
}
