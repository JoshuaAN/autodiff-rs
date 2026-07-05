use autodiff::{expression::{Expression, Node}, var::VarId};

pub struct Problem {
    // The list of decision variables, which are the root of the problem's
    // expression tree
    decision_variables: Vec<VarId>,

    // The cost function: f(x)
    f: Option<Expression>,

    // The list of equality constraints: cₑ(x) = 0
    equality_constraints: Vec<Expression>,

    // The list of inequality constraints: cᵢ(x) ≥ 0
    inequality_constraints: Vec<Expression>,
}

impl Problem {
    pub fn new() -> Self {
        Self {
            decision_variables: Vec::new(),
            f: None,
            equality_constraints: Vec::new(),
            inequality_constraints: Vec::new(),
        }
    }

    pub fn decision_variable(&mut self) -> Expression {
        Expression::variable(VarId::from(self.decision_variables.len()))
    }

    pub fn minimize(&mut self, f: Expression) {
        self.f = Some(f);
    }

    pub fn maximize(&mut self, f: Expression) {
        self.f = Some(-f);
    }

    pub fn subject_to(&mut self, c: Constraint) {
        match c.order {
            Order::Geq => {
                self.inequality_constraints.push(c.rhs - c.lhs);
            }
            Order::Leq => {
                self.inequality_constraints.push(c.lhs - c.rhs);
            }
            Order::Eq => {
                self.equality_constraints.push(c.lhs - c.rhs);
            }
        }
    }

    pub fn solve(&mut self) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Geq,
    Leq,
    Eq,
}

pub struct Constraint {
    lhs: Expression,
    order: Order,
    rhs: Expression,
}

impl Constraint {
    pub fn geq(lhs: impl Into<Expression>, rhs: impl Into<Expression>) -> Constraint {
        Constraint { lhs: lhs.into(), order: Order::Geq, rhs: rhs.into() }
    }
    pub fn leq(lhs: impl Into<Expression>, rhs: impl Into<Expression>) -> Constraint {
        Constraint { lhs: lhs.into(), order: Order::Leq, rhs: rhs.into() }
    }
    pub fn eq(lhs: impl Into<Expression>, rhs: impl Into<Expression>) -> Constraint {
        Constraint { lhs: lhs.into(), order: Order::Eq, rhs: rhs.into() }
    }
}

#[macro_export]
macro_rules! constraint {
    (@[$($lhs:tt)*] >= $($rhs:tt)*) => {
        $crate::problem::Constraint::geq(($($lhs)*), ($($rhs)*))
    };
    (@[$($lhs:tt)*] <= $($rhs:tt)*) => {
        $crate::problem::Constraint::leq(($($lhs)*), ($($rhs)*))
    };
    (@[$($lhs:tt)*] == $($rhs:tt)*) => {
        $crate::problem::Constraint::eq(($($lhs)*), ($($rhs)*))
    };
    (@[$($lhs:tt)*] $next:tt $($rest:tt)*) => {
        $crate::constraint!(@[$($lhs)* $next] $($rest)*)
    };
    ($($all:tt)*) => {
        $crate::constraint!(@[] $($all)*)
    };
}
