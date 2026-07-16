
use autodiff::tape::{Tape, Var};
use autodiff::node::{NodeId};

pub struct NlpSolver {
  tape: Tape,
  decision_variables: Vec<NodeId>,
  f: Option<NodeId>,
  equality_constraints: Vec<NodeId>,
  inequality_constraints: Vec<NodeId>,
}

impl NlpSolver {
  pub fn new() -> Self {
    NlpSolver { 
      tape: Tape::new(),
      decision_variables: Vec::new(), 
      f: None,
      equality_constraints: Vec::new(),
      inequality_constraints: Vec::new(),
    }
  }

  pub fn decision_variable(&mut self) -> Var<'_> {
    let v = self.tape.param();
    self.decision_variables.push(v.id());
    v
  }

  pub fn minimize(&mut self, f: Var<'_>) {
    self.f = Some(f.id())
  }

  pub fn maximize(&mut self, f: Var<'_>) {
    self.f = Some((-f).id())
  }

  pub fn subject_to()
}