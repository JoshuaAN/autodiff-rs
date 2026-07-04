struct Problem {
    // The list of decision variables, which are the root of the problem's
    // expression tree
    decision_variables: Vec<>,

    // The cost function: f(x)
    f: Option<>,

    // The list of equality constraints: cₑ(x) = 0
    equality_constraints: Vec<>,

    // The list of inequality constraints: cᵢ(x) ≥ 0
    inequality_constraints: Vec<>,
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

    pub fn minimize() {}

    pub fn maximize() {}

    pub fn subject_to() {}

    pub fn solve() {}
}
