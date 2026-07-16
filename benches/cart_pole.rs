use autodiff::tape::Var;
use solver::{NlpSolver, constraint};

/// State: x = [x_cart, θ, ẋ, θ̇]ᵀ, control: u = f_x
type State<'t> = [Var<'t>; 4];

fn cart_pole_dynamics<'t>(x: &State<'t>, u: Var<'t>) -> State<'t> {
    // https://underactuated.mit.edu/acrobot.html#cart_pole
    //
    // θ is CCW+ measured from negative y-axis.
    //
    // q = [x, θ]ᵀ
    // q̇ = [ẋ, θ̇]ᵀ
    // u = f_x
    //
    // M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
    // M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
    // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    //
    //        [ m_c + m_p  m_p l cosθ]
    // M(q) = [m_p l cosθ    m_p l²  ]
    //
    //           [0  −m_p lθ̇ sinθ]
    // C(q, q̇) = [0       0      ]
    //
    //          [     0      ]
    // τ_g(q) = [−m_p gl sinθ]
    //
    //     [1]
    // B = [0]
    const M_C: f64 = 5.0; // Cart mass (kg)
    const M_P: f64 = 0.5; // Pole mass (kg)
    const L: f64 = 0.5; // Pole length (m)
    const G: f64 = 9.806; // Acceleration due to gravity (m/s²)

    let theta = x[1];
    let xdot = x[2];
    let thetadot = x[3];

    let sin_t = theta.sin();
    let cos_t = theta.cos();

    //        [ m_c + m_p  m_p l cosθ]
    // M(q) = [m_p l cosθ    m_p l²  ]
    let m00 = M_C + M_P;
    let m01 = M_P * L * cos_t;
    let m11 = M_P * L * L;

    // RHS = τ_g(q) − C(q, q̇)q̇ + Bu
    let rhs0 = M_P * L * thetadot * thetadot * sin_t + u;
    let rhs1 = -(M_P * G * L) * sin_t;

    // q̈ = M⁻¹ rhs via Cramer's rule on the symmetric 2×2:
    // det = m00·m11 − m01²
    let det = m00 * m11 - m01 * m01;
    let qddot0 = (m11 * rhs0 - m01 * rhs1) / det;
    let qddot1 = (m00 * rhs1 - m01 * rhs0) / det;

    [xdot, thetadot, qddot0, qddot1]
}

fn rk4<'t, const S: usize>(
    f: impl Fn(&[Var<'t>; S], Var<'t>) -> [Var<'t>; S],
    x: &[Var<'t>; S],
    u: Var<'t>,
    dt: f64,
) -> [Var<'t>; S] {
    let h = dt;

    let k1 = f(x, u);
    let x2: [Var; S] = std::array::from_fn(|i| x[i] + k1[i] * (h / 2.0));
    let k2 = f(&x2, u);
    let x3: [Var; S] = std::array::from_fn(|i| x[i] + k2[i] * (h / 2.0));
    let k3 = f(&x3, u);
    let x4: [Var; S] = std::array::from_fn(|i| x[i] + k3[i] * h);
    let k4 = f(&x4, u);

    std::array::from_fn(|i| x[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (h / 6.0))
}

pub fn cart_pole(dt: f64, n: usize) -> NlpSolver {
    const U_MAX: f64 = 20.0; // N
    const D_MAX: f64 = 2.0; // m

    const X_INITIAL: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    const X_FINAL: [f64; 4] = [1.0, std::f64::consts::PI, 0.0, 0.0];

    let solver = NlpSolver::new();

    // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
    let x: Vec<State> = (0..=n)
        .map(|_| std::array::from_fn(|_| solver.decision_variable()))
        .collect();

    // u = f_x
    let u: Vec<Var> = (0..n).map(|_| solver.decision_variable()).collect();

    // Initial guess: lerp cart position and pole angle along the horizon
    for k in 0..=n {
        let t = k as f64 / n as f64;
        solver.set_initial(x[k][0], X_INITIAL[0] + t * (X_FINAL[0] - X_INITIAL[0]));
        solver.set_initial(x[k][1], X_INITIAL[1] + t * (X_FINAL[1] - X_INITIAL[1]));
    }

    // Initial and final conditions
    for i in 0..4 {
        solver.subject_to(constraint!(x[0][i] == X_INITIAL[i]));
        solver.subject_to(constraint!(x[n][i] == X_FINAL[i]));
    }

    // Cart position constraints
    for k in 0..=n {
        solver.subject_to(constraint!(0.0 <= x[k][0] <= D_MAX));
    }

    // Input constraints
    for k in 0..n {
        solver.subject_to(constraint!(-U_MAX <= u[k] <= U_MAX));
    }

    // Dynamics constraints - RK4 integration
    for k in 0..n {
        let x_next = rk4(cart_pole_dynamics, &x[k], u[k], dt);
        for i in 0..4 {
            solver.subject_to(constraint!(x[k + 1][i] == x_next[i]));
        }
    }

    // Minimize sum squared inputs
    let mut j = solver.constant(0.0);
    for k in 0..n {
        j = j + u[k] * u[k];
    }
    solver.minimize(j);

    solver
}

pub fn main() {}
