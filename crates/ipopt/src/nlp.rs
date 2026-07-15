use std::ffi::{CString, c_char};

use ipopt_sys::{self as ffi, ApplicationReturnStatus, UserDataPtr};
use ipopt_sys::ipindex;

use crate::bridge::{Callbacks, ConFn, GradFn, HessFn, JacFn, ObjFn, tramp_f, tramp_g, tramp_grad_f, tramp_h, tramp_jac_g};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    SolveSucceeded,
    SolvedToAcceptableLevel,
    InfeasibleProblemDetected,
    SearchDirectionBecomesTooSmall,
    DivergingIterates,
    UserRequestedStop,
    FeasiblePointFound,
    MaximumIterationsExceeded,
    RestorationFailed,
    ErrorInStepComputation,
    MaximumCpuTimeExceeded,
    MaximumWallTimeExceeded,
    NotEnoughDegreesOfFreedom,
    InvalidProblemDefinition,
    InvalidOption,
    InvalidNumberDetected,
    UnrecoverableException,
    NonIpoptExceptionThrown,
    InsufficientMemory,
    InternalError,
}

impl From<ApplicationReturnStatus> for Status {
    fn from(s: ApplicationReturnStatus) -> Self {
        use ApplicationReturnStatus as R;
        match s {
            R::Solve_Succeeded => Status::SolveSucceeded,
            R::Solved_To_Acceptable_Level => Status::SolvedToAcceptableLevel,
            R::Infeasible_Problem_Detected => Status::InfeasibleProblemDetected,
            R::Search_Direction_Becomes_Too_Small => Status::SearchDirectionBecomesTooSmall,
            R::Diverging_Iterates => Status::DivergingIterates,
            R::User_Requested_Stop => Status::UserRequestedStop,
            R::Feasible_Point_Found => Status::FeasiblePointFound,
            R::Maximum_Iterations_Exceeded => Status::MaximumIterationsExceeded,
            R::Restoration_Failed => Status::RestorationFailed,
            R::Error_In_Step_Computation => Status::ErrorInStepComputation,
            R::Maximum_CpuTime_Exceeded => Status::MaximumCpuTimeExceeded,
            R::Maximum_WallTime_Exceeded => Status::MaximumWallTimeExceeded,
            R::Not_Enough_Degrees_Of_Freedom => Status::NotEnoughDegreesOfFreedom,
            R::Invalid_Problem_Definition => Status::InvalidProblemDefinition,
            R::Invalid_Option => Status::InvalidOption,
            R::Invalid_Number_Detected => Status::InvalidNumberDetected,
            R::Unrecoverable_Exception => Status::UnrecoverableException,
            R::NonIpopt_Exception_Thrown => Status::NonIpoptExceptionThrown,
            R::Insufficient_Memory => Status::InsufficientMemory,
            R::Internal_Error => Status::InternalError,
        }
    }
}

pub struct Solution {
    pub status: Status,

    /// Optimal point (or last iterate if solve failure).
    pub x: Vec<f64>,

    /// Final objective value.
    pub obj: f64,

    /// Final constraint values.
    pub g: Vec<f64>,

    /// Final constraint multipliers.
    pub mult_g: Vec<f64>,

    /// Final lower bound multipliers.
    pub mult_x_l: Vec<f64>,

    /// Final upper bound multipliers.
    pub mult_x_u: Vec<f64>,
}

impl Solution {
    pub fn is_success(&self) -> bool {
        matches!(
            self.status,
            Status::SolveSucceeded | Status::SolvedToAcceptableLevel
        )
    }
}

#[derive(Debug)]
pub enum Error {
    /// Required callback never set.
    Missing(&'static str),

    /// Bad lengths or sparsity out of range.
    Invalid(String),

    /// CreateIpoptProblem failed.
    CreateFailed,

    /// IPOPT rejected an option
    BadOption(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(what) => write!(f, "missing required callback: {what}"),
            Error::Invalid(msg) => write!(f, "invalid problem definition: {msg}"),
            Error::CreateFailed => write!(f, "CreateIpoptProblem failed"),
            Error::BadOption(key) => write!(f, "IPOPT rejected option '{key}'"),
        }
    }
}
impl std::error::Error for Error {}

enum OptVal {
    Num(f64),
    Int(i32),
    Str(String),
}

#[derive(Default)]
pub struct Nlp {
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    obj: Option<ObjFn>,
    grad: Option<GradFn>,
    con: Option<ConFn>,
    jac: Option<(Vec<(usize, usize)>, JacFn)>,
    hess: Option<(Vec<(usize, usize)>, HessFn)>,
    options: Vec<(String, OptVal)>,
}

impl Nlp {
    pub fn new(num_vars: usize) -> Self {
        Nlp {
            n: num_vars,
            // IPOPT interprets |bound| >= 1e19 as infinity
            x_l: vec![-1e19; num_vars],
            x_u: vec![1e19; num_vars],
            ..Default::default()
        }
    }

    pub fn bounds(mut self, x_l: Vec<f64>, x_u: Vec<f64>) -> Self {
        self.x_l = x_l;
        self.x_u = x_u;
        self
    }

    pub fn constraint_bounds(mut self, g_l: Vec<f64>, g_u: Vec<f64>) -> Self {
        self.m = g_l.len();
        self.g_l = g_l;
        self.g_u = g_u;
        self
    }

    pub fn objective(mut self, f: impl FnMut(&[f64]) -> f64 + 'static) -> Self {
        self.obj = Some(Box::new(f));
        self
    }

    pub fn gradient(mut self, grad: impl FnMut(&[f64], &mut [f64]) + 'static) -> Self {
        self.grad = Some(Box::new(grad));
        self
    }

    pub fn constraints(mut self, con: impl FnMut(&[f64], &mut [f64]) + 'static) -> Self {
        self.con = Some(Box::new(con));
        self
    }

    pub fn jacobian(
        mut self,
        sparsity: Vec<(usize, usize)>,
        f: impl FnMut(&[f64], &mut [f64]) + 'static,
    ) -> Self {
        self.jac = Some((sparsity, Box::new(f)));
        self
    }

    pub fn hessian(
        mut self,
        sparsity: Vec<(usize, usize)>,
        f: impl FnMut(&[f64], f64, &[f64], &mut [f64]) + 'static,
    ) -> Self {
        self.hess = Some((sparsity, Box::new(f)));
        self
    }

    pub fn num_option(mut self, key: &str, val: f64) -> Self {
        self.options.push((key.into(), OptVal::Num(val)));
        self
    }

    pub fn int_option(mut self, key: &str, val: i32) -> Self {
        self.options.push((key.into(), OptVal::Int(val)));
        self
    }

    pub fn str_option(mut self, key: &str, val: &str) -> Self {
        self.options.push((key.into(), OptVal::Str(val.into())));
        self
    }

    pub fn solve(self, x0: &[f64]) -> Result<Solution, Error> {
        let (n, m) = (self.n, self.m);

        // Validate parameter lengths
        if x0.len() != n {
            return Err(Error::Invalid(format!(
                "x0 has length {}, expected {n}",
                x0.len()
            )));
        }
        if self.x_l.len() != n || self.x_u.len() != n {
            return Err(Error::Invalid("variable bounds must have length n".into()));
        }
        if self.g_u.len() != m {
            return Err(Error::Invalid("g_l and g_u must have equal length".into()));
        }
        let obj = self.obj.ok_or(Error::Missing("objective"))?;
        let grad = self.grad.ok_or(Error::Missing("gradient"))?;
        let (hess_sp, hess_fn) = self.hess.ok_or(Error::Missing("hessian"))?;

        let (con, (jac_sp, jac_fn)): (ConFn, (Vec<(usize, usize)>, JacFn)) = if m > 0 {
            (
                self.con.ok_or(Error::Missing("constraints"))?,
                self.jac.ok_or(Error::Missing("jacobian"))?,
            )
        } else {
            // The problem is unconstrained, so set up stub functions.
            (Box::new(|_, _| {}), (Vec::new(), Box::new(|_, _| {})))
        };

        for &(r, c) in &jac_sp {
            if r >= m || c >= n {
                return Err(Error::Invalid(format!(
                    "jacobian sparsity entry ({r}, {c}) out of range for {m}x{n}"
                )));
            }
        }
        for &(r, c) in &hess_sp {
            if r >= n || c >= n {
                return Err(Error::Invalid(format!(
                    "hessian sparsity entry ({r}, {c}) out of range for {n}x{n}"
                )));
            }
            if c > r {
                return Err(Error::Invalid(format!(
                    "hessian sparsity entry ({r}, {c}) is above the diagonal; \
                     IPOPT wants the lower triangle"
                )));
            }
        }

        let to_ix = |v: Vec<(usize, usize)>| -> Vec<(ipindex, ipindex)> {
            v.into_iter()
                .map(|(r, c)| (r as ipindex, c as ipindex))
                .collect()
        };

        // ---- payload ---------------------------------------------------------
        let mut cb = Box::new(Callbacks {
            eval_f: obj,
            eval_grad_f: grad,
            eval_g: con,
            eval_jac_g: jac_fn,
            eval_h: hess_fn,
            jac_sparsity: to_ix(jac_sp),
            hess_sparsity: to_ix(hess_sp),
            panic: None,
        });
        let nele_jac = cb.jac_sparsity.len() as ipindex;
        let nele_hess = cb.hess_sparsity.len() as ipindex;

        // CreateIpoptProblem copies the bounds internally, so these temporary
        // mutable buffers only need to live until the call returns.
        let mut x_l = self.x_l;
        let mut x_u = self.x_u;
        let mut g_l = self.g_l;
        let mut g_u = self.g_u;

        struct Guard(ffi::IpoptProblem);
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe { ffi::FreeIpoptProblem(self.0) }
            }
        }

        // Bindgen wraps C function pointers in Option (they're nullable in C),
        // so trampolines are passed as Some(...).
        let problem = unsafe {
            ffi::CreateIpoptProblem(
                n as ipindex,
                x_l.as_mut_ptr(),
                x_u.as_mut_ptr(),
                m as ipindex,
                g_l.as_mut_ptr(),
                g_u.as_mut_ptr(),
                nele_jac,
                nele_hess,
                0, // C-style, 0-based indexing
                Some(tramp_f),
                Some(tramp_g),
                Some(tramp_grad_f),
                Some(tramp_jac_g),
                Some(tramp_h),
            )
        };
        if problem.is_null() {
            return Err(Error::CreateFailed);
        }
        let guard = Guard(problem);

        // ---- options -----------------------------------------------------------
        // The generated signatures take *mut c_char, but IPOPT only reads the
        // strings, so casting away const from CString::as_ptr() is fine.
        for (key, val) in &self.options {
            let k = CString::new(key.as_str()).map_err(|_| Error::BadOption(key.clone()))?;
            let ok = unsafe {
                match val {
                    OptVal::Num(v) => {
                        ffi::AddIpoptNumOption(guard.0, k.as_ptr() as *mut c_char, *v)
                    }
                    OptVal::Int(v) => {
                        ffi::AddIpoptIntOption(guard.0, k.as_ptr() as *mut c_char, *v as ipindex)
                    }
                    OptVal::Str(s) => {
                        let v =
                            CString::new(s.as_str()).map_err(|_| Error::BadOption(key.clone()))?;
                        ffi::AddIpoptStrOption(
                            guard.0,
                            k.as_ptr() as *mut c_char,
                            v.as_ptr() as *mut c_char,
                        )
                    }
                }
            };
            if !ok {
                return Err(Error::BadOption(key.clone()));
            }
        }

        // ---- solve ----------------------------------------------------------------
        let mut x = x0.to_vec();
        let mut g = vec![0.0; m];
        let mut obj_val = 0.0;
        let mut mult_g = vec![0.0; m];
        let mut mult_x_l = vec![0.0; n];
        let mut mult_x_u = vec![0.0; n];

        let status = unsafe {
            ffi::IpoptSolve(
                guard.0,
                x.as_mut_ptr(),
                if m > 0 {
                    g.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                &mut obj_val,
                if m > 0 {
                    mult_g.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                mult_x_l.as_mut_ptr(),
                mult_x_u.as_mut_ptr(),
                &mut *cb as *mut Callbacks as UserDataPtr,
            )
        };
        drop(guard); // FreeIpoptProblem

        // Re-throw any panic captured inside a callback.
        if let Some(payload) = cb.panic.take() {
            std::panic::resume_unwind(payload);
        }

        Ok(Solution {
            status: status.into(),
            x,
            obj: obj_val,
            g,
            mult_g,
            mult_x_l,
            mult_x_u,
        })
    }
}
