use std::{any::Any, panic::{AssertUnwindSafe, catch_unwind}};

use ipopt_sys::{UserDataPtr, ipindex, ipnumber};


pub(crate) type ObjFn = Box<dyn FnMut(&[f64]) -> f64>;
pub(crate) type GradFn = Box<dyn FnMut(&[f64], &mut [f64])>;
pub(crate) type ConFn = Box<dyn FnMut(&[f64], &mut [f64])>;
pub(crate) type JacFn = Box<dyn FnMut(&[f64], &mut [f64])>;
/// (x, obj_factor, lambda, values): Hessian of the Lagrangian,
/// lower triangle, values in sparsity-pattern order.
pub(crate) type HessFn = Box<dyn FnMut(&[f64], f64, &[f64], &mut [f64])>;

pub(crate) struct Callbacks {
    pub(crate) eval_f: ObjFn,
    pub(crate) eval_grad_f: GradFn,
    pub(crate) eval_g: ConFn,
    pub(crate) eval_jac_g: JacFn,
    pub(crate) eval_h: HessFn,
    pub(crate) jac_sparsity: Vec<(ipindex, ipindex)>,
    pub(crate) hess_sparsity: Vec<(ipindex, ipindex)>,
    pub(crate) panic: Option<Box<dyn Any + Send + 'static>>,
}

unsafe fn ro<'a>(p: *const f64, len: usize) -> &'a [f64] {
    if p.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, len) }
    }
}
unsafe fn rw<'a>(p: *mut f64, len: usize) -> &'a mut [f64] {
    if p.is_null() || len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }
}

pub(crate) unsafe extern "C" fn tramp_f(
    n: ipindex,
    x: *mut ipnumber,
    _new_x: bool,
    obj_value: *mut ipnumber,
    ud: UserDataPtr,
) -> bool {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    if cb.panic.is_some() {
        return false;
    }
    let x = unsafe { ro(x, n as usize) };
    match catch_unwind(AssertUnwindSafe(|| (cb.eval_f)(x))) {
        Ok(v) => {
            unsafe {
                *obj_value = v;
            }
            true
        }
        Err(p) => {
            cb.panic = Some(p);
            false
        }
    }
}

pub(crate) unsafe extern "C" fn tramp_grad_f(
    n: ipindex,
    x: *mut ipnumber,
    _new_x: bool,
    grad_f: *mut ipnumber,
    ud: UserDataPtr,
) -> bool {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    if cb.panic.is_some() {
        return false;
    }
    let x = unsafe { ro(x, n as usize) };
    let grad = unsafe { rw(grad_f, n as usize) };
    match catch_unwind(AssertUnwindSafe(|| (cb.eval_grad_f)(x, grad))) {
        Ok(()) => true,
        Err(p) => {
            cb.panic = Some(p);
            false
        }
    }
}

pub(crate) unsafe extern "C" fn tramp_g(
    n: ipindex,
    x: *mut ipnumber,
    _new_x: bool,
    m: ipindex,
    g: *mut ipnumber,
    ud: UserDataPtr,
) -> bool {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    if cb.panic.is_some() {
        return false;
    }
    let x = unsafe { ro(x, n as usize) };
    let g = unsafe { rw(g, m as usize) };
    match catch_unwind(AssertUnwindSafe(|| (cb.eval_g)(x, g))) {
        Ok(()) => true,
        Err(p) => {
            cb.panic = Some(p);
            false
        }
    }
}

pub(crate) unsafe extern "C" fn tramp_jac_g(
    n: ipindex,
    x: *mut ipnumber,
    _new_x: bool,
    _m: ipindex,
    nele_jac: ipindex,
    i_row: *mut ipindex,
    j_col: *mut ipindex,
    values: *mut ipnumber,
    ud: UserDataPtr,
) -> bool {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    if cb.panic.is_some() {
        return false;
    }
    if values.is_null() {
        // Structure phase (x is NULL here): report the sparsity pattern.
        for (k, &(r, c)) in cb.jac_sparsity.iter().enumerate() {
            unsafe { *i_row.add(k) = r };
            unsafe { *j_col.add(k) = c };
        }
        true
    } else {
        let x = unsafe { ro(x, n as usize) };
        let vals = unsafe { rw(values, nele_jac as usize) };
        match catch_unwind(AssertUnwindSafe(|| (cb.eval_jac_g)(x, vals))) {
            Ok(()) => true,
            Err(p) => {
                cb.panic = Some(p);
                false
            }
        }
    }
}

pub(crate) unsafe extern "C" fn tramp_h(
    n: ipindex,
    x: *mut ipnumber,
    _new_x: bool,
    obj_factor: ipnumber,
    m: ipindex,
    lambda: *mut ipnumber,
    _new_lambda: bool,
    nele_hess: ipindex,
    i_row: *mut ipindex,
    j_col: *mut ipindex,
    values: *mut ipnumber,
    ud: UserDataPtr,
) -> bool {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    if cb.panic.is_some() {
        return false;
    }
    if values.is_null() {
        for (k, &(r, c)) in cb.hess_sparsity.iter().enumerate() {
            unsafe { *i_row.add(k) = r };
            unsafe { *j_col.add(k) = c };
        }
        true
    } else {
        let x = unsafe { ro(x, n as usize) };
        let lambda = unsafe { ro(lambda, m as usize) };
        let vals = unsafe { rw(values, nele_hess as usize) };
        match catch_unwind(AssertUnwindSafe(|| {
            (cb.eval_h)(x, obj_factor, lambda, vals)
        })) {
            Ok(()) => true,
            Err(p) => {
                cb.panic = Some(p);
                false
            }
        }
    }
}