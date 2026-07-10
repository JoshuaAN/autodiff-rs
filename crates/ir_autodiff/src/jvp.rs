use std::collections::HashMap;

use ir::{
    bits::Bits64,
    function::Function,
    instruction::InstructionData,
    op::{BinaryOp, UnaryOp},
    ty::TyData,
    value::Value,
};

type Tan = Option<Value>;

/// Returns a function computing (returns, return tangents) from (primals, tangents),
/// where the input is a function computing (returns) from (primals.).
pub fn jvp(func: &Function) -> Function {
    let n = func.num_params();

    // Inputs are primals and tangents.
    let primal_tys: Vec<TyData> = func
        .args()
        .iter()
        .map(|&p| func.ty_data(func.value_ty(p)))
        .collect();
    let param_tys: Vec<TyData> = primal_tys
        .iter()
        .chain(primal_tys.iter())
        .copied()
        .collect();
    let mut out = Function::new(&param_tys);

    // Maps old values to new values and their tangents.
    let mut env: HashMap<Value, (Value, Tan)> = HashMap::new();
    let new_params: Vec<Value> = out.args().to_vec();
    let (primals, tangents) = new_params.split_at(n);
    for ((&old, &p), &t) in func.args().iter().zip(primals).zip(tangents) {
        env.insert(old, (p, Some(t)));
    }

    for &inst in func.layout() {
        use BinaryOp::*;
        use InstructionData::*;
        use UnaryOp::*;

        let (data, tan) = match func.inst_data(inst) {
            Constant(c) => (Constant(c), None),
            Unary(op, a) => {
                let (pa, ta) = env[&a];
                let d = Unary(op, pa);
                let tan = ta.map(|ta| match op {
                    // d(-a) = -da
                    Neg => out.emit(Unary(Neg, ta)),
                    // d(sin a) = cos(a) * da
                    Sin => {
                        let c = out.emit(Unary(Cos, pa));
                        out.emit(Binary(Mul, c, ta))
                    }
                    // d(cos a) = -sin(a) * da
                    Cos => {
                        let s = out.emit(Unary(Sin, pa));
                        let m = out.emit(Binary(Mul, s, ta));
                        out.emit(Unary(Neg, m))
                    }
                });
                (d, tan)
            }
            Binary(op, a, b) => {
                let (pa, ta) = env[&a];
                let (pb, tb) = env[&b];
                let d = Binary(op, pa, pb);
                let tan = match op {
                    // d(a+b) = da + db  (either side may still be zero)
                    Add => add_tan(&mut out, ta, tb),
                    // d(a-b) = da - db
                    Sub => sub_tan(&mut out, ta, tb),
                    // d(a*b) = da*b + a*db
                    Mul => {
                        let l = ta.map(|ta| out.emit(Binary(Mul, ta, pb)));
                        let r = tb.map(|tb| out.emit(Binary(Mul, pa, tb)));
                        add_tan(&mut out, l, r)
                    }
                    // d(a/b) = (da - q*db)/b where q = a/b.
                    // The q here CSEs with the primal quotient below.
                    Div => {
                        let num = match tb {
                            None => ta,
                            Some(tb) => {
                                let q = out.emit(Binary(Div, pa, pb));
                                let qd = out.emit(Binary(Mul, q, tb));
                                sub_tan(&mut out, ta, Some(qd))
                            }
                        };
                        num.map(|n| out.emit(Binary(Div, n, pb)))
                    }
                };
                (d, tan)
            }
            Broadcast(a, ty, map) => {
                let (pa, ta) = env[&a];
                let ty = out.import_ty(func, ty);
                let d = Broadcast(pa, ty, map);
                let tan = ta.map(|ta| out.emit(Broadcast(ta, ty, map)));
                (d, tan)
            }
            Reduce(op, a, s) => {
                let (pa, ta) = env[&a];
                let d = Reduce(op, pa, s);
                let tan = ta.map(|ta| out.emit(Reduce(op, ta, s)));
                (d, tan)
            }
        };

        let v = out.emit(data);
        env.insert(func.inst_result(inst), (v, tan));
    }

    let rets: Vec<Value> = func.returns.iter().map(|&r| env[&r].0).collect();
    let tans: Vec<Value> = func
        .returns
        .iter()
        .map(|&r| match env[&r].1 {
            Some(t) => t,
            None => {
                let z = out.emit(InstructionData::Constant(Bits64::from_f64(0.0)));
                let ty = out.value_ty(env[&r].0);
                out.broadcast_to(z, ty)
            }
        })
        .collect();
    out.returns = rets.into_iter().chain(tans).collect();
    out
}

fn add_tan(out: &mut Function, a: Tan, b: Tan) -> Tan {
    match (a, b) {
        (None, t) => t,
        (t, None) => t,
        (Some(a), Some(b)) => Some(out.emit(InstructionData::Binary(BinaryOp::Add, a, b))),
    }
}

fn sub_tan(out: &mut Function, a: Tan, b: Tan) -> Tan {
    match (a, b) {
        (t, None) => t,
        (None, Some(b)) => Some(out.emit(InstructionData::Unary(UnaryOp::Neg, b))),
        (Some(a), Some(b)) => Some(out.emit(InstructionData::Binary(BinaryOp::Sub, a, b))),
    }
}
