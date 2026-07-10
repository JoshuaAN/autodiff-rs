use std::collections::HashMap;

use ir::{function::Function, instruction::InstructionData, ty::{Ty, TyData}, value::Value};

pub fn vmap(func: &Function, in_batched: &[bool], b: u32) -> Function {
    assert_eq!(in_batched.len(), func.num_params());
    assert!(b > 0);

    let param_tys: Vec<TyData> = func
        .args()
        .iter()
        .zip(in_batched)
        .map(|(&p, &ib)| {
            let td = func.ty_data(func.value_ty(p));
            if ib { td.prepend(b) } else { td }
        })
        .collect();
    let mut out = Function::new(&param_tys);

    // Maps old value to new value and whether it is batched.
    let mut env: HashMap<Value, (Value, bool)> = HashMap::new();
    let new_params: Vec<Value> = out.args().to_vec();
    for ((&old, new), &ib) in func.args().iter().zip(new_params).zip(in_batched) {
        env.insert(old, (new, ib));
    }

    for &inst in func.layout() {
        let (data, batched) = match func.inst_data(inst) {
            InstructionData::Constant(c) => (InstructionData::Constant(c), false),
            InstructionData::Unary(op, a) => {
                let (a_new, a_batched) = env[&a];
                (InstructionData::Unary(op, a_new), a_batched)
            }
            InstructionData::Binary(op, a, b) => {
                let (mut a_new, a_batched) = env[&a];
                let (mut b_new, b_batched) = env[&b];
                if a_batched && !b_batched {
                    let t = out.value_ty(a_new);
                    b_new = out.broadcast_to(b_new, t);
                } else if !a_batched && b_batched {
                    let t = out.value_ty(b_new);
                    a_new = out.broadcast_to(a_new, t);
                }
                (InstructionData::Binary(op, a_new, b_new), a_batched || b_batched)
            }
            InstructionData::Broadcast(a, ty) => {
                todo!("vmap not implemented for broadcast")
            }
            InstructionData::Reduce(op, a, s) => {
                todo!("vmap not implemented for reduce")
            }
        };
        let v = out.emit(data);
        env.insert(func.inst_result(inst), (v, batched));
    }

    for &r in &func.returns {
        let (v, r_batched) = env[&r];
        let v = if r_batched {
            v
        } else {
            let td = out.ty_data(out.value_ty(v)).prepend(b);
            let ty = out.intern_ty(td);
            out.broadcast_to(v, ty)
        };
        out.returns.push(v);
    }

    out
}
