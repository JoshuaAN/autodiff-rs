use ir::{function::Function, ty::Ty};

pub fn vmap(func: &Function, in_batched: &[bool], b: u32) -> Function {
    assert_eq!(in_batched.len(), func.num_params());
    assert!(b > 0);
    if !in_batched.iter().any(|&x| x) {
        return func.clone();
    };

    let params: Vec<Ty> = func
        .params
        .iter()
        .zip(in_batched)
        .map(|(&p, &ib)| {
            let t = func.value_ty(p);
            if ib { t.prepend(b) } else { t }
        })
        .collect();

    let mut f = Function::new(&params);

    todo!("finish vmap");
}
