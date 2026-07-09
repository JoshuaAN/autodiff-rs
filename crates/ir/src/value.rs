use index_vec::define_index_type;

use crate::{instruction::Inst, ty::Ty};

define_index_type! { pub struct Value = u32; }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueData {
    /// The value is the result of an instruction.
    Result { ty: Ty, inst: Inst },

    /// The value is a parameter of the function.
    Param { ty: Ty, num: u16 },
}

impl ValueData {
    pub fn ty(&self) -> Ty {
        match self {
            ValueData::Result { ty, inst: _ } => *ty,
            ValueData::Param { ty, num: _ } => *ty,
        }
    }
}
