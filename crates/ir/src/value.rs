use crate::instruction::Inst;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    /// The value is the result of an instruction.
    Result(Inst),

    /// The value is a parameter of the function.
    Param(usize),
}
