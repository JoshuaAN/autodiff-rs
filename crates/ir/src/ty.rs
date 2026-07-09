const MAX_RANK: usize = 8;

/// Represents an array type in the program. All values are arrays in the IR, where
/// scalars are arrays with rank zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ty {
    /// Number of dimensions in the Tensor.
    rank: u8,

    /// The dimension sizes of the Tensor.
    dims: [u32; MAX_RANK],
}
