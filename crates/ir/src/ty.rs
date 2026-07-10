use index_vec::define_index_type;

const MAX_RANK: usize = 8;

define_index_type! { pub struct Ty = u32; }

/// Represents an array type in the program. All values are arrays in the IR, where
/// scalars are arrays with rank zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TyData {
    /// Number of dimensions in the Tensor.
    rank: u8,

    /// The dimension sizes of the Tensor.
    dims: [u32; MAX_RANK],
}

impl TyData {
    pub const SCALAR: TyData = TyData {
        rank: 0,
        dims: [0; MAX_RANK],
    };

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }

    pub fn is_scalar(self) -> bool {
        self.rank == 0
    }

    pub fn prepend(self, n: u32) -> TyData {
        assert!((self.rank as usize) < MAX_RANK, "rank overflow");
        let mut dims = [0; MAX_RANK];
        dims[0] = n;
        let r = self.rank as usize;
        dims[1..=r].copy_from_slice(&self.dims[..r]);
        TyData {
            rank: self.rank + 1,
            dims,
        }
    }
}

/// A set of dimension indices, stored as a bitmask.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DimSet(pub u8);

/// Maps source dim i to output dim map[i]. Entries past `len` are zero, so derived
/// Eq/Hash are canonical.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DimMap {
    len: u8,
    map: [u8; MAX_RANK],
}
