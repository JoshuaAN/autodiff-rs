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

impl Ty {
    pub const SCALAR: Ty = Ty {
        rank: 0,
        dims: [0; MAX_RANK],
    };

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }

    pub fn is_scalar(self) -> bool {
        self.rank == 0
    }

    pub fn prepend(self, n: u32) -> Ty {
        assert!((self.rank as usize) < MAX_RANK);

        let mut dims = [0; MAX_RANK];
        dims[0] = n;
        let r = self.rank as usize;
        dims[1..=r].copy_from_slice(&self.dims[..r]);
        Ty {
            rank: self.rank + 1,
            dims,
        }
    }

    pub fn broadcast(a: Ty, b: Ty) -> Result<Ty, String> {
        match (a.is_scalar(), b.is_scalar()) {
            (true, _) => Ok(b),
            (_, true) => Ok(a),
            _ if a == b => Ok(a),
            _ => Err(format!("shape mismatch: {:?} vs {:?}", a.dims(), b.dims())),
        }
    }
}
