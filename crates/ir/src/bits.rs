#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Bits64(u64);

impl Bits64 {
    pub fn from_f64(x: f64) -> Self {
        Bits64(x.to_bits())
    }
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}
