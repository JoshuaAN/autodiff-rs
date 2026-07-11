use crate::tape::Tape;

/// A tape compiler.
pub trait Backend {
    type Func: CompiledFunction;
    type Error: std::error::Error;
 
    fn compile(&mut self, tape: &Tape) -> Result<Self::Func, Self::Error>;
}
 
/// An executable kernel produced by a [`Backend`].
pub trait CompiledFunction {
    fn num_params(&self) -> usize;
    fn num_returns(&self) -> usize;
 
    /// Evaluates the kernel.
    ///
    /// # Panics
    ///
    /// If `args.len() != num_params()` or `returns.len() != num_returns()`.
    fn call(&mut self, args: &[f64], returns: &mut [f64]);
}
