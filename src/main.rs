use tape::{drivers::Jacobian, tape::Tape};

fn main() {
    let tape = Tape::new();

    let x = tape.param();
    let y = tape.param();
    let z = x * x * y + x * y.sin();

    let jac = Jacobian::new(&tape, &[x, y], &[z]);

    jac.eval(&[1.5, 2.3]);
}
