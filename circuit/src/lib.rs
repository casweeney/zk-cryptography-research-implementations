use ark_ff::PrimeField;

pub enum Operator {
    Add,
    Mul
}

pub struct Gate<F: PrimeField> {
    pub output: F,
    pub left_input: F,
    pub right_input: F,
    pub operator: Operator
}

pub struct Layer<F: PrimeField> {
    pub gates: Vec<Gate<F>>
}

pub struct Circuit<F: PrimeField> {
    pub layers: Vec<Layer<F>>
}

//////////////// Gate Implementation ///////////////////

impl <F: PrimeField>Gate<F> {
    pub fn new(left_input: F, right_input: F, operator: Operator) -> Self {
        Self {
            output: evaluate_inputs(left_input, right_input, &operator),
            left_input,
            right_input,
            operator
        }
    }
}

pub fn evaluate_inputs<F: PrimeField>(left: F, right: F, operator: &Operator) -> F {
    match operator {
        Operator::Add => left + right,
        Operator::Mul => left * right
    }
}

/////////////// Layer implementation ////////////////////
impl <F: PrimeField>Layer<F> {
    pub fn new(gates: Vec<Gate<F>>) -> Self {
        Self {
            gates
        }
    }
}


//////////////// Circuit Implementation ///////////////////

impl <F: PrimeField>Circuit<F> {
    pub fn new(layers: Vec<Layer<F>>) -> Self {
        Self {
            layers
        }
    }
    
    pub fn execute(values: Vec<F>) {

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_gate_add_operator() {
        let gate = Gate::new(Fq::from(5), Fq::from(3), Operator::Add);

        assert_eq!(gate.output, Fq::from(8));
    }

    #[test]
    fn test_gate_mul_operator() {
        let gate = Gate::new(Fq::from(5), Fq::from(3), Operator::Mul);

        assert_eq!(gate.output, Fq::from(15));
    }
}