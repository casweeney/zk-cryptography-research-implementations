use ark_ff::PrimeField;

pub enum Operator {
    Add(bool),
    Mul(bool)
}

pub struct Gate<F: PrimeField> {
    pub output: F,
    pub left: F,
    pub right: F,
    pub operator: Operator
}

pub struct Layer<F: PrimeField> {
    pub gates: Vec<Gate<F>>
}

pub struct Circuit<F: PrimeField> {
    pub layers: Vec<Layer<F>>
}

impl <F: PrimeField>Gate<F> {
    pub fn new() -> Self {
        todo!()
    }
}

impl <F: PrimeField>Circuit<F> {
    pub fn execute(values: Vec<F>) {
        
    }
}