use ark_ff::PrimeField;
use std::marker::PhantomData;

pub enum Operator {
    Add,
    Mul
}

pub struct Gate {
    pub left_index: usize,
    pub right_index: usize,
    pub operator: Operator
}

pub struct Layer {
    pub gates: Vec<Gate>
}

pub struct Circuit<F: PrimeField> {
    pub layers: Vec<Layer>,
    _phantom: PhantomData<F>
}

//////////////// Gate Implementation ///////////////////
impl Gate {
    pub fn new(left_index: usize, right_index: usize, operator: Operator) -> Self {
        Self {
            left_index,
            right_index,
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
impl Layer {
    pub fn new(gates: Vec<Gate>) -> Self {
        Self {
            gates
        }
    }
}


//////////////// Circuit Implementation ///////////////////

impl <F: PrimeField>Circuit<F> {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers,
            _phantom: PhantomData
        }
    }

    pub fn evaluate(&self, values: Vec<F>) -> F {
        let mut current_input = values;

        for i in 0..self.layers.len() {
            let mut resultant_evaluations = Vec::new();

            for j in 0..self.layers[i].gates.len() {
                let current_gate_evaluation = match self.layers[i].gates[j].operator {
                    Operator::Add => current_input[self.layers[i].gates[j].left_index] + current_input[self.layers[i].gates[j].right_index],
                    Operator::Mul => current_input[self.layers[i].gates[j].left_index] * current_input[self.layers[i].gates[j].right_index]
                };

                resultant_evaluations.push(current_gate_evaluation);
            }

            current_input = resultant_evaluations;
        }

        current_input[0]
    }
}