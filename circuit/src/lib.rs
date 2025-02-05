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
                let left_index_value = current_input[self.layers[i].gates[j].left_index];
                let right_index_value = current_input[self.layers[i].gates[j].right_index];

                let current_gate_evaluation = match self.layers[i].gates[j].operator {
                    Operator::Add => left_index_value + right_index_value,
                    Operator::Mul => left_index_value * right_index_value
                };

                resultant_evaluations.push(current_gate_evaluation);
            }

            current_input = resultant_evaluations;
        }

        current_input[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_circuit_evaluation() {
        let input = vec![Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5)];
        let gate1 = Gate::new(0, 1, Operator::Add);
        let gate2 = Gate::new(2, 3, Operator::Mul);

        let gate3 = Gate::new(0, 1, Operator::Mul);

        let layer1 = Layer::new(vec![gate1, gate2]);
        let layer2 = Layer::new(vec![gate3]);

        let circuit = Circuit::<Fq>::new(vec![layer1, layer2]);

        assert_eq!(circuit.evaluate(input), Fq::from(100));
    }

    #[test]
    fn test_circuit_evaluation2() {
        let input = vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)];
        let gate1 = Gate::new(0, 1, Operator::Add);
        let gate2 = Gate::new(2, 3, Operator::Mul);

        let gate3 = Gate::new(0, 1, Operator::Add);

        let layer1 = Layer::new(vec![gate1, gate2]);
        let layer2 = Layer::new(vec![gate3]);

        let circuit = Circuit::<Fq>::new(vec![layer1, layer2]);

        assert_eq!(circuit.evaluate(input), Fq::from(15));
    }
}