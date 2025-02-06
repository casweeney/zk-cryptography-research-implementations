use ark_ff::PrimeField;
use std::marker::PhantomData;

pub enum Operator {
    Add,
    Mul
}

pub struct Gate {
    pub left_index: usize,
    pub right_index: usize,
    // Used to decide the index position where a gate result will be placed in the output vector
    pub output_index: usize,
    pub operator: Operator
}

pub struct Layer {
    pub gates: Vec<Gate>
}

pub struct Circuit<F: PrimeField> {
    pub layers: Vec<Layer>,
    // Track intermediate values at each layer
    pub layer_evaluations: Vec<Vec<F>>, 
    _phantom: PhantomData<F>
}

//////////////// Gate Implementation ///////////////////
impl Gate {
    pub fn new(left_index: usize, right_index: usize, output_index: usize, operator: Operator) -> Self {
        Self {
            left_index,
            right_index,
            output_index,
            operator
        }
    }
}

/////////////// Layer implementation ///////////////////
impl Layer {
    pub fn new(gates: Vec<Gate>) -> Self {
        Self {
            gates
        }
    }
}


//////////////// Circuit Implementation /////////////////
impl <F: PrimeField>Circuit<F> {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers,
            layer_evaluations: Vec::new(),
            _phantom: PhantomData
        }
    }

    pub fn evaluate(&mut self, values: Vec<F>) -> Vec<F> {
        let mut current_input = values.clone();

        // Store the initial input values
        self.layer_evaluations = vec![current_input.clone()];

        // Using Iterators
        for layer in self.layers.iter() {
            let max_output_index = layer.gates.iter()
                .map(|gate| gate.output_index)
                .max()
                .unwrap_or(0);


            let mut resultant_evaluations = vec![F::zero(); max_output_index + 1];

            for gate in layer.gates.iter() {
                let left_index_value = current_input[gate.left_index];
                let right_index_value = current_input[gate.right_index];

                let current_gate_evaluation = match gate.operator {
                    Operator::Add => left_index_value + right_index_value,
                    Operator::Mul => left_index_value * right_index_value
                };

                // place the result of the evaluation of each gate at the specified output index
                resultant_evaluations[gate.output_index] = current_gate_evaluation;
            }

            current_input = resultant_evaluations;

            // store each layer's output, so that we can have the evaluations of each layer represented as an array
            self.layer_evaluations.push(current_input.clone());
        }

        current_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_circuit_evaluation() {
        let input = vec![Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5)];

        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        let gate2 = Gate::new(2, 3, 1,  Operator::Mul);

        let gate3 = Gate::new(0, 1, 0, Operator::Mul);

        let layer1 = Layer::new(vec![gate1, gate2]);
        let layer2 = Layer::new(vec![gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer1, layer2]);

        let result = circuit.evaluate(input);

        assert_eq!(result[0], Fq::from(100));
    }

    #[test]
    fn test_circuit_evaluation2() {
        let input = vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)];
        let gate1 = Gate::new(0, 1, 1, Operator::Add);
        let gate2 = Gate::new(2, 3, 0, Operator::Mul);

        let gate3 = Gate::new(0, 1, 0, Operator::Add);

        let layer1 = Layer::new(vec![gate1, gate2]);
        let layer2 = Layer::new(vec![gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer1, layer2]);
        let result = circuit.evaluate(input);

        assert_eq!(result[0], Fq::from(15));
    }

    #[test]
    fn test_circuit_evaluation3() {
        let input = vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5), Fq::from(6), Fq::from(7), Fq::from(8)];
        
        // layer 1 gates
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        let gate2 = Gate::new(2, 3, 1, Operator::Mul);
        let gate3 = Gate::new(4, 5, 2, Operator::Mul);
        let gate4 = Gate::new(6, 7, 3, Operator::Mul);

        // layer 2 gates
        let gate5 = Gate::new(0, 1, 0, Operator::Add);
        let gate6 = Gate::new(2, 3, 1, Operator::Mul);

        // layer 3 gates
        let gate7 = Gate::new(0, 1, 0, Operator::Add);

        // Layers
        let layer1 = Layer::new(vec![gate1, gate2, gate3, gate4]);
        let layer2 = Layer::new(vec![gate5, gate6]);
        let layer3 = Layer::new(vec![gate7]);

        let mut circuit = Circuit::<Fq>::new(vec![layer1, layer2, layer3]);
        let result = circuit.evaluate(input);

        assert_eq!(result[0], Fq::from(1695));
    }
}