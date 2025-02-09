use ark_ff::PrimeField;
use std::marker::PhantomData;
use polynomials::multilinear::evaluation_form::MultilinearPolynomial;

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
        let mut current_input = values;

        let mut reversed_evaluations = Vec::new();
        reversed_evaluations.push(current_input.clone());

        // Iterate through the layers vector: in each iteration, iterate through the gates of each layer
        for layer in self.layers.iter().rev() {
            let max_output_index = layer.gates.iter()
                .map(|gate| gate.output_index)
                .max()
                .unwrap_or(0);

            let mut resultant_evaluations = vec![F::zero(); max_output_index + 1];

            // Iterate through the gates vector of each layer: 
            // use the left_index, right_index and operator of each Gate struct to perform an operation 
            // based on the values in the left and right index positions.
            // The operation is based on the Operator of the Gate: Add or Mul
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
            reversed_evaluations.push(current_input.clone());
        }

        reversed_evaluations.reverse();
        self.layer_evaluations = reversed_evaluations;

        self.layer_evaluations[0].clone()
    }

    // This function gets the evaluations of a layer: Vec<F> whose index is passed as layer_index,
    // then it converts it to a Multilinear polynomial
    // This will be used for the MLE: Multilinear Extension
    pub fn w_i_polynomial(&self, layer_index: usize) -> MultilinearPolynomial<F> {
        assert!(layer_index < self.layer_evaluations.len(), "layer index out of bounds");

        MultilinearPolynomial::new(&self.layer_evaluations[layer_index])
    }

    pub fn add_and_mul_i_mle(&mut self, layer_index: usize) -> (Vec<F>, Vec<F>) {
        let (_num_of_vars, bool_hypercuber_combinationss) = num_of_mle_vars_and_bool_hypercube_combinations(layer_index);

        let mut add_i_values = vec![0; bool_hypercuber_combinationss];
        let mut mul_i_values = vec![0; bool_hypercuber_combinationss];

        let layer_gates = &self.layers[layer_index].gates;

        for (gate_index, gate) in layer_gates.iter().enumerate() {
            match gate.operator {
                Operator::Add => {
                    let position_index = convert_binary_to_decimal(gate.output_index, gate.left_index, gate.right_index);
                    add_i_values[position_index] = 1;
                },
                Operator::Mul => {
                    let position_index = convert_binary_to_decimal(gate.output_index, gate.left_index, gate.right_index);
                    mul_i_values[position_index] = 1;
                }
            }
        }

        (add_i_values, mul_i_values)
    }
}


pub fn num_of_mle_vars_and_bool_hypercube_combinations(layer_index: usize) -> (usize, usize) {
    if layer_index == 0 {
        return (3, 1 << 3);
    }

    let var_a_length = layer_index;
    let var_b_and_c_length = var_a_length + 1;

    let num_of_variables = var_a_length + (2 * var_b_and_c_length);
    let bool_hypercube_combinations = 1 << num_of_variables;

    (num_of_variables, 1 << bool_hypercube_combinations)
}

pub fn convert_binary_to_decimal(variable_a: usize, variable_b: usize, variable_c: usize) -> usize {
    (variable_a << 2) | (variable_b << 1) | variable_c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_circuit_evaluation() {
        let input = vec![Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5)];

        let gate1 = Gate::new(0, 1, 0, Operator::Mul);
        let gate2 = Gate::new(0, 1, 0, Operator::Add);
        let gate3 = Gate::new(2, 3, 1,  Operator::Mul);
        
        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1]);

        let result = circuit.evaluate(input);

        let expected_layers_evaluation = vec![
            vec![Fq::from(100)],
            vec![Fq::from(5), Fq::from(20)],
            vec![Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5)]
        ];

        assert_eq!(result[0], Fq::from(100));
        assert_eq!(circuit.layer_evaluations, expected_layers_evaluation);
    }

    #[test]
    fn test_circuit_evaluation2() {
        let input = vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)];
        
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        // switched output index
        let gate2 = Gate::new(0, 1, 1, Operator::Add);
        let gate3 = Gate::new(2, 3, 0, Operator::Mul);

        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1]);
        let result = circuit.evaluate(input);

        let expected_layers_evaluation = vec![
            vec![Fq::from(15)],
            vec![Fq::from(12), Fq::from(3)],
            vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)]
        ];

        assert_eq!(result[0], Fq::from(15));
        assert_eq!(circuit.layer_evaluations, expected_layers_evaluation)
    }

    #[test]
    fn test_circuit_evaluation3() {
        let input = vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5), Fq::from(6), Fq::from(7), Fq::from(8)];

        // layer 0 gates
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        
        // layer 1 gates
        let gate2 = Gate::new(0, 1, 0, Operator::Add);
        let gate3 = Gate::new(2, 3, 1, Operator::Mul);

        // layer 2 gates
        let gate4 = Gate::new(0, 1, 0, Operator::Add);
        let gate5 = Gate::new(2, 3, 1, Operator::Mul);
        let gate6 = Gate::new(4, 5, 2, Operator::Mul);
        let gate7 = Gate::new(6, 7, 3, Operator::Mul);

        // Layers
        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);
        let layer2 = Layer::new(vec![gate4, gate5, gate6, gate7]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1, layer2]);
        let result = circuit.evaluate(input);

        assert_eq!(result[0], Fq::from(1695));
    }
}