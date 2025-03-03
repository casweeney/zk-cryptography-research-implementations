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
    _phantom: PhantomData<F>
}

pub struct CircuitEvaluationResult<F: PrimeField> {
    pub output: Vec<F>,
    pub layer_evaluations: Vec<Vec<F>>
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
            _phantom: PhantomData
        }
    }

    pub fn evaluate(&mut self, values: Vec<F>) -> CircuitEvaluationResult<F> {
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
                resultant_evaluations[gate.output_index] += current_gate_evaluation;
            }

            current_input = resultant_evaluations;
            reversed_evaluations.push(current_input.clone());
        }

        reversed_evaluations.reverse();

        CircuitEvaluationResult {
            output: reversed_evaluations[0].clone(),
            layer_evaluations: reversed_evaluations
        }
    }

    // This function gets the evaluations of a layer: Vec<F> whose index is passed as layer_index,
    // then it converts it to a Multilinear polynomial
    // This will be used for the MLE: Multilinear Extension
    pub fn w_i_polynomial(circuit_evaluation: &CircuitEvaluationResult<F>, layer_index: usize) -> MultilinearPolynomial<F> {
        assert!(layer_index < circuit_evaluation.layer_evaluations.len(), "layer index out of bounds");

        MultilinearPolynomial::new(&circuit_evaluation.layer_evaluations[layer_index])
    }

    pub fn add_i_and_mul_i_mle(&mut self, layer_index: usize) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let number_of_layer_variables = num_of_layer_variables(layer_index);
        let boolean_hypercube_combinations = 1 << number_of_layer_variables; // 2 ^ number_of_layer_variables

        let mut add_i_values = vec![F::zero(); boolean_hypercube_combinations];
        let mut mul_i_values = vec![F::zero(); boolean_hypercube_combinations];

        for gate in self.layers[layer_index].gates.iter() {
            match gate.operator {
                Operator::Add => {
                    let position_index = convert_to_binary_and_to_decimal(layer_index, gate.output_index, gate.left_index, gate.right_index);
                    add_i_values[position_index] = F::one();
                },
                Operator::Mul => {
                    let position_index = convert_to_binary_and_to_decimal(layer_index, gate.output_index, gate.left_index, gate.right_index);
                    mul_i_values[position_index] = F::one();
                }
            }
        }

        let add_i_polynomial = MultilinearPolynomial::new(&add_i_values);
        let mul_i_polynomial = MultilinearPolynomial::new(&mul_i_values);

        (add_i_polynomial, mul_i_polynomial)
    }
}


pub fn num_of_layer_variables(layer_index: usize) -> usize {
    if layer_index == 0 {
        return 3;
    }

    let var_a_length = layer_index;
    let var_b_length = var_a_length + 1;
    let var_c_length = var_a_length + 1;

    let num_of_variables = var_a_length + var_b_length + var_c_length;

    num_of_variables
}

pub fn convert_to_binary_and_to_decimal(layer_index: usize, variable_a: usize, variable_b: usize, variable_c: usize) -> usize {
    // convert decimal to binary
    let a_in_binary = convert_decimal_to_padded_binary(variable_a, layer_index);
    let b_in_binary = convert_decimal_to_padded_binary(variable_b, layer_index + 1);
    let c_in_binary = convert_decimal_to_padded_binary(variable_c, layer_index + 1);

    // combine a, b and c binaries
    let combined_binary = a_in_binary + &b_in_binary + &c_in_binary;
    
    // convert the combined binaries back to decimal
    usize::from_str_radix(&combined_binary, 2).unwrap_or(0)
}

pub fn convert_decimal_to_padded_binary(decimal_number: usize, bit_length: usize) -> String {
    format!("{:0>width$b}", decimal_number, width = bit_length)
}

// unused function: just another method of converting a decimal number to padded binary number
pub fn transform_decimal_to_padded_binary(decimal_number: usize, mut bit_length: usize) -> String {
    if bit_length == 0 {
        bit_length = 1;
    }
    
    let binary = format!("{:b}", decimal_number);

    "0".repeat(bit_length.saturating_sub(binary.len())) + &binary
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

        assert_eq!(result.output[0], Fq::from(100));
        assert_eq!(result.layer_evaluations, expected_layers_evaluation);
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

        assert_eq!(result.output[0], Fq::from(15));
        assert_eq!(result.layer_evaluations, expected_layers_evaluation)
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

        assert_eq!(result.output[0], Fq::from(1695));
    }

    #[test]
    fn test_num_of_layer_variables() {
        // Assert Equal
        assert_eq!(num_of_layer_variables(0), 3);
        assert_eq!(num_of_layer_variables(1), 5);
        assert_eq!(num_of_layer_variables(2), 8);
        assert_eq!(num_of_layer_variables(3), 11);
        assert_eq!(num_of_layer_variables(4), 14);

        // Assert Not Equal
        assert_ne!(num_of_layer_variables(2), 7);
        assert_ne!(num_of_layer_variables(3), 9);
    }

    #[test]
    fn test_add_i_and_mul_i_mle_layer0() {
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        // switched output index
        let gate2 = Gate::new(0, 1, 1, Operator::Add);
        let gate3 = Gate::new(2, 3, 0, Operator::Mul);

        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1]);

        let (add_i_poly, mul_i_poly) = circuit.add_i_and_mul_i_mle(0);
        let expected_add_i_poly = MultilinearPolynomial::new(
            &vec![Fq::from(0), Fq::from(1), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0)]
        );

        let expected_mul_i_poly = MultilinearPolynomial::new(
            &vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(0)]
        );

        assert_eq!(add_i_poly, expected_add_i_poly);
        assert_eq!(mul_i_poly, expected_mul_i_poly);
    }

    #[test]
    fn test_add_i_and_mul_i_mle_layer1() {
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        // switched output index
        let gate2 = Gate::new(0, 1, 1, Operator::Add);
        let gate3 = Gate::new(2, 3, 0, Operator::Mul);

        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1]);
        // let result = circuit.evaluate(input);

        let (add_i_poly, mul_i_poly) = circuit.add_i_and_mul_i_mle(1);

        // For layer 1: 2^5 = 32 combinations
        let mut expected_add = vec![Fq::from(0); 32];
        expected_add[17] = Fq::from(1);  // position from gate2: "10001" = 17
        let expected_add_i_poly = MultilinearPolynomial::new(&expected_add);

        let mut expected_mul = vec![Fq::from(0); 32];
        expected_mul[11] = Fq::from(1);  // position from gate3: "01011" = 11
        let expected_mul_i_poly = MultilinearPolynomial::new(&expected_mul);

        assert_eq!(add_i_poly, expected_add_i_poly);
        assert_eq!(mul_i_poly, expected_mul_i_poly);
    }
}