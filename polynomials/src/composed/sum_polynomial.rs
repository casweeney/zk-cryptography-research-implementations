use ark_ff::PrimeField;
use crate::multilinear::evaluation_form::MultilinearPolynomial;

// Sum Polynomial hold 2 or more multilinear polynomials and performs addition operations on them
pub struct SumPolynomial<F: PrimeField> {
    pub polynomials: Vec<MultilinearPolynomial<F>>
}

impl <F: PrimeField>SumPolynomial<F> {
    pub fn new(polynomials: Vec<MultilinearPolynomial<F>>) -> Self {
        // get the number of variables of the first multilinear polynomial
        // then iterate through all the polynomials and check if they have the same number of variables
        // assert if their number of variables are not the same
        let num_of_variables = polynomials[0].number_of_variables();
        assert!(
            polynomials.iter().all(|polynomial| polynomial.number_of_variables() == num_of_variables),
            "different number of variables"
        );

        Self {
            polynomials
        }
    }
    
    pub fn evaluate(&self, values: &Vec<F>) -> F {
        let mut result = F::zero();

        for polynomial in self.polynomials.iter() {
            result += polynomial.evaluate(&values);
        }

        result
    }
    
    pub fn partial_evaluate(&self, evaluating_variable: usize, value: F) -> Vec<MultilinearPolynomial<F>> {
        let mut evaluated_polynomials: Vec<MultilinearPolynomial<F>> = Vec::new();

        for polynomial in self.polynomials.iter() {
            let partial_evaluation = MultilinearPolynomial::partial_evaluate(&polynomial.evaluated_values, evaluating_variable, value);

            evaluated_polynomials.push(MultilinearPolynomial::new(&partial_evaluation));
        }

        evaluated_polynomials
    }

    pub fn add_polynomials_element_wise(&self) -> MultilinearPolynomial<F> {
        assert!(self.polynomials.len() > 1, "more than one polynomial required for mul operation");

        let mut resultant_values = self.polynomials[0].evaluated_values.to_vec();

        for polynomial in self.polynomials.iter().skip(1) {
            for (i, value) in polynomial.evaluated_values.iter().enumerate() {
                resultant_values[i] += value
            }
        }

        MultilinearPolynomial::new(&resultant_values)
    }

    pub fn convert_to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        for polynomial in &self.polynomials {
            bytes.extend_from_slice(&polynomial.convert_to_bytes());
        }

        bytes
    }

    pub fn degree(&self) -> usize {
        self.polynomials.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    #[should_panic(expected = "different number of variables")]
    fn test_new_with_different_polynomial_lengths() {
        let poly1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(2)]);  // 1 variable
        let poly2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);  // 2 variables
    
        // This is expected to panic
        SumPolynomial::new(vec![poly1, poly2]);
    }

    #[test]
    fn test_evaluate_sum_poly() {
        let polynomail1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let sum_polynomial = SumPolynomial::new(vec![polynomail1, polynomail2]);

        // a = 1, b = 2
        let values = vec![Fq::from(1), Fq::from(2)];

        assert_eq!(sum_polynomial.evaluate(&values), Fq::from(10));
    }

    #[test]
    fn test_partial_evaluate_sum_poly() {
        let polynomail1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let sum_polynomial = SumPolynomial::new(vec![polynomail1, polynomail2]);

        let expect_poly1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(4)]);
        let expect_poly2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(6)]);

        let expected_partial_eval_result = vec![expect_poly1, expect_poly2];

        assert_eq!(sum_polynomial.partial_evaluate(0, Fq::from(2)), expected_partial_eval_result);
    }

    #[test]
    fn test_add_polynomials_element_wise() {
        let polynomail1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let sum_polynomial = SumPolynomial::new(vec![polynomail1, polynomail2]);

        let expected_sum = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(5)]);

        assert_eq!(sum_polynomial.add_polynomials_element_wise(), expected_sum);
    }

    #[test]
    fn test_degree_sum_poly() {
        let polynomail1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let sum_polynomial = SumPolynomial::new(vec![polynomail1, polynomail2]);

        assert_eq!(sum_polynomial.degree(), 2);
    }
}