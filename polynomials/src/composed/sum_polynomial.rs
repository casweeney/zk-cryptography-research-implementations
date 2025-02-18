use ark_ff::PrimeField;
use crate::composed::product_polynomial::ProductPolynomial;
use crate::multilinear::evaluation_form::MultilinearPolynomial;

// Sum Polynomial hold 2 or more multilinear polynomials and performs addition operations on them
#[derive(Clone, Debug, PartialEq)]
pub struct SumPolynomial<F: PrimeField> {
    pub product_polynomials: Vec<ProductPolynomial<F>>
}

impl <F: PrimeField>SumPolynomial<F> {
    pub fn new(product_polynomials: Vec<ProductPolynomial<F>>) -> Self {
        // We expect that all the multilinear polynomial will have the number same variables
        let first_polynomial = &product_polynomials[0].polynomials[0];
        let num_of_variables = first_polynomial.number_of_variables();

        assert!(
            product_polynomials.iter().all(|product_poly| product_poly.polynomials.iter().all(|polynomial| polynomial.number_of_variables() == num_of_variables)),
            "different number of variables"
        );

        Self {
            product_polynomials
        }
    }
    
    pub fn evaluate(&self, values: &Vec<F>) -> F {
        let mut result = F::zero();

        for product_polynomial in self.product_polynomials.iter() {
            result += product_polynomial.evaluate(&values);
        }

        result
    }
    
    pub fn partial_evaluate(&self, evaluating_variable: usize, value: F) -> Self {
        let mut evaluated_polynomials = Vec::new();

        for product_polynomial in self.product_polynomials.iter() {
            let evaluated_product_poly = product_polynomial.partial_evaluate(evaluating_variable, value);

            evaluated_polynomials.push(ProductPolynomial::new(evaluated_product_poly));
        }

        Self {
            product_polynomials: evaluated_polynomials
        }
    }

    // This function reduces the Vec of Product polynomials
    // to one Polynomial by basically performing element-wise addition
    pub fn add_polynomials_element_wise(&self) -> MultilinearPolynomial<F> {
        assert!(self.product_polynomials.len() > 1, "more than one product polynomial required for add operation");

        let first_product = self.product_polynomials[0].multiply_polynomials_element_wise();

        let mut resultant_values = first_product.evaluated_values.to_vec();

        for product_polynomial in self.product_polynomials.iter().skip(1) {
            let multiplied_poly = product_polynomial.multiply_polynomials_element_wise();

            for (i, value) in multiplied_poly.evaluated_values.iter().enumerate() {
                resultant_values[i] += value
            }
        }

        MultilinearPolynomial::new(&resultant_values)
    }

    pub fn convert_to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        for product_polynomial in &self.product_polynomials {
            bytes.extend_from_slice(&product_polynomial.convert_to_bytes());
        }

        bytes
    }

    pub fn degree(&self) -> usize {
        self.product_polynomials.len()
    }

    pub fn number_of_variables(&self) -> u32 {
        self.product_polynomials[0].polynomials[0].number_of_variables()
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
        
        let product_poly1 = ProductPolynomial::new(vec![poly1]);
        let product_poly2 = ProductPolynomial::new(vec![poly2]);
    
        // This is expected to panic
        SumPolynomial::new(vec![product_poly1, product_poly2]);
    }

    #[test]
    fn test_evaluate_sum_poly() {
        // First product polynomial
        let poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
        let product_poly1 = ProductPolynomial::new(vec![poly1a, poly1b]);

        // Second product polynomial
        let poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(1)]);
        let poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let product_poly2 = ProductPolynomial::new(vec![poly2a, poly2b]);

        let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);

        // a = 1, b = 2
        let values = vec![Fq::from(1), Fq::from(2)];

        assert_eq!(sum_polynomial.evaluate(&values), Fq::from(32));
    }

    #[test]
    fn test_partial_evaluate_sum_poly() {
        // First product polynomial
        let poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
        let product_poly1 = ProductPolynomial::new(vec![poly1a, poly1b]);

        // Second product polynomial
        let poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(1)]);
        let poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let product_poly2 = ProductPolynomial::new(vec![poly2a, poly2b]);

        let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);
        let evaluated_sum_poly = sum_polynomial.partial_evaluate(0, Fq::from(2));

        // Expected partial evaluations:
        let expect_poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(4)]);
        let expect_poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(6)]);
        let expect_product1 = ProductPolynomial::new(vec![expect_poly1a, expect_poly1b]);

        // For second product: poly2a(2) * poly2b = [0,2] * [0,2]
        let expect_poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(2)]);
        let expect_poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(4)]);
        let expect_product2 = ProductPolynomial::new(vec![expect_poly2a, expect_poly2b]);

        let expected_sum_poly = SumPolynomial::new(vec![expect_product1, expect_product2]);

        assert_eq!(evaluated_sum_poly.product_polynomials[0].polynomials, expected_sum_poly.product_polynomials[0].polynomials);
        assert_eq!(evaluated_sum_poly.product_polynomials[1].polynomials, expected_sum_poly.product_polynomials[1].polynomials);
    }

    #[test]
    fn test_add_polynomials_element_wise() {
        // First product polynomial: (2x)(3y)
    let poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
    let poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
    let product_poly1 = ProductPolynomial::new(vec![poly1a, poly1b]);

    // Second product polynomial: (1x)(2y)
    let poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(1)]);
    let poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
    let product_poly2 = ProductPolynomial::new(vec![poly2a, poly2b]);

    let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);

    let expected_sum = MultilinearPolynomial::new(&vec![
        Fq::from(0), 
        Fq::from(0), 
        Fq::from(0), 
        Fq::from(8)  // (2*3) + (1*2) = 6 + 2 = 8
    ]);

    assert_eq!(sum_polynomial.add_polynomials_element_wise(), expected_sum);
    }

    #[test]
    fn test_degree_sum_poly() {
        let poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
        let product_poly1 = ProductPolynomial::new(vec![poly1a, poly1b]);

        let poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(1)]);
        let poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let product_poly2 = ProductPolynomial::new(vec![poly2a, poly2b]);

        let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);

        assert_eq!(sum_polynomial.degree(), 2);
    }

    #[test]
    fn test_number_of_variables() {
        let poly1a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let poly1b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
        let product_poly1 = ProductPolynomial::new(vec![poly1a, poly1b]);

        let poly2a = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(1)]);
        let poly2b = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let product_poly2 = ProductPolynomial::new(vec![poly2a, poly2b]);

        let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);

        assert_eq!(sum_polynomial.number_of_variables(), 2);
    }
}