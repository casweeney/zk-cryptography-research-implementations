use crate::multilinear::evaluation_form::MultilinearPolynomial;
use ark_ff::PrimeField;

// Product Polynomial hold 2 or more multilinear polynomials and performs multiplication operations on them
#[derive(Clone, Debug, PartialEq)]
pub struct ProductPolynomial<F: PrimeField> {
    pub polynomials: Vec<MultilinearPolynomial<F>>,
}

impl<F: PrimeField> ProductPolynomial<F> {
    pub fn new(polynomials: Vec<MultilinearPolynomial<F>>) -> Self {
        // get the number of variables of the first multilinear polynomial
        // then iterate through all the polynomials and check if they have the same number of variables
        // assert if their number of variables are not the same
        let num_of_variables = polynomials[0].number_of_variables();
        assert!(
            polynomials
                .iter()
                .all(|polynomial| polynomial.number_of_variables() == num_of_variables),
            "different number of variables"
        );

        Self { polynomials }
    }

    pub fn evaluate(&self, values: &Vec<F>) -> F {
        let mut result = F::one();

        for polynomial in self.polynomials.iter() {
            result *= polynomial.evaluate(&values);
        }

        result
    }

    pub fn partial_evaluate(
        &self,
        evaluating_variable: usize,
        value: F,
    ) -> Vec<MultilinearPolynomial<F>> {
        let mut evaluated_polynomials: Vec<MultilinearPolynomial<F>> = Vec::new();

        for polynomial in self.polynomials.iter() {
            let partially_evaluated_polynomial = MultilinearPolynomial::partial_evaluate(
                &polynomial.evaluated_values,
                evaluating_variable,
                value,
            );

            evaluated_polynomials.push(partially_evaluated_polynomial);
        }

        evaluated_polynomials
    }

    // This function reduces the Vec of Multilinear polynomials to one Polynomial by
    // basically performing element-wise multiplication on the multilinear polynomials that makes up the ProductPolynomial
    pub fn multiply_polynomials_element_wise(&self) -> MultilinearPolynomial<F> {
        assert!(
            self.polynomials.len() > 1,
            "more than one polynomial required for mul operation"
        );

        let mut resultant_values = self.polynomials[0].evaluated_values.to_vec();

        for polynomial in self.polynomials.iter().skip(1) {
            for (i, value) in polynomial.evaluated_values.iter().enumerate() {
                resultant_values[i] *= value
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
        let poly1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(2)]); // 1 variable
        let poly2 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]); // 2 variables

        // This is expected to panic
        ProductPolynomial::new(vec![poly1, poly2]);
    }

    #[test]
    fn test_evaluate_product_poly() {
        let polynomail1 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let polynomials = vec![polynomail1, polynomail2];

        let product_polynomial = ProductPolynomial::new(polynomials);
        // a = 1, b = 2
        let values = vec![Fq::from(1), Fq::from(2)];

        assert_eq!(product_polynomial.evaluate(&values), Fq::from(24));
    }

    #[test]
    fn test_partial_evaluate_product_poly() {
        let polynomail1 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let polynomials = vec![polynomail1, polynomail2];
        let product_polynomial = ProductPolynomial::new(polynomials);

        let expect_poly1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(4)]);
        let expect_poly2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(6)]);
        let expected_partial_eval_result = vec![expect_poly1, expect_poly2];

        assert_eq!(
            product_polynomial.partial_evaluate(0, Fq::from(2)),
            expected_partial_eval_result
        );
    }

    #[test]
    fn test_multiply_polynomials_element_wise() {
        let polynomail1 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let polynomials = vec![polynomail1, polynomail2];

        let product_polynomial = ProductPolynomial::new(polynomials);
        let expected_product =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(6)]);

        assert_eq!(
            product_polynomial.multiply_polynomials_element_wise(),
            expected_product
        );
    }

    #[test]
    fn test_product_poly_degree() {
        let polynomail1 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 =
            MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let polynomials = vec![polynomail1, polynomail2];

        let product_polynomial = ProductPolynomial::new(polynomials);

        assert_eq!(product_polynomial.degree(), 2);
    }
}
