use ark_ff::PrimeField;
use crate::multilinear::evaluation_form::{MultilinearPolynomial, partial_evaluate};

pub struct ProductPolynomial<F: PrimeField> {
    pub polynomials: Vec<MultilinearPolynomial<F>>
}

pub struct SumPolynomial<F: PrimeField> {
    pub polynomials: Vec<MultilinearPolynomial<F>>
}

impl <F: PrimeField>ProductPolynomial<F> {
    pub fn new(polynomials: Vec<MultilinearPolynomial<F>>) -> Self {
        Self {
            polynomials
        }
    }
    
    pub fn evaluate_product_poly(&self, values: &Vec<F>) -> F {
        let mut result = F::one();

        for polynomial in self.polynomials.iter() {
            result *= polynomial.evaluate(&values);
        }

        result
    }
    
    pub fn partial_evaluate_product_poly(polynomials: Vec<MultilinearPolynomial<F>>) -> Vec<MultilinearPolynomial<F>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_evaluate_product_poly() {
        let polynomail1 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
        let polynomail2 = MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);

        let polynomials = vec![polynomail1, polynomail2];

        let product_polynomial = ProductPolynomial::new(polynomials);
        let values = vec![Fq::from(1), Fq::from(2)];

        assert_eq!(product_polynomial.evaluate_product_poly(&values), Fq::from(24));
    }
}