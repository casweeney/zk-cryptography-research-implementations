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
    
    pub fn evaluate_product_poly(polynomials: Vec<MultilinearPolynomial<F>>, values: Vec<F>) -> F {
        let mut result = F::one();

        for polynomial in polynomials.iter() {
            result *= polynomial.evaluate(&values);
        }

        result
    }
    
    pub fn partial_evaluate_product_poly(polynomials: Vec<MultilinearPolynomial<F>>) {
        todo!()
    }
}