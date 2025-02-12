use ark_ff::PrimeField;
use crate::multilinear::evaluation_form::{MultilinearPolynomial, partial_evaluate};

pub struct ComposedMultilinearPolynomial<F: PrimeField> {
    pub evaluated_values: Vec<F>
}

pub fn evaluate<F: PrimeField>(polynomials: Vec<MultilinearPolynomial<F>>, values: Vec<F>) -> F {
    let mut result = F::one();

    for polynomial in polynomials.iter() {
        result *= polynomial.evaluate(values.clone());
    }

    result
}

// pub fn partial_evaluate<F: PrimeField>(polynomials: Vec<ComposedMultilinearPolynomial<F>>) {

// }