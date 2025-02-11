use ark_ff::PrimeField;

pub struct ComposedMultilinearPolynomial<F: PrimeField> {
    pub evaluated_values: Vec<F>
}

