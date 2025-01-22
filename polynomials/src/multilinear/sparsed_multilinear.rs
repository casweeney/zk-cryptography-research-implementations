use ark_ff::PrimeField;

struct SparsedMultilinearPolynomial<F: PrimeField> {
    coefficient: F,
    variables: Vec<()>
}