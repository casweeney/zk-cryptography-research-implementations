use ark_ff::PrimeField;
use circuit::arithmetic_circuit::Circuit;
use sumcheck_protocol::gkr_sumcheck::gkr_sumcheck_protocol::{GKRSumcheck, GKRSumcheckProverProof};
use polynomials::{
    composed::{product_polynomial::ProductPolynomial, sum_polynomial::SumPolynomial},
    multilinear::evaluation_form::MultilinearPolynomial
};
use transcripts::fiat_shamir::fiat_shamir_transcript::Transcript;

pub struct Proof<F: PrimeField> {
    claimed_sum: F,
    sumcheck_proofs: Vec<GKRSumcheckProverProof<F>>,
}

fn circuit_preprocessing() {
    //  should return add_i and mul_i for each layer
}

// pub fn prove(circuit) {

// }

// pub fn verify(circuit) {

// }

pub fn compute_fbc_polynomial<F: PrimeField>(
    add_i_bc: MultilinearPolynomial<F>,
    mul_i_bc: MultilinearPolynomial<F>,
    w_b_polynomial: MultilinearPolynomial<F>,
    w_c_polynomial: MultilinearPolynomial<F>
) -> SumPolynomial<F> {
    let add_wbc = MultilinearPolynomial::polynomial_tensor_add(&w_b_polynomial, &w_c_polynomial);
    let mul_wbc = MultilinearPolynomial::polynomial_tensor_mul(&w_b_polynomial, &w_c_polynomial);

    let add_i_term = ProductPolynomial::new(vec![add_i_bc, add_wbc]);
    let mul_i_term = ProductPolynomial::new(vec![mul_i_bc, mul_wbc]);

    SumPolynomial::new(vec![add_i_term, mul_i_term])
}

pub fn compute_new_add_i_mul_i<F: PrimeField>(
    alpha: F,
    beta: F,
    add_i_abc: MultilinearPolynomial<F>,
    mul_i_abc: MultilinearPolynomial<F>,
    rb_values: Vec<F>,
    rc_values: Vec<F>,
) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {

    let mut add_rb_bc = MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rb_values[0]);
    let mut mul_rb_bc = MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, rb_values[0]);

    let mut add_rc_bc = MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rc_values[0]);
    let mut mul_rc_bc = MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, rc_values[0]);

    for rb in rb_values.iter().skip(1) {
        add_rb_bc = MultilinearPolynomial::partial_evaluate(&add_rb_bc.evaluated_values, 0, *rb);
        mul_rb_bc = MultilinearPolynomial::partial_evaluate(&mul_rb_bc.evaluated_values, 0, *rb);
    }

    for rc in rc_values.iter().skip(1) {
        add_rc_bc = MultilinearPolynomial::partial_evaluate(&add_rc_bc.evaluated_values, 0, *rc);
        mul_rc_bc = MultilinearPolynomial::partial_evaluate(&mul_rc_bc.evaluated_values, 0, *rc);
    }

    let new_add_i = MultilinearPolynomial::add_polynomials(&add_rb_bc.scalar_mul(alpha), &add_rc_bc.scalar_mul(beta));
    let new_mul_i = MultilinearPolynomial::add_polynomials(&mul_rb_bc.scalar_mul(alpha), &mul_rc_bc.scalar_mul(beta));

    (new_add_i, new_mul_i)
}