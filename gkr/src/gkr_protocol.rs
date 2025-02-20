use ark_ff::PrimeField;
use circuit::arithmetic_circuit::Circuit;
use sumcheck_protocol::gkr_sumcheck::gkr_sumcheck_protocol::{GKRSumcheck, GKRSumcheckProverProof};
use polynomials::{
    composed::{product_polynomial::ProductPolynomial, sum_polynomial::SumPolynomial},
    multilinear::evaluation_form::MultilinearPolynomial
};
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};

pub struct Proof<F: PrimeField> {
    pub claimed_output: F,
    pub sumcheck_proofs: Vec<GKRSumcheckProverProof<F>>,
}

fn circuit_preprocessing() {
    //  should return add_i and mul_i for each layer
}

pub fn prove<F: PrimeField>(circuit: &mut Circuit<F>, initial_claimed_sum: F) -> Proof<F> {
    let mut transcript = Transcript::new();
    let mut layer_proofs = Vec::new();

    let (add_i_abc_polynomial, mul_i_abc_polynomial) = circuit.add_i_and_mul_i_mle(0);
    let mut w0_polynomial = circuit.w_i_polynomial(0);

    if w0_polynomial.evaluated_values.len() == 1 {
        let mut w0_padded_with_zero = w0_polynomial.evaluated_values;
        w0_padded_with_zero.push(F::zero());
        w0_polynomial = MultilinearPolynomial::new(&w0_padded_with_zero);
    }

    transcript.append(&w0_polynomial.convert_to_bytes());
    let random_challenge_a: F = transcript.random_challenge_as_field_element(); // ra
    let claimed_sum = w0_polynomial.evaluate(&vec![random_challenge_a]); // m0

    let add_i_bc = MultilinearPolynomial::partial_evaluate(&add_i_abc_polynomial.evaluated_values, 0, random_challenge_a);
    let mul_i_bc = MultilinearPolynomial::partial_evaluate(&mul_i_abc_polynomial.evaluated_values, 0, random_challenge_a);

    let wb_poly = circuit.w_i_polynomial(1);
    let wc_poly = circuit.w_i_polynomial(1);

    let w0_fbc = compute_fbc_polynomial(add_i_bc, mul_i_bc, &wb_poly, &wc_poly);

    let mut sumcheck = GKRSumcheck::init(w0_fbc);
    let sumcheck_proof = sumcheck.prove(claimed_sum);
    layer_proofs.push(sumcheck_proof);

    // Handling subsequent layers
    for layer_index in 1..circuit.layers.len() {
        let (add_i_abc_polynomial, mul_i_abc_polynomial) = circuit.add_i_and_mul_i_mle(layer_index);
        
        // use the randomness from the sumcheck proof, split into two vec! for rb and rc
        let previous_layer_challenges = &layer_proofs[layer_index - 1].random_challenges;
        let mid = previous_layer_challenges.len() / 2;
        let (rb_values, rc_values) = previous_layer_challenges.split_at(mid);

        transcript.append(&wb_poly.convert_to_bytes());
        transcript.append(&wc_poly.convert_to_bytes());

        let alpha: F = transcript.random_challenge_as_field_element();
        let beta: F = transcript.random_challenge_as_field_element();

        let (new_add_i, new_mul_i) = compute_new_add_i_mul_i(
            alpha,
            beta,
            add_i_abc_polynomial,
            mul_i_abc_polynomial,
            rb_values.to_vec(),
            rc_values.to_vec()
        );

        // get wb and wc of the next layer
        let next_wb_poly = circuit.w_i_polynomial(layer_index + 1);
        let next_wc_poly = circuit.w_i_polynomial(layer_index + 1);

        let layer_fbc = compute_fbc_polynomial(
            new_add_i,
            new_mul_i,
            &next_wb_poly,
            &next_wc_poly
        );

        let mut sumcheck = GKRSumcheck::init(layer_fbc);

        // Get claimed sum using linear combination form
        let claimed_sum = alpha * wb_poly.evaluate(&rb_values.to_vec()) + beta * wc_poly.evaluate(&rc_values.to_vec());

        let layer_proof = sumcheck.prove(claimed_sum);

        layer_proofs.push(layer_proof);
    }

    Proof {
        sumcheck_proofs: layer_proofs,
        claimed_output: initial_claimed_sum
    }
}

// pub fn verify(circuit) {

// }

pub fn compute_fbc_polynomial<F: PrimeField>(
    add_i_bc: MultilinearPolynomial<F>,
    mul_i_bc: MultilinearPolynomial<F>,
    w_b_polynomial: &MultilinearPolynomial<F>,
    w_c_polynomial: &MultilinearPolynomial<F>
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