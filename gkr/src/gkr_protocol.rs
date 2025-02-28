use ark_ff::PrimeField;
use circuit::arithmetic_circuit::Circuit;
use sumcheck_protocol::gkr_sumcheck::sumcheck_gkr_protocol::{
    SumcheckProverProof,
    prove as sumcheck_prove,
    verify as sumcheck_verify,
    field_element_to_bytes
};
use polynomials::{
    composed::{product_polynomial::ProductPolynomial, sum_polynomial::SumPolynomial},
    multilinear::evaluation_form::MultilinearPolynomial
};
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub claimed_output: F,
    pub sumcheck_proofs: Vec<SumcheckProverProof<F>>,
    pub wb_evals: Vec<F>,
    pub wc_evals: Vec<F>
}

pub fn prove<F: PrimeField>(circuit: &mut Circuit<F>, inputs: &[F]) -> Proof<F> {
    circuit.evaluate(inputs.to_vec());

    let mut transcript = Transcript::new();
    let mut layer_proofs = Vec::new();
    let mut wb_evals= Vec::new();
    let mut wc_evals = Vec::new();
    let mut alpha = F::zero();
    let mut beta = F::zero();
    let mut rb_values = Vec::new();
    let mut rc_values = Vec::new();

    // handling layer 0 computation
    let mut w0_polynomial = circuit.w_i_polynomial(0);

    if w0_polynomial.evaluated_values.len() == 1 {
        let mut w0_padded_with_zero = w0_polynomial.evaluated_values;
        w0_padded_with_zero.push(F::zero());
        w0_polynomial = MultilinearPolynomial::new(&w0_padded_with_zero);
    }

    transcript.append(&w0_polynomial.convert_to_bytes());
    let random_challenge_a: F = transcript.random_challenge_as_field_element(); // ra
    let mut claimed_sum = w0_polynomial.evaluate(&vec![random_challenge_a]); // m0


    // Handling subsequent layers
    for layer_index in 0..circuit.layers.len() {
        let (add_i_abc_polynomial, mul_i_abc_polynomial) = circuit.add_i_and_mul_i_mle(layer_index);

        let (add_i_bc, mul_i_bc) = if layer_index == 0 {
            (
                MultilinearPolynomial::partial_evaluate(&add_i_abc_polynomial.evaluated_values, 0, random_challenge_a),
                MultilinearPolynomial::partial_evaluate(&mul_i_abc_polynomial.evaluated_values, 0, random_challenge_a)
            )
        } else {
            compute_new_add_i_mul_i(
                alpha,
                beta,
                add_i_abc_polynomial,
                mul_i_abc_polynomial,
                &rb_values,
                &rc_values
            )
        };

        let wb_poly = circuit.w_i_polynomial(layer_index + 1);
        let wc_poly = wb_poly.clone();

        let fbc_polynomial = compute_fbc_polynomial(add_i_bc, mul_i_bc, &wb_poly, &wc_poly);

        let sumcheck_proof = sumcheck_prove(fbc_polynomial, claimed_sum, &mut transcript);
        layer_proofs.push(sumcheck_proof.clone());
        
        if layer_index < circuit.layers.len() - 1 {
            let sumcheck_challenges = sumcheck_proof.random_challenges;

            // Evaluate wb and wc to be used by verifier
            let (wb_evaluation, wc_evaluation) = evaluate_wb_wc(&wb_poly, &wc_poly, &sumcheck_challenges);

            wb_evals.push(wb_evaluation);
            wc_evals.push(wc_evaluation);

            // use the randomness from the sumcheck proof, split into two vec! for rb and rc
            let middle = sumcheck_challenges.len() / 2;
            let (current_rb_values, current_rc_values) = sumcheck_challenges.split_at(middle);
            rb_values = current_rb_values.to_vec();
            rc_values = current_rc_values.to_vec();

            transcript.append(&field_element_to_bytes(wb_evaluation));
            alpha = transcript.random_challenge_as_field_element();

            transcript.append(&field_element_to_bytes(wc_evaluation));
            beta = transcript.random_challenge_as_field_element();

            // Compute claimed sum using linear combination form
            claimed_sum = (alpha * wb_evaluation) + (beta * wc_evaluation);
        }
    }

    Proof {
        sumcheck_proofs: layer_proofs,
        claimed_output: claimed_sum,
        wb_evals,
        wc_evals
    }
}

pub fn verify<F: PrimeField>(circuit: &mut Circuit<F>, proof: Proof<F>, inputs: &[F]) -> bool {
    let mut transcript = Transcript::new();
    let mut alpha = F::zero();
    let mut beta = F::zero();
    let mut prev_sumcheck_challenges = Vec::new();

    // layer 0 computation
    let mut w0_polynomial = circuit.w_i_polynomial(0);

    if w0_polynomial.evaluated_values.len() == 1 {
        let mut w0_padded_with_zero = w0_polynomial.evaluated_values;
        w0_padded_with_zero.push(F::zero());
        w0_polynomial = MultilinearPolynomial::new(&w0_padded_with_zero);
    }

    transcript.append(&w0_polynomial.convert_to_bytes());
    let random_challenge_a: F = transcript.random_challenge_as_field_element();

    let mut claimed_sum = w0_polynomial.evaluate(&vec![random_challenge_a]);

    for layer_index in 0..circuit.layers.len() {
        if claimed_sum != proof.sumcheck_proofs[layer_index].claimed_sum {
            return false;
        }

        // Get the verification result
        let verify_result = sumcheck_verify(&proof.sumcheck_proofs[layer_index], &mut transcript);
        if !verify_result.is_proof_valid {
            return false;
        }

        let sumcheck_challenges = verify_result.random_challenges;

        let (wb_evaluation, wc_evaluation) = if layer_index < circuit.layers.len() - 1 {
            (proof.wb_evals[layer_index], proof.wc_evals[layer_index])
        } else {
            let wb_poly = MultilinearPolynomial::new(&inputs);
            let wc_poly = wb_poly.clone();

            evaluate_wb_wc(&wb_poly, &wc_poly, &sumcheck_challenges)
        };

        let expected_claim = if layer_index == 0 {
            compute_verifier_initial_claim(
                circuit,
                layer_index,
                random_challenge_a,
                &sumcheck_challenges,
                wb_evaluation,
                wc_evaluation
            )
        } else {
            compute_verifier_folded_claim(
                circuit,
                layer_index,
                &sumcheck_challenges,
                &prev_sumcheck_challenges,
                wb_evaluation,
                wc_evaluation,
                alpha,
                beta
            )
        };

        if expected_claim != verify_result.last_claimed_sum {
            return false;
        }

        prev_sumcheck_challenges = sumcheck_challenges.to_vec();

        transcript.append(&field_element_to_bytes(wb_evaluation));
        alpha = transcript.random_challenge_as_field_element();

        transcript.append(&field_element_to_bytes(wc_evaluation));
        beta = transcript.random_challenge_as_field_element();

        claimed_sum = (alpha * wb_evaluation) + (beta * wc_evaluation);
    }

    true
}

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
    rb_values: &[F],
    rc_values: &[F],
) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
    // Partial evaluating add_i_abc and mul_i_abc at all the random values using loop
    // We first evaluated at with random values at 0 index, so that we don't have to clone
    // rb => random challenges for b, rc => random challenges for c
    let mut add_rb_bc = MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rb_values[0]);
    let mut add_rc_bc = MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rc_values[0]);

    let mut mul_rb_bc = MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, rb_values[0]);
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

pub fn evaluate_wb_wc<F: PrimeField>(
    wb_poly: &MultilinearPolynomial<F>,
    wc_poly: &MultilinearPolynomial<F>,
    sumcheck_challenges: &[F]
) -> (F, F) {
    let middle = sumcheck_challenges.len() / 2;
    let (rb_values, rc_values) = sumcheck_challenges.split_at(middle);

    let wb_poly_evaluated = wb_poly.evaluate(rb_values);
    let wc_poly_evaluated = wc_poly.evaluate(rc_values);

    (wb_poly_evaluated, wc_poly_evaluated)
}

pub fn compute_verifier_initial_claim<F: PrimeField> (
    circuit: &mut Circuit<F>,
    layer_index: usize,
    initial_random_challenge: F,
    sumcheck_challenges: &[F],
    wb_evaluation: F,
    wc_evaluation: F

) -> F {
    let (add_i_abc, mul_i_abc) = circuit.add_i_and_mul_i_mle(layer_index);

    let (add_i_bc, mul_i_bc) = (
        MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, initial_random_challenge),
        MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, initial_random_challenge)
    );

    let add_i_r = add_i_bc.evaluate(&sumcheck_challenges);
    let mul_i_r = mul_i_bc.evaluate(&sumcheck_challenges);

    (add_i_r * (wb_evaluation + wc_evaluation)) + (mul_i_r * (wb_evaluation * wc_evaluation))
}

pub fn compute_verifier_folded_claim<F: PrimeField> (
    circuit: &mut Circuit<F>,
    layer_index: usize,
    current_sumcheck_challenges: &[F],
    previous_sumcheck_challenges: &[F],
    wb_evaluation: F,
    wc_evaluation: F,
    alpha: F,
    beta: F
) -> F {
    let (prev_rb, prev_rc) = previous_sumcheck_challenges.split_at(previous_sumcheck_challenges.len() / 2);

    let (add_i_abc, mul_i_abc) = circuit.add_i_and_mul_i_mle(layer_index);

    let (new_add_i, new_mul_i) = compute_new_add_i_mul_i(
        alpha,
        beta,
        add_i_abc,
        mul_i_abc,
        prev_rb,
        prev_rc
    );


    let add_r = new_add_i.evaluate(&current_sumcheck_challenges);
    let mul_r = new_mul_i.evaluate(&current_sumcheck_challenges);

    (add_r * (wb_evaluation + wc_evaluation)) + (mul_r * (wb_evaluation * wc_evaluation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;
    use circuit::arithmetic_circuit::{Gate, Layer, Operator};

    #[test]
    pub fn test_gkr_protocol1() {
        let gate1 = Gate::new(0, 1, 0, Operator::Mul);
        let gate2 = Gate::new(0, 1, 0, Operator::Add);
        let gate3 = Gate::new(2, 3, 1,  Operator::Mul);
        
        let layer0 = Layer::new(vec![gate1]);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1]);
        let inputs = vec![Fq::from(2), Fq::from(3), Fq::from(4), Fq::from(5)];

        let proof = prove(&mut circuit, &inputs);

        assert_eq!(verify(&mut circuit, proof, &inputs), true);
    }

    #[test]
    pub fn test_gkr_protocol2() {
        // Layer 0
        let gate1 = Gate::new(0, 1, 0, Operator::Add);
        let layer0 = Layer::new(vec![gate1]);

        // Layer 1
        let gate2 = Gate::new(0, 1, 0, Operator::Mul);
        let gate3 = Gate::new(2, 3, 1,  Operator::Add);
        let layer1 = Layer::new(vec![gate2, gate3]);

        let gate4 = Gate::new(0, 1, 0, Operator::Add);
        let gate5 = Gate::new(2, 3, 1, Operator::Add);
        let gate6 = Gate::new(4, 5, 2, Operator::Add);
        let gate7 = Gate::new(6, 7, 3, Operator::Add);
        let layer2 = Layer::new(vec![gate4, gate5, gate6, gate7]);

        let mut circuit = Circuit::<Fq>::new(vec![layer0, layer1, layer2]);
        let inputs = vec![
            Fq::from(1),
            Fq::from(2),
            Fq::from(3),
            Fq::from(4),
            Fq::from(5),
            Fq::from(6),
            Fq::from(7),
            Fq::from(8)
        ];

        let proof = prove(&mut circuit, &inputs);

        assert_eq!(verify(&mut circuit, proof, &inputs), true);
    }
}