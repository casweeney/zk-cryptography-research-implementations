use ark_ff::PrimeField;
use circuit::arithmetic_circuit::Circuit;
use polynomials::{
    composed::{product_polynomial::ProductPolynomial, sum_polynomial::SumPolynomial},
    multilinear::evaluation_form::MultilinearPolynomial,
};

pub fn compute_fbc_polynomial<F: PrimeField>(
    add_i_bc: MultilinearPolynomial<F>,
    mul_i_bc: MultilinearPolynomial<F>,
    w_b_polynomial: &MultilinearPolynomial<F>,
    w_c_polynomial: &MultilinearPolynomial<F>,
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
    // The goal is to remove the "a" variable, so that we get add_i_bc and mul_i_bc
    // The random challenges array (rb_values and rc_values) are based on the layer bits: =>
    // => Eg: layer2 will have 2 bits value for variable a => rb_values and rc_values array length will be 2 each

    // We first evaluated at with random values at 0 index, so that we don't have to use .clone()
    // rb => random challenges for b, rc => random challenges for c
    let mut add_rb_bc =
        MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rb_values[0]);
    let mut add_rc_bc =
        MultilinearPolynomial::partial_evaluate(&add_i_abc.evaluated_values, 0, rc_values[0]);

    let mut mul_rb_bc =
        MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, rb_values[0]);
    let mut mul_rc_bc =
        MultilinearPolynomial::partial_evaluate(&mul_i_abc.evaluated_values, 0, rc_values[0]);

    for rb in rb_values.iter().skip(1) {
        add_rb_bc = MultilinearPolynomial::partial_evaluate(&add_rb_bc.evaluated_values, 0, *rb);
        mul_rb_bc = MultilinearPolynomial::partial_evaluate(&mul_rb_bc.evaluated_values, 0, *rb);
    }

    for rc in rc_values.iter().skip(1) {
        add_rc_bc = MultilinearPolynomial::partial_evaluate(&add_rc_bc.evaluated_values, 0, *rc);
        mul_rc_bc = MultilinearPolynomial::partial_evaluate(&mul_rc_bc.evaluated_values, 0, *rc);
    }

    let new_add_i = MultilinearPolynomial::add_polynomials(
        &add_rb_bc.scalar_mul(alpha),
        &add_rc_bc.scalar_mul(beta),
    );
    let new_mul_i = MultilinearPolynomial::add_polynomials(
        &mul_rb_bc.scalar_mul(alpha),
        &mul_rc_bc.scalar_mul(beta),
    );

    (new_add_i, new_mul_i)
}

pub fn evaluate_wb_wc<F: PrimeField>(
    wb_poly: &MultilinearPolynomial<F>,
    wc_poly: &MultilinearPolynomial<F>,
    sumcheck_challenges: &[F],
) -> (F, F) {
    let middle = sumcheck_challenges.len() / 2;
    let (rb_values, rc_values) = sumcheck_challenges.split_at(middle);

    let wb_poly_evaluated = wb_poly.evaluate(rb_values);
    let wc_poly_evaluated = wc_poly.evaluate(rc_values);

    (wb_poly_evaluated, wc_poly_evaluated)
}

pub fn compute_verifier_initial_claim<F: PrimeField>(
    circuit: &mut Circuit<F>,
    layer_index: usize,
    initial_random_challenge: F,
    sumcheck_challenges: &[F],
    wb_evaluation: F,
    wc_evaluation: F,
) -> F {
    let (add_i_abc, mul_i_abc) = circuit.add_i_and_mul_i_mle(layer_index);

    let (add_i_bc, mul_i_bc) = (
        MultilinearPolynomial::partial_evaluate(
            &add_i_abc.evaluated_values,
            0,
            initial_random_challenge,
        ),
        MultilinearPolynomial::partial_evaluate(
            &mul_i_abc.evaluated_values,
            0,
            initial_random_challenge,
        ),
    );

    let add_i_r = add_i_bc.evaluate(&sumcheck_challenges);
    let mul_i_r = mul_i_bc.evaluate(&sumcheck_challenges);

    (add_i_r * (wb_evaluation + wc_evaluation)) + (mul_i_r * (wb_evaluation * wc_evaluation))
}

pub fn compute_verifier_folded_claim<F: PrimeField>(
    circuit: &mut Circuit<F>,
    layer_index: usize,
    current_sumcheck_challenges: &[F],
    previous_sumcheck_challenges: &[F],
    wb_evaluation: F,
    wc_evaluation: F,
    alpha: F,
    beta: F,
) -> F {
    let (prev_rb, prev_rc) =
        previous_sumcheck_challenges.split_at(previous_sumcheck_challenges.len() / 2);

    let (add_i_abc, mul_i_abc) = circuit.add_i_and_mul_i_mle(layer_index);

    let (new_add_i, new_mul_i) =
        compute_new_add_i_mul_i(alpha, beta, add_i_abc, mul_i_abc, prev_rb, prev_rc);

    let add_r = new_add_i.evaluate(&current_sumcheck_challenges);
    let mul_r = new_mul_i.evaluate(&current_sumcheck_challenges);

    (add_r * (wb_evaluation + wc_evaluation)) + (mul_r * (wb_evaluation * wc_evaluation))
}
