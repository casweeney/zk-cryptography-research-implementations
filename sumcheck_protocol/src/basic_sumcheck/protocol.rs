#[cfg(test)]
mod tests {
    use crate::basic_sumcheck::prover::Prover;
    use crate::basic_sumcheck::verifier::Verifier;
    // use ark_bls12_381::Fr;
    use field_tracker::{Ft, print_summary};
    type Fr = Ft!(ark_bls12_381::Fr);

    #[test]
    fn test_sumcheck_protocol_init() {
        let polynomial_evaluated_values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(2),
            Fr::from(7),
            Fr::from(3),
            Fr::from(3),
            Fr::from(6),
            Fr::from(11),
        ];

        let prover = Prover::init(&polynomial_evaluated_values);

        assert_eq!(prover.initial_claimed_sum, Fr::from(32));
        assert_eq!(prover.is_initialized, true);
    }

    #[test]
    fn test_sumcheck_protocol_prove_and_verify() {
        // let polynomial_evaluated_values = vec![
        //     Fr::from(0),
        //     Fr::from(0),
        //     Fr::from(2),
        //     Fr::from(7),
        //     Fr::from(3),
        //     Fr::from(3),
        //     Fr::from(6),
        //     Fr::from(11),
        // ];

        // Created a large array to test benchmarking
        let polynomial_evaluated_values = vec![
            Fr::from(3); 1 << 20
        ];

        let mut prover = Prover::init(&polynomial_evaluated_values);
        let proof = prover.prove();

        let mut verifier: Verifier<Fr> = Verifier::init();
        assert_eq!(verifier.is_initialized, true);

        assert_eq!(verifier.verify(proof), true);

        // print benchmark summary
        print_summary!();
    }

    #[test]
    fn test_sumcheck_protocol_prove_and_verify2() {
        let polynomial_evaluated_values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
        ];

        let mut prover = Prover::init(&polynomial_evaluated_values);
        let proof = prover.prove();

        let mut verifier: Verifier<Fr> = Verifier::init();
        assert_eq!(verifier.is_initialized, true);

        assert_eq!(verifier.verify(proof), true);
    }

    #[test]
    fn test_sumcheck_protocol_prove_and_verify3() {
        let polynomial_evaluated_values = vec![
            Fr::from(1),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
            Fr::from(2),
            Fr::from(4),
            Fr::from(6),
            Fr::from(8),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
            Fr::from(9),
            Fr::from(4),
            Fr::from(6),
            Fr::from(8),
            Fr::from(10),
        ];

        let mut prover = Prover::init(&polynomial_evaluated_values);
        let proof = prover.prove();

        let mut verifier: Verifier<Fr> = Verifier::init();
        assert_eq!(verifier.is_initialized, true);

        assert_eq!(verifier.verify(proof), true);
    }
}
