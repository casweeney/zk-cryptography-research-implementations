use criterion::{criterion_group, criterion_main, Criterion};
use ark_bls12_381::Fr;
use sumcheck_protocol::basic_sumcheck::{prover::Prover, verifier::Verifier};

fn basic_sumcheck_benchmark(c: &mut Criterion) {
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

    c.bench_function("basic_sumcheck_benchmark", |b| {
        b.iter(|| {
            let mut prover = Prover::init(&polynomial_evaluated_values);
            let proof = prover.prove();

            let mut verifier: Verifier<Fr> = Verifier::init();
            let verification = verifier.verify(proof);

            assert_eq!(verification, true);
        });
    });
}

criterion_group!(benches, basic_sumcheck_benchmark);
criterion_main!(benches);