use criterion::{criterion_group, criterion_main, Criterion};
use ark_bls12_381::{Bls12_381, Fr};
use circuit::arithmetic_circuit::{Gate, Layer, Operator, Circuit};
use multilinear_kzg::trusted_setup::TrustedSetup;
use gkr::succinct_gkr_protocol::{prove_succinct, verify_succinct};

fn succinct_gkr_benchmark(c: &mut Criterion) {
    let gate1 = Gate::new(0, 1, 0, Operator::Mul);
    let gate2 = Gate::new(0, 1, 0, Operator::Add);
    let gate3 = Gate::new(2, 3, 1, Operator::Mul);

    let layer0 = Layer::new(vec![gate1]);
    let layer1 = Layer::new(vec![gate2, gate3]);

    let mut circuit = Circuit::<Fr>::new(vec![layer0, layer1]);
    let inputs = vec![Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(5)];

    let taus = vec![Fr::from(5), Fr::from(2)];
    let trusted_setup = TrustedSetup::<Bls12_381>::initialize_setup(&taus);

    c.bench_function("succinct_gkr_benchmark", |b| {
        b.iter(|| {
            let succinct_proof = prove_succinct(&mut circuit, &inputs, &trusted_setup);

            assert_eq!(
                verify_succinct(&mut circuit, succinct_proof, &trusted_setup),
                true
            );
        });
    });
}

criterion_group!(benches, succinct_gkr_benchmark);
criterion_main!(benches);