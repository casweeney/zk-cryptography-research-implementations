use criterion::{criterion_group, criterion_main, Criterion};
use ark_bls12_381::Fr;
use circuit::arithmetic_circuit::{Gate, Layer, Operator, Circuit};
use gkr::gkr_protocol::{prove, verify};

fn gkr_protocol_benchmark(c: &mut Criterion) {
    let gate1 = Gate::new(0, 1, 0, Operator::Mul);
    let gate2 = Gate::new(0, 1, 0, Operator::Add);
    let gate3 = Gate::new(2, 3, 1, Operator::Mul);

    let layer0 = Layer::new(vec![gate1]);
    let layer1 = Layer::new(vec![gate2, gate3]);

    let mut circuit = Circuit::<Fr>::new(vec![layer0, layer1]);
    let inputs = vec![Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(5)];

    c.bench_function("gkr_protocol_benchmark", |b| {
        b.iter(|| {
            let proof = prove(&mut circuit, &inputs);

            assert_eq!(verify(&mut circuit, proof, &inputs), true);
        });
    });
}

criterion_group!(benches, gkr_protocol_benchmark);
criterion_main!(benches);