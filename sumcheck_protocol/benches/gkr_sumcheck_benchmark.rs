use criterion::{criterion_group, criterion_main, Criterion};
use ark_bn254::Fq;
use sumcheck_protocol::gkr_sumcheck::sumcheck_gkr_protocol::{prove, verify};
use polynomials::{
    multilinear::evaluation_form::MultilinearPolynomial,
    composed::{product_polynomial::ProductPolynomial, sum_polynomial::SumPolynomial}
};
use transcripts::fiat_shamir::{
    fiat_shamir_transcript::Transcript,
    interface::FiatShamirTranscriptInterface,
};

fn gkr_sumcheck_benchmark(c: &mut Criterion) {
    let poly1a =
        MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
    let poly2a =
        MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
    let product_poly1 = ProductPolynomial::new(vec![poly1a, poly2a]);

    let poly1b =
        MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(2)]);
    let poly2b =
        MultilinearPolynomial::new(&vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3)]);
    let product_poly2 = ProductPolynomial::new(vec![poly1b, poly2b]);

    let sum_polynomial = SumPolynomial::new(vec![product_poly1, product_poly2]);

    let mut prover_transcript = Transcript::new();
    let mut verifier_transcript = Transcript::new();

    c.bench_function("gkr_sumcheck_benchmark", |b| {
        b.iter(|| {
            let result = prove(sum_polynomial.clone(), Fq::from(12), &mut prover_transcript);

            let verified = verify(&result, &mut verifier_transcript);

            assert_eq!(verified.is_proof_valid, true);
        });
    });
}

criterion_group!(benches, gkr_sumcheck_benchmark);
criterion_main!(benches);