use criterion::{criterion_group, criterion_main, Criterion};

fn densed_univariate_benchmark(c: &mut Criterion) {
    // Your benchmark code goes here
}

criterion_group!(benches, densed_univariate_benchmark);
criterion_main!(benches);