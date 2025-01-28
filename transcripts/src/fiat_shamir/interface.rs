use ark_ff::PrimeField;

pub trait FiatShamirTranscriptInterface {
    fn new() -> Self;
    fn append(&mut self, incoming_data: &[u8]);
    fn sample_random_challenge(&mut self) -> [u8; 32];
    fn random_challenge_as_field_element<F: PrimeField>(&mut self) -> F;
}