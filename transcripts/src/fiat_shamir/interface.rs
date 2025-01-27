pub trait FiatShamirTranscriptInterface {
    fn new() -> Self;
    fn append(&mut self, incoming_data: &[u8]);
    fn sample_random_challenge(&mut self) -> [u8; 32];
}