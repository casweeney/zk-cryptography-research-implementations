use ark_ff::PrimeField;
use sha3::{Digest, Keccak256};
use crate::fiat_shamir::interface::FiatShamirTranscriptInterface;

pub struct Transcript {
    hasher: Keccak256,
}

impl FiatShamirTranscriptInterface for Transcript {
    fn new() -> Self {
        Self {
            hasher: Keccak256::new()
        }
    }

    fn append(&mut self, incoming_data: &[u8]) {
        self.hasher.update(incoming_data);
    }

    fn sample_random_challenge(&mut self) -> [u8; 32] {

        todo!()
    }
}