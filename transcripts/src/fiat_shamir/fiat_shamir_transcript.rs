use sha3::{Keccak256, Digest};
use crate::fiat_shamir::interface::FiatShamirTranscriptInterface;

pub struct Transcript {
    hasher: Keccak256,
}

impl FiatShamirTranscriptInterface for Transcript {
    /// This function uses the new() associated function from Keccak256 to create a hash function which starts an initial state.
    /// It is in this initial state that the append() function will append data to.
    fn new() -> Self {
        Self {
            hasher: Keccak256::new()
        }
    }

    /// Takes in incoming_data as argument and uses the update() method from the hasher to incrementally update the state with the incoming_data
    /// The new data (incoming_data) is appended to the existing data in the state, and a new state is computed based on the combined data
    /// Note: The order of appending/updating/absorbing data matters: Hash(data1, data2) is different from Hash(data2, data1)
    /// incoming_data => sum of polynomial evaluated values (y_values) |OR| univariate polynomial in evaluated form
    fn append(&mut self, incoming_data: &[u8]) {
        self.hasher.update(incoming_data);
    }


    /// Generates and returns 32-byte random challenge by calling finalize_reset()
    /// Random challenge generated by computing current state of appended data
    /// Resets the state, then it appends the computed hash back to the hash function state so that future operation rounds can build on it
    fn sample_random_challenge(&mut self) -> [u8; 32] {
        let mut output_hash = [0; 32]; // fixed sized array of 32-bytes initially filled with zeros
        output_hash.copy_from_slice(&self.hasher.finalize_reset());
        self.hasher.update(output_hash);

        output_hash
    }
}