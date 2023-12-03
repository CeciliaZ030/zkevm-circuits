use eth_types::{Word, U256};
use halo2_proofs::circuit;

use super::{ExecStep, FixedCParams, CircuitsParams};
use crate::operation::RWCounter;

#[derive(Debug, Default, Clone)]
pub struct Chunk {
    /// current context 
    pub ctx: ChunkContext,
    /// fixed param for the chunk
    pub fixed_param: FixedCParams,
    /// Begin op of a chunk
    pub beginchunk: ExecStep,
    /// End op of a chunk
    pub endchunk: Option<ExecStep>,
    // Set in chunk_convert, used when converting the next chunk
    /// rw_table permutation fingerprint for this chunk
    pub rw_fingerprint: Word,
    /// rw_table permutation fingerprint for this chunk
    pub chrono_rw_fingerprint: Word,
}

/// Block-wise execution steps that don't belong to any Transaction.
#[derive(Debug, Default)]
pub struct ChunkSteps {
    /// Begin op of a chunk
    pub beginchunk: ExecStep,
    /// End op of a chunk
    pub endchunk: Option<ExecStep>,
}

/// Context of a [`ChunkContext`].
#[derive(Debug, Clone)]
pub struct ChunkContext {
    /// index of current chunk, start from 0
    pub idx: usize,
    /// Used to track the inner chunk counter in every operation in the chunk.
    /// Contains the next available value.
    pub rwc: RWCounter,
    /// number of chunks
    pub totalchunks: usize,
    /// initial rw counter
    pub initial_rwc: usize,
    /// end rw counter
    pub end_rwc: usize,
    ///
    pub is_dynamic: bool,
}

impl Default for ChunkContext {
    fn default() -> Self {
        Self::new(0, 1, false)
    }
}

impl ChunkContext {
    /// Create a new Self
    pub fn new(cur_idx: usize, totalchunks: usize, is_dynamic: bool) -> Self {
        Self {
            rwc: RWCounter::new(),
            idx: cur_idx,
            totalchunks,
            initial_rwc: 1, // rw counter start from 1
            end_rwc: 0,     // end_rwc should be set in later phase
            is_dynamic,
        }
    }

    /// new Self with one chunk
    pub fn new_onechunk() -> Self {
        Self {
            rwc: RWCounter::new(),
            idx: 0,
            totalchunks: 1,
            initial_rwc: 1, // rw counter start from 1
            end_rwc: 0,     // end_rwc should be set in later phase
            is_dynamic: false
        }
    }

    ///
    pub fn next(&mut self, initial_rwc: usize) {
        assert!(self.idx + 1 < self.totalchunks, "Exceed total chunks");
        self.idx += 1;
        self.initial_rwc = initial_rwc;
        self.end_rwc = 0;
    }

    /// is first chunk
    pub fn is_firstchunk(&self) -> bool {
        self.idx == 0
    }

    /// is last chunk
    pub fn is_lastchunk(&self) -> bool {
        self.totalchunks - self.idx - 1 == 0
    }
}
