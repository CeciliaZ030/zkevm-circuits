use super::{ExecStep, FixedCParams};
use crate::operation::RWCounter;

#[derive(Debug, Default, Clone)]
pub struct Chunk {
    /// current context
    pub ctx: ChunkContext,
    /// fixed param for the chunk
    pub fixed_param: FixedCParams,
    /// Begin op of a chunk
    pub begin_chunk: Option<ExecStep>,
    /// End op of a chunk
    pub end_chunk: Option<ExecStep>,
}

/// Context of chunking, used to track the current chunk index and inner rw counter
/// also the global rw counter from start to end.
#[derive(Debug, Clone)]
pub struct ChunkContext {
    /// Index of current chunk, start from 0
    pub idx: usize,
    /// Used to track the inner chunk counter in every operation in the chunk.
    /// Contains the next available value.
    pub rwc: RWCounter,
    /// Number of chunks
    pub total_chunks: usize,
    /// Initial global rw counter
    pub initial_rwc: usize,
    /// End global rw counter
    pub end_rwc: usize,
    /// If this block is chunked dynamically
    pub is_dynamic: bool,
}

impl Default for ChunkContext {
    fn default() -> Self {
        Self::new(0, 1, false)
    }
}

impl ChunkContext {
    /// Create a new Self
    pub fn new(cur_idx: usize, total_chunks: usize, is_dynamic: bool) -> Self {
        Self {
            rwc: RWCounter::new(),
            idx: cur_idx,
            total_chunks,
            initial_rwc: 1, // rw counter start from 1
            end_rwc: 0,     // end_rwc should be set in later phase
            is_dynamic,
        }
    }

    /// New chunking context with one chunk
    pub fn new_one_chunk() -> Self {
        Self {
            rwc: RWCounter::new(),
            idx: 0,
            total_chunks: 1,
            initial_rwc: 1, // rw counter start from 1
            end_rwc: 0,     // end_rwc should be set in later phase
            is_dynamic: false,
        }
    }

    /// Proceed the context to next chunk, record the initial rw counter
    /// update the chunk idx and reset the inner rw counter
    pub fn next(&mut self, initial_rwc: usize) {
        assert!(self.idx + 1 < self.total_chunks, "Exceed total chunks");
        self.idx += 1;
        self.rwc = RWCounter::new();
        self.initial_rwc = initial_rwc;
        self.end_rwc = 0;
    }

    /// Is first chunk
    pub fn is_first_chunk(&self) -> bool {
        self.idx == 0
    }

    /// Is last chunk
    pub fn is_last_chunk(&self) -> bool {
        self.total_chunks - self.idx - 1 == 0
    }
}
