//! Properties, that we can test:
//! 1. Chunk boundaries must not coincide with transaction boundaries. This means that we can
//! never have:
//!     1.1. `ExecStep::BeginChunk` followed by `ExecState::BeginTx`(maybe preceded
//! by `ExecState::EndChunk`)
//!     1.2 `ExecState::EndTx` followed by
//! `ExecState::EndChunk` (and then maybe folowed by `ExecState::BeginChunk`).
//! 2. The first (0th) chunk is not initiated with `BeginChunk`.
//! 3. The last (`total_chunk`th) chunk is not terminated by `EndChunk`.
//! 4. COUNT(`BeginChunk`) == COUNT(`EndChunk`) == `total_chunks - 1`.
//! 5. `BeginChunk` and `EndChunk` must alternate, i.e. there must never be two occurences of
//! one of them without the other inbetween.
//! 6. The sequence in 5. must start with `EndChunk` and end with `BeginChunk`.
//! 7. Between any pair of `BeginChunk`and `EndChunk` there is at least one `BeginTx`` and one
//! `EndTx``. (Is this always true or can we have mega transactions that emcompasses a whole
//! chunk? )
//!
//! ! This module tests that the chunking mechanism.

use std::{convert::identity, f32::consts::E};

use crate::{mock::{self, BlockData}, circuit_input_builder::{input_state_ref, MAX_CHUNK_SIZE, ExecStep, ExecState}};
use ::mock::{eth, MockTransaction, TestContext};
use eth_types::{address, geth_types::GethData, Transaction, Word, U256, U64};

#[test]
fn test_chunck() {
    //! Create single block with 1024 transactions.
    //! Check that correctly create four chunks of [300, 300, 300, 124] transactions respectively.
    pub const N_TRANSACTIONS: usize = 1024;
    pub const TOTAL_CHUNK_COUNT: usize = (N_TRANSACTIONS + MAX_CHUNK_SIZE - 1) / MAX_CHUNK_SIZE;
    let gas_cost = Word::from(53000);
    let required_gas = (gas_cost * N_TRANSACTIONS).try_into().unwrap();

    let mock_block = TestContext::<2, N_TRANSACTIONS>::new(
        None,
        |accs| {
            accs[0]
                .address(address!("0x0000000000000000000000000000000000000000"))
                .balance(eth(0));
            accs[1]
                .address(address!("0x000000000000000000000000000000000cafe001"))
                .balance(eth(required_gas))
                .nonce(100);
        },
        |mut txs, accs| {
            for tx in txs {
                tx.from(accs[1].address).to(accs[0].address).gas(Word::from(gas_cost));
            }
        },
        |block, _tx| block.number(0xcafeu64),
    ).unwrap();

    // Handles
    let gd: GethData = mock_block.into();
    let bd = mock::BlockData::new_from_geth_data(gd);
    let ci = mock::BlockData::new_circuit_input_builder(&bd);
    let cctxs: Vec<crate::circuit_input_builder::ChunkContext> = ci.chunk_ctxs;
    let txs = ci.block.txs();
    dbg!(cctxs.clone());

    let total_rw_count: usize = txs.iter().map(|tx| tx.steps().len()).sum();

    // Chunknomics
    // Verify the amount of chunks generated
    assert_eq!(TOTAL_CHUNK_COUNT, cctxs.len());

    // Verify the RWCounter of each block.
    for current_chunk_ctx in cctxs.iter() {
        assert!(usize::from(current_chunk_ctx.rwc) < MAX_CHUNK_SIZE);
    }

    // Create a list of pairs: (tx#, ExecStep)
    // Todo: add chunk index (chunk#, tx#, ExecStep)
    let exec_steps = txs
        .iter()
        .enumerate()
        .flat_map(|(i, tx)| tx.steps().iter().map(move |t| (i.to_owned(), t)));
    let curr_tx = txs.iter();
    let curr_chunk_ctx = cctxs.iter();

    // Chunkspection
    let mut begin_chunk_count = 0;
    let mut end_chunk_count = 0;
    // We must start with matching `EndChunk` so we initialize with `BeginChunk`.
    let mut prev_chunk_label = ExecState::BeginChunk;

    for (i, es) in exec_steps {
        match (es.exec_state.to_owned(), prev_chunk_label.to_owned()) {
            (ExecState::BeginChunk, ExecState::EndChunk) => {
                begin_chunk_count += 1;
                prev_chunk_label = es.exec_state.to_owned();
            }
            (ExecState::BeginChunk, _) => assert!(false, "Property 2 or 5 violated."),
            (ExecState::EndChunk, ExecState::BeginChunk) => {
                begin_chunk_count += 1;
                prev_chunk_label = es.exec_state.to_owned();
            }
            (ExecState::EndChunk, _) => assert!(false, "Property 2 or 5 violated."),
            _ => (),
        }
    }
    assert_eq!(
        prev_chunk_label,
        ExecState::EndChunk,
        "Property 6 violated."
    );

    // Property 4.
    assert_eq!(begin_chunk_count, TOTAL_CHUNK_COUNT - 1);
    assert_eq!(end_chunk_count, TOTAL_CHUNK_COUNT - 1);

}
