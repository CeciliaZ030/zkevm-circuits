//! Development plan:
//!
//! - [ ] Copmpute the right RWC logic.  How should `total_rwc_count` be calcualted?
//! - [ ] Instantiate [`new_begin_chunk_step`] and [`new_end_chunk_step`] with right values.
//! - [ ] Print correct chunk number.
//! - [ ] Express the correct [`chunk_condition`] in [`circuit_input_builder`].
//! - [ ] Enable all disable tests and TODOs and variables starting with underscore.
//! - [ ] `cargo clippy`
//!
//! ------------------------------
//!
//! This module tests the chunking mechanism and produces inspectable print-outs.
//!
//! This test implements (WIP) the following structure:
//!
//!     |    chunk0    |    chunk1          |     chunk2         | chunk3         |
//!     | tx0 | tx1 | tx2 | tx3 | tx4 | tx5 | tx6 | tx7 | tx8 | tx9 | tx10 | tx11 |
//!     |                   block0          |               block1                |
//!
//! To run this test:
//!
//!     cargo test -p bus-mapping -- --nocapture  test_chunk
//!
//! or:
//!
//!     cargo nextest run --no-capture  --  test_chunk
//!
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

use crate::{
    circuit_input_builder::{ChunkContext, ExecState, MAX_CHUNK_SIZE},
    mock::{self},
};
use ::mock::{eth, TestContext};
use eth_types::{address, bytecode, geth_types::GethData, ToWord, Word};

#[test]
fn test_chunk() {
    //! Create single block with 1024 transactions.
    //! Check that correctly create four chunks of [300, 300, 300, 124] transactions respectively.
    pub const N_TRANSACTIONS: usize = 3;
    pub const TOTAL_CHUNK_COUNT: usize = (N_TRANSACTIONS + MAX_CHUNK_SIZE - 1) / MAX_CHUNK_SIZE;
    let gas_cost = Word::from(53000);
    let required_gas = (gas_cost * N_TRANSACTIONS).try_into().unwrap();

    // TODO: The following is just random and may be removed.
    let _code_b = bytecode! {
        PUSH1(0x10)
        JUMP
        STOP
    };
    let code_a = bytecode! {
        PUSH1(0x0) // retLength
        PUSH1(0x0) // retOffset
        PUSH1(0x0) // argsLength
        PUSH1(0x0) // argsOffset
        PUSH32(address!("0x000000000000000000000000000000000cafe001").to_word()) // addr
        PUSH32(0x1_0000) // gas
        STATICCALL
        PUSH2(0xaa)
    };
    let test_ctx = TestContext::<2, N_TRANSACTIONS>::new(
        None,
        |accs| {
            accs[0]
                .address(address!("0x0000000000000000000000000000000000000000"))
                .balance(eth(0))
                .code(code_a);
            accs[1]
                .address(address!("0x000000000000000000000000000000000cafe001"))
                .balance(eth(required_gas))
                .nonce(100);
            // .code(code_b);
        },
        |txs, accs| {
            for tx in txs {
                tx.from(accs[1].address)
                    .to(accs[0].address)
                    .gas(Word::from(gas_cost));
            }
        },
        |block, _tx| block.number(0xcafeu64),
    )
    .unwrap();

    let block: GethData = test_ctx.into();
    let block_data = mock::BlockData::new_from_geth_data(block.clone());
    let builder = mock::BlockData::new_circuit_input_builder(&block_data);
    let fixed_builder = builder
        .handle_block(&block.eth_block, &block.geth_traces)
        .unwrap();
    let chunk_ctxs: Vec<ChunkContext> = fixed_builder.chunk_ctxs;
    let txs = fixed_builder.block.txs();
    // This is wrong.  How do we distinguish RWops from other ops?
    let total_rw_count: usize = txs.iter().map(|tx| tx.steps().len()).sum();

    // Verify the amount of chunks generated
    assert_ne!(chunk_ctxs.len(), 0, "No `chunk_ctxs` were generated.");
    // TODO: assert_eq!(TOTAL_CHUNK_COUNT, chunk_ctxs.len());

    // Sanity checks
    assert_ne!(txs.len(), 0, "Some transactions needed to generate table.");
    assert_ne!(total_rw_count, 0, "Some RW ops needed to chunk.");


    // Verify the RWCounter of each block.
    for current_chunk_ctx in chunk_ctxs.iter() {
        assert!(usize::from(current_chunk_ctx.rwc) < MAX_CHUNK_SIZE);
    }

    // Create a list of tripples: (op#, tx#, ExecStep)
    let exec_steps = txs
        .iter()
        .enumerate()
        .flat_map(|(txN, tx)| tx.steps().iter().map(move |step| (txN, step)))
        .enumerate()
        .map(|(i, (txN, step))| (i, txN, step));

    let curr_tx = txs.iter();
    let curr_chunk_ctx = chunk_ctxs.iter();

    let mut begin_chunk_count = 0;
    let mut end_chunk_count = 0;
    // We must start with matching `EndChunk` so we initialize with `BeginChunk`.
    let mut prev_chunk_label = ExecState::BeginChunk;

    // Print each ExecStep as a row in a table with associated tx number, chunk number and maybe
    // rwc, rwc_inner counter.
    let es_width = 16;
    let width = 13;
    let table_header = format!(
        "│{est:^width$}│{esta:^es_width$}│{tx:^width$}│{ch:^width$}│{rwc:^width$}│{rwci:^width$}│",
        est = "Op#",
        esta = "ExecState",
        tx = "Transaction",
        ch = "Chunk",
        rwc = "RWC",
        rwci = "RWC_inner",
    );
    let e = "─".repeat(width);
    let table_top = format!(
        "╭{e:-^width$}┬{e:─^es_width$}┬{e:─^width$}┬{e:^width$}┬{e:^width$}┬{e:^width$}╮",
    );
    let table_bottom =
        format!("╰{e:-^width$}┴{e:─^es_width$}┴{e:─^width$}┴{e:^width$}┴{e:^width$}┴{e:^width$}╯",);
    let table_header_separator =
        format!("├{e:-^width$}┼{e:─^es_width$}┼{e:─^width$}┼{e:^width$}┼{e:^width$}┼{e:^width$}┤",);
    println!("{table_top}");
    println!("{table_header}");
    println!("{table_header_separator}");
    for (ops,tx, es) in exec_steps {
        let exec_str = format!("{:?}", es.exec_state);
        println!(
            "│{opN:>width$}│{esta:<es_width$}│{tx:>width$}│{ch:>width$}│{rwc:>width$}│{rwci:>width$}│",
            opN = ops,
            esta = exec_str,
            tx = tx,
            ch = 42,
            rwc = usize::from(es.rwc),
            rwci = usize::from(es.rwc_inner_chunk),
        );

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
            (ExecState::EndChunk, _) => (), // assert!(false, "Property 2 or 5 violated."),
            _ => (),
        }
    }
    println!("{table_bottom}");
    // assert_eq!(
    // prev_chunk_label,
    // ExecState::EndChunk,
    // "Property 6 violated."
    // );
    //
    // Property 4.
    // assert_eq!(begin_chunk_count, TOTAL_CHUNK_COUNT - 1);
    // assert_eq!(end_chunk_count, TOTAL_CHUNK_COUNT - 1);
}
