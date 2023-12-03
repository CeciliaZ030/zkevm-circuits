use crate::{
    evm_circuit::{
        execution::ExecutionGadget,
        param::N_BYTES_PROGRAM_COUNTER,
        step::ExecutionState,
        util::{
            common_gadget::{SameContextGadget, WordByteRangeGadget},
            constraint_builder::{
                ConstrainBuilderCommon, EVMConstraintBuilder, StepStateTransition,
                Transition::{Delta, To},
            },
            math_gadget::IsZeroWordGadget,
            select, CachedRegion,
        },
        witness::{Block, Call, Chunk, ExecStep, Transaction},
    },
    util::{
        word::{Word, WordCell, WordExpr},
        Expr,
    },
};
use eth_types::{evm_types::OpcodeId, Field};
use halo2_proofs::{circuit::Value, plonk::Error};

#[derive(Clone, Debug)]
pub(crate) struct JumpiGadget<F> {
    same_context: SameContextGadget<F>,
    dest: WordByteRangeGadget<F, N_BYTES_PROGRAM_COUNTER>,
    condition: WordCell<F>,
    is_condition_zero: IsZeroWordGadget<F, WordCell<F>>,
}

impl<F: Field> ExecutionGadget<F> for JumpiGadget<F> {
    const NAME: &'static str = "JUMPI";

    const EXECUTION_STATE: ExecutionState = ExecutionState::JUMPI;

    fn configure(cb: &mut EVMConstraintBuilder<F>) -> Self {
        let dest = WordByteRangeGadget::construct(cb);
        let condition = cb.query_word_unchecked();

        // Pop the value from the stack
        cb.stack_pop(dest.original());
        cb.stack_pop(condition.to_word());

        // Determine if the jump condition is met
        let is_condition_zero = IsZeroWordGadget::construct(cb, &condition);
        let should_jump = 1.expr() - is_condition_zero.expr();

        // Lookup opcode at destination when should_jump
        cb.condition(should_jump.clone(), |cb| {
            cb.require_equal(
                "JUMPI destination must be within range if condition is non-zero",
                dest.not_overflow(),
                1.expr(),
            );

            cb.opcode_lookup_at(dest.valid_value(), OpcodeId::JUMPDEST.expr(), 1.expr());
        });

        // Transit program_counter to destination when should_jump, otherwise by
        // delta 1.
        let next_program_counter = select::expr(
            should_jump,
            dest.valid_value(),
            cb.curr.state.program_counter.expr() + 1.expr(),
        );

        // State transition
        let opcode = cb.query_cell();
        let step_state_transition = StepStateTransition {
            rw_counter: Delta(2.expr()),
            program_counter: To(next_program_counter),
            stack_pointer: Delta(2.expr()),
            gas_left: Delta(-OpcodeId::JUMPI.constant_gas_cost().expr()),
            ..Default::default()
        };
        let same_context = SameContextGadget::construct(cb, opcode, step_state_transition);

        Self {
            same_context,
            dest,
            condition,
            is_condition_zero,
        }
    }

    fn assign_exec_step(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        offset: usize,
        block: &Block<F>,
        chunk: &Chunk<F>,
        _: &Transaction,
        _: &Call,
        step: &ExecStep,
    ) -> Result<(), Error> {
        self.same_context.assign_exec_step(region, offset, step)?;

        let [destination, condition] = [0, 1].map(|index| block.get_rws(step, index).stack_value());

        self.dest.assign(region, offset, destination)?;
        self.condition.assign_u256(region, offset, condition)?;
        self.is_condition_zero
            .assign_value(region, offset, Value::known(Word::from(condition)))?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        evm_circuit::test::{rand_range, rand_word},
        test_util::CircuitTestBuilder,
    };
    use eth_types::{bytecode, Word};
    use mock::TestContext;

    fn test_ok(destination: usize, condition: Word) {
        assert!((68..(1 << 24) - 1).contains(&destination));

        let mut bytecode = bytecode! {
            PUSH32(condition)
            PUSH32(destination)
            JUMPI
            STOP
        };
        for _ in 0..(destination - 68) {
            bytecode.write(0, false);
        }
        bytecode.append(&bytecode! {
            JUMPDEST
            STOP
        });

        CircuitTestBuilder::new_from_test_ctx(
            TestContext::<2, 1>::simple_ctx_with_bytecode(bytecode).unwrap(),
        )
        .run();
    }

    #[test]
    fn jumpi_gadget_simple() {
        test_ok(68, 1.into());
        test_ok(100, 1.into());
        test_ok(1 << 11, 1.into());
        test_ok(68, 0.into());
        test_ok(100, 0.into());
        test_ok(1 << 11, 0.into());
    }

    #[test]
    #[ignore]
    fn jumpi_gadget_huge_bytecode() {
        test_ok(0x5ffe, 1.into());
        test_ok(0x5ffe, 0.into());
    }

    #[test]
    fn jumpi_gadget_rand() {
        test_ok(rand_range(68..1 << 11), Word::zero());
        test_ok(rand_range(68..1 << 11), rand_word());
    }

    #[test]
    #[ignore]
    fn jumpi_gadget_rand_huge_bytecode() {
        test_ok(rand_range(1 << 11..0x5fff), Word::zero());
        test_ok(rand_range(1 << 11..0x5fff), rand_word());
    }

    #[test]
    fn jumpi_gadget_with_zero_cond_and_overflow_dest() {
        let bytecode = bytecode! {
            PUSH32(Word::MAX)
            PUSH32(Word::zero())
            JUMPI
        };

        CircuitTestBuilder::new_from_test_ctx(
            TestContext::<2, 1>::simple_ctx_with_bytecode(bytecode).unwrap(),
        )
        .run();
    }
}
