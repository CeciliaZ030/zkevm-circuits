use crate::{
    evm_circuit::{
        execution::ExecutionGadget,
        step::ExecutionState,
        util::{
            common_gadget::SameContextGadget,
            constraint_builder::{EVMConstraintBuilder, StepStateTransition, Transition::Delta},
            CachedRegion,
        },
        witness::{Block, Call, Chunk, ExecStep, Transaction},
    },
    table::CallContextFieldTag,
    util::{
        word::{WordCell, WordExpr},
        Expr,
    },
};
use bus_mapping::evm::OpcodeId;
use eth_types::Field;
use halo2_proofs::plonk::Error;

#[derive(Clone, Debug)]
pub(crate) struct CallerGadget<F> {
    same_context: SameContextGadget<F>,
    caller_address: WordCell<F>,
}

impl<F: Field> ExecutionGadget<F> for CallerGadget<F> {
    const NAME: &'static str = "CALLER";

    const EXECUTION_STATE: ExecutionState = ExecutionState::CALLER;

    fn configure(cb: &mut EVMConstraintBuilder<F>) -> Self {
        let caller_address = cb.query_word_unchecked();

        // Lookup rw_table -> call_context with caller address
        cb.call_context_lookup_read(
            None, // cb.curr.state.call_id
            CallContextFieldTag::CallerAddress,
            caller_address.to_word(),
        );

        // Push the value to the stack
        cb.stack_push(caller_address.to_word());

        // State transition
        let opcode = cb.query_cell();
        let step_state_transition = StepStateTransition {
            rw_counter: Delta(2.expr()),
            program_counter: Delta(1.expr()),
            stack_pointer: Delta((-1).expr()),
            gas_left: Delta(-OpcodeId::CALLER.constant_gas_cost().expr()),
            ..Default::default()
        };
        let same_context = SameContextGadget::construct(cb, opcode, step_state_transition);

        Self {
            same_context,
            caller_address,
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

        let caller = block.get_rws(step, 1).stack_value();

        self.caller_address.assign_u256(region, offset, caller)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::test_util::CircuitTestBuilder;
    use eth_types::bytecode;
    use mock::TestContext;

    #[test]
    fn caller_gadget_test() {
        let bytecode = bytecode! {
            CALLER
            STOP
        };

        CircuitTestBuilder::new_from_test_ctx(
            TestContext::<2, 1>::simple_ctx_with_bytecode(bytecode).unwrap(),
        )
        .run();
    }
}
