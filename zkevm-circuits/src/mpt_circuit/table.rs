use eth_types::Field;
use gadgets::util::Expr;
use halo2_proofs::plonk::Expression;
use strum_macros::EnumIter;
use gadgets::{impl_expr, util::Scalar};
use crate::circuit_tools::cell_manager::Table;

#[derive(Clone, Copy, Debug)]
pub enum FixedTableTag {
    /// All zero lookup data
    Disabled,
    /// Power of randomness: [1, r], [2, r^2],...
    RMult,
    /// 0 - 15
    Range16,
    /// 0 - 255
    Range256,
    /// For checking there are 0s after the RLP stream ends
    RangeKeyLen256,
    /// For checking there are 0s after the RLP stream ends
    RangeKeyLen16,
    /// For checking RLP
    RLP,
    /// For distinguishing odd key part in extension
    ExtOddKey,
    /// State transition steps constriants
    /// 2
    StartNode,
    /// 21
    BranchNode,
    /// 12
    AccountNode,
    /// 6
    StorageNode
}
impl_expr!(FixedTableTag);


#[derive(Clone, Debug)]
pub(crate) enum Lookup<F> {
    /// Lookup to fixed table, which contains serveral pre-built tables such as
    /// range tables or bitwise tables.
    Fixed {
        /// Tag to specify which table to lookup.
        tag: Expression<F>,
        /// Values that must satisfy the pre-built relationship.
        values: [Expression<F>; 5],
    },

    /// Lookup to keccak table.
    KeccakTable {
        /// Accumulator to the input.
        input_rlc: Expression<F>,
        /// Length of input that is being hashed.
        input_len: Expression<F>,
        /// Output (hash) until this state. This is the RLC representation of
        /// the final output keccak256 hash of the input.
        output_rlc: Expression<F>,
    },

    /// Conditional lookup enabled by the first element.
    Conditional(Expression<F>, Box<Lookup<F>>),
}

impl<F: Field> Lookup<F> {
    pub(crate) fn conditional(self, condition: Expression<F>) -> Self {
        Self::Conditional(condition, self.into())
    }

    pub(crate) fn table(&self) -> Table {
        match self {
            Self::Fixed { .. } => Table::Fixed,
            Self::KeccakTable {.. } => Table::Keccak,
            Self::Conditional(_, lookup) => lookup.table(),
        }
    }

    pub(crate) fn input_exprs(&self) -> Vec<Expression<F>> {
        match self {
            Self::Fixed { tag, values } => [vec![tag.clone()], values.to_vec()].concat(),
            Self::KeccakTable {
                input_rlc,
                input_len,
                output_rlc,
            } => vec![
                1.expr(), // is_enabled
                input_rlc.clone(),
                input_len.clone(),
                output_rlc.clone(),
            ],
            Self::Conditional(condition, lookup) => lookup
                .input_exprs()
                .into_iter()
                .map(|expr| condition.clone() * expr)
                .collect(),
        }
    }

    pub(crate) fn degree(&self) -> usize {
        self.input_exprs()
            .iter()
            .map(|expr| expr.degree())
            .max()
            .unwrap()
    }

}