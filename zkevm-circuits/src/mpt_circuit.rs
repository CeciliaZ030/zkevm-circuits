//! The MPT circuit implementation.
use eth_types::Field;
use gadgets::{impl_expr, util::Scalar, is_zero::IsZeroConfig};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, VirtualCells},
    poly::Rotation,
};

use std::{convert::TryInto, env::var};

mod account_leaf;
mod branch;
mod columns;
mod extension;
mod extension_branch;
mod helpers;
mod param;
mod rlp_gadgets;
mod selectors;
mod start;
mod storage_leaf;
mod witness_row;
mod table;

use columns::MainCols;
use extension_branch::ExtensionBranchConfig;
use witness_row::{MptWitnessRow, MptWitnessRowType};

use param::*;

use self::{
    account_leaf::AccountLeafConfig,
    helpers::key_memory,
    param::{
        ARITY, BRANCH_0_KEY_POS, DRIFTED_POS, IS_ACCOUNT_DELETE_MOD_POS, IS_BALANCE_MOD_POS,
        IS_BRANCH_C_PLACEHOLDER_POS, IS_BRANCH_S_PLACEHOLDER_POS, IS_CODEHASH_MOD_POS,
        IS_EXT_LONG_EVEN_C16_POS, IS_EXT_LONG_EVEN_C1_POS, IS_EXT_LONG_ODD_C16_POS,
        IS_EXT_LONG_ODD_C1_POS, IS_EXT_SHORT_C16_POS, IS_EXT_SHORT_C1_POS, IS_NONCE_MOD_POS,
        IS_NON_EXISTING_ACCOUNT_POS, IS_NON_EXISTING_STORAGE_POS, IS_STORAGE_MOD_POS,
        RLP_LIST_LONG, RLP_LIST_SHORT,
    },
    witness_row::{
        AccountNode, AccountRowType, BranchNode, ExtensionBranchNode, ExtensionBranchRowType,
        ExtensionNode, Node, StartNode, StartRowType, StorageNode, StorageRowType,
    },
};
use crate::{mpt_circuit::helpers::Indexable, evm_circuit::util::CachedRegion};
use crate::{
    evm_circuit::util::math_gadget::IsZeroGadget,
    assign, assignf, circuit,
    circuit_tools::{cell_manager::CellManager, constraint_builder::merge_lookups, memory::Memory},
    matchr, matchw,
    mpt_circuit::{
        helpers::{extend_rand, main_memory, parent_memory, MPTConstraintBuilder},
        start::StartConfig,
        storage_leaf::StorageLeafConfig,
    },
    table::{DynamicTableColumns, KeccakTable, MptTable, ProofType},
    util::{power_of_randomness_from_instance, Challenges},
};

/// State machine config.
#[derive(Clone, Debug, Default)]
pub struct StateMachineConfig<F> {
    start_config: StartConfig<F>,
    branch_config: ExtensionBranchConfig<F>,
    storage_config: StorageLeafConfig<F>,
    account_config: AccountLeafConfig<F>,
}

/// Merkle Patricia Trie context
#[derive(Clone, Debug)]
pub struct MPTContext<F> {
    pub(crate) q_enable: Column<Fixed>,
    pub(crate) q_not_first: Column<Fixed>,
    pub(crate) q_row: Column<Advice>,
    pub(crate) q_row_inv: Column<Advice>,

    pub(crate) mpt_table: MptTable,
    pub(crate) main: MainCols<F>,
    pub(crate) managed_columns: Vec<Column<Advice>>,
    pub(crate) r: Vec<Expression<F>>,
    pub(crate) memory: Memory<F>,
}

impl<F: Field> MPTContext<F> {
    pub(crate) fn bytes(&self) -> Vec<Column<Advice>> {
        self.main.bytes.to_vec()
    }

    pub(crate) fn s(&self, meta: &mut VirtualCells<F>, rot: i32) -> Vec<Expression<F>> {
        self.bytes()[0..34]
            .iter()
            .map(|&byte| meta.query_advice(byte, Rotation(rot)))
            .collect::<Vec<_>>()
    }

}

/// Merkle Patricia Trie config.
#[derive(Clone)]
pub struct MPTConfig<F> {
    pub(crate) q_enable: Column<Fixed>,
    pub(crate) q_not_first: Column<Fixed>,

    pub(crate) main: MainCols<F>,
    pub(crate) managed_columns: Vec<Column<Advice>>,
    pub(crate) memory: Memory<F>,

    keccak_table: KeccakTable,
    fixed_table: [Column<Fixed>; 5],
    state_machine: StateMachineConfig<F>,

    pub(crate) q_node: Column<Advice>,
    pub(crate) q_row: Column<Advice>,
    pub(crate) q_row_inv: Column<Advice>,

    pub(crate) is_start: Column<Advice>,
    pub(crate) is_branch: Column<Advice>,
    pub(crate) is_account: Column<Advice>,
    pub(crate) is_storage: Column<Advice>,
    pub(crate) r: F,
    pub(crate) mpt_table: MptTable,
    cb: MPTConstraintBuilder<F>,
}

/// Enumerator to determine the type of row in the fixed table.
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

#[derive(Default)]
pub(crate) struct MPTState<F> {
    pub(crate) memory: Memory<F>,
}

impl<F: Field> MPTState<F> {
    fn new(memory: &Memory<F>) -> Self {
        Self {
            memory: memory.clone(),
            ..Default::default()
        }
    }
}

impl<F: Field> MPTConfig<F> {
    /// Configure MPT Circuit
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        power_of_randomness: [Expression<F>; HASH_WIDTH],
        keccak_table: KeccakTable,
    ) -> Self {
        let q_enable = meta.fixed_column();
        let q_not_first = meta.fixed_column();

        let mpt_table = MptTable::construct(meta);

        let q_node = meta.advice_column();
        let q_row = meta.advice_column();
        let q_row_inv = meta.advice_column();

        let is_start = meta.advice_column();
        let is_branch = meta.advice_column();
        let is_account = meta.advice_column();
        let is_storage = meta.advice_column();

        let main = MainCols::new(meta);

        let fixed_table: [Column<Fixed>; 5] = (0..5)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let managed_columns = (0..20).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let memory_columns = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let mut memory = Memory::new(memory_columns);
        memory.allocate(meta, key_memory(false));
        memory.allocate(meta, key_memory(true));
        memory.allocate(meta, parent_memory(false));
        memory.allocate(meta, parent_memory(true));
        memory.allocate(meta, main_memory());

        let mut cb = MPTConstraintBuilder::new(33 + 10, None);

        let mut ctx = MPTContext {
            q_enable: q_enable.clone(),
            q_not_first: q_not_first.clone(),
            q_row: q_row.clone(),
            q_row_inv: q_row_inv.clone(),

            mpt_table: mpt_table.clone(),
            main: main.clone(),
            managed_columns: managed_columns.clone(),
            r: extend_rand(&power_of_randomness),
            memory: memory.clone(),
        };

        let mut state_machine = StateMachineConfig::default();

        println!("total advices {}", meta.num_advice_columns());

        meta.create_gate("MPT", |meta| {
            // 20 cols * 32 height in CellManager
            let cell_manager = CellManager::new(meta, &ctx.managed_columns);
            cb.base.set_cell_manager(cell_manager);
            
            circuit!([meta, cb.base], {
                let is_q_row_zero = 1.expr() - a!(q_row) * a!(q_row_inv);
                // State machine
                // TODO(Brecht): state machine constraints
                ifx!{f!(q_enable) => {
                    // Always start with the start state
                    ifx! {not!(f!(q_not_first)) => {
                        require!(a!(is_start) => true);
                    }};
                    // When q_row == 0, we start at a new node,
                    // one of [is_start, is_branch, is_account, is_storage] needs to be 1,
                    // if not we goes to  _ => require!(true => false)
                    // Otherwise q_row > 0, we're in the middle of some node rows, all flags needs to be 0;
                    ifx! {is_q_row_zero.expr() => {
                        matchx! {
                            a!(is_start) => {
                                // require!(a!(q_row) + a!(q_node) => 0.expr());                      
                                state_machine.start_config = StartConfig::configure(meta, &mut cb, ctx.clone());
                            },
                            a!(is_branch) => {
                                state_machine.branch_config = ExtensionBranchConfig::configure(meta, &mut cb, ctx.clone());
                            },
                            a!(is_account) => {
                                state_machine.account_config = AccountLeafConfig::configure(meta, &mut cb, ctx.clone());
                            },
                            a!(is_storage)  => {
                                state_machine.storage_config = StorageLeafConfig::configure(meta, &mut cb, ctx.clone());
                            },
                            _ => require!(true => false),
                        };
                    } elsex {
                        require! ((a!(is_start) + a!(is_branch) + a!(is_account) + a!(is_storage)) => 0.expr());
                    }};

                    // Lookahead
                    // when q_row.next() != 0 then q_row.next == q_row + 1
                    ifx! {a!(q_row, 1i32) => {
                        require!(a!(q_row) + 1.expr() => a!(q_row, 1i32))
                    }};
                   
                    // Main state machine
                    // Only account and storage rows can have lookups, disable lookups on all other rows
                    matchx! {
                        a!(is_account) => (),
                        a!(is_storage) => (),
                        _ => require!(a!(ctx.mpt_table.proof_type) => ProofType::Disabled.expr()),
                    }
                }}

                // TODO(Brecht): decode 1 RLP item/row
                /* Range checks */
                // These range checks ensure that the value in the RLP columns are all byte value.
                // These lookups also enforce the byte value to be zero the byte index >= length.
                // TODO(Brecht): do 2 bytes/lookup when circuit height >= 2**21
                /*ifx!{f!(position_cols.q_enable) => {
                    // Sanity checks (can be removed, here for safety)
                    require!(cb.length_s.sum_conditions() => bool);
                    // Range checks
                    for (idx, &byte) in ctx.s().into_iter().enumerate() {
                        require!((cb.get_range_s(), a!(byte), cb.get_length_s() - (idx + 1).expr()) => @"fixed");
                    }
                }}*/

                /* Populate lookup tables */
                require!(@"keccak" => keccak_table.columns().iter().map(|table| a!(table)).collect());
                require!(@"fixed" => fixed_table.iter().map(|table| f!(table)).collect());

                /* Memory banks */
                ifx!{f!(q_enable) => {
                    ctx.memory.generate_constraints(&mut cb.base, not!(f!(q_not_first)));
                }}
            });

            cb.base.generate_constraints()
        });

        let disable_lookups: usize = var("DISABLE_LOOKUPS")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .expect("Cannot parse DISABLE_LOOKUPS env var as usize");
        println!("DISABLE_LOOKUPS={:?}", disable_lookups);
        if disable_lookups == 0 {
            cb.base.generate_lookups(
                meta,
                &[
                    vec!["fixed".to_string(), "keccak".to_string(), "bytes".to_string()],
                    ctx.memory.tags(),
                ]
                .concat(),
            );
        } else if disable_lookups == 1 {
            // let cm = cb.base.cell_manager?;
            cb.base.generate_lookups(
                meta,
                &[vec!["keccak".to_string()], ctx.memory.tags()].concat(),
            );
        } else if disable_lookups == 2 {
            cb.base.generate_lookups(meta, &ctx.memory.tags());
        } else if disable_lookups == 3 {
            cb.base
                .generate_lookups(meta, &vec!["fixed".to_string(), "keccak".to_string()]);
        } else if disable_lookups == 4 {
            cb.base.generate_lookups(meta, &vec!["fixed".to_string()]);
        }

        println!("num lookups: {}", meta.lookups().len());
        println!("num advices: {}", meta.num_advice_columns());
        println!("num fixed: {}", meta.num_fixed_columns());
        //cb.base.print_stats();

        MPTConfig {
            q_enable,
            q_not_first,
            q_node,
            q_row,
            q_row_inv,
            is_start,
            is_branch,
            is_account,
            is_storage,
            main,
            managed_columns,
            memory,
            keccak_table,
            fixed_table,
            state_machine,
            r: 0.scalar(),
            mpt_table,
            cb,
        }
    }

    /// Make the assignments to the MPTCircuit
    pub fn assign(
        &mut self,
        layouter: &mut impl Layouter<F>,
        witness: &mut [MptWitnessRow<F>],
        r: F,
    ) -> Result<(), Error> {
        self.r = r;
        let mut height = 0;
        let mut memory = self.memory.clone();

        // TODO(Brecht): change this on the witness generation side
        let mut key_rlp_bytes = Vec::new();
        for (_, row) in witness
            .iter_mut()
            .filter(|r| r.get_type() != MptWitnessRowType::HashToBeComputed)
            .enumerate()
        {
            // Get the proof type directly
            if row.get_byte_rev(IS_STORAGE_MOD_POS) == 1 {
                row.proof_type = ProofType::StorageChanged;
            }
            if row.get_byte_rev(IS_NONCE_MOD_POS) == 1 {
                row.proof_type = ProofType::NonceChanged;
            }
            if row.get_byte_rev(IS_BALANCE_MOD_POS) == 1 {
                row.proof_type = ProofType::BalanceChanged;
            }
            if row.get_byte_rev(IS_CODEHASH_MOD_POS) == 1 {
                row.proof_type = ProofType::CodeHashExists;
            }
            if row.get_byte_rev(IS_ACCOUNT_DELETE_MOD_POS) == 1 {
                row.proof_type = ProofType::AccountDestructed;
            }
            if row.get_byte_rev(IS_NON_EXISTING_ACCOUNT_POS) == 1 {
                row.proof_type = ProofType::AccountDoesNotExist;
            }
            if row.get_byte_rev(IS_NON_EXISTING_STORAGE_POS) == 1 {
                row.proof_type = ProofType::StorageDoesNotExist;
            }

            if row.get_type() == MptWitnessRowType::BranchChild {
                //println!("- {:?}", row.bytes);
                let mut child_s_bytes = row.bytes[0..34].to_owned();
                if child_s_bytes[1] == 160 {
                    child_s_bytes[0] = 0;
                    child_s_bytes.rotate_left(1);
                } else {
                    child_s_bytes[0] = 0;
                    child_s_bytes[1] = 0;
                    child_s_bytes.rotate_left(2);
                };

                let mut child_c_bytes = row.bytes[34..68].to_owned();
                if child_c_bytes[1] == 160 {
                    child_c_bytes[0] = 0;
                    child_c_bytes.rotate_left(1);
                } else {
                    child_c_bytes[0] = 0;
                    child_c_bytes[1] = 0;
                    child_c_bytes.rotate_left(2);
                };

                row.bytes = [
                    child_s_bytes.clone(),
                    child_c_bytes.clone(),
                    row.bytes[68..].to_owned(),
                ]
                .concat();
                //println!("+ {:?}", row.bytes);
            }

            if row.get_type() == MptWitnessRowType::ExtensionNodeS
                || row.get_type() == MptWitnessRowType::ExtensionNodeC
            {
                //println!("- {:?}", row.bytes);
                let mut value_bytes = row.bytes[34..68].to_owned();
                if value_bytes[1] == 160 {
                    value_bytes[0] = 0;
                    value_bytes.rotate_left(1);
                } else {
                    value_bytes[0] = 0;
                    value_bytes[1] = 0;
                    value_bytes.rotate_left(2);
                };
                row.bytes = [
                    row.bytes[0..34].to_owned(),
                    value_bytes.clone(),
                    row.bytes[68..].to_owned(),
                ]
                .concat();
                //println!("+ {:?}", row.bytes);
            }

            // Separate the list rlp bytes from the key bytes
            if row.get_type() == MptWitnessRowType::StorageLeafSKey
                || row.get_type() == MptWitnessRowType::StorageLeafCKey
                || row.get_type() == MptWitnessRowType::StorageNonExisting
                || row.get_type() == MptWitnessRowType::NeighbouringStorageLeaf
                || row.get_type() == MptWitnessRowType::AccountLeafKeyS
                || row.get_type() == MptWitnessRowType::AccountLeafKeyC
                || row.get_type() == MptWitnessRowType::AccountNonExisting
                || row.get_type() == MptWitnessRowType::AccountLeafNeighbouringLeaf
                || row.get_type() == MptWitnessRowType::ExtensionNodeS
            {
                let len = if row.get_type() == MptWitnessRowType::ExtensionNodeS {
                    34
                } else {
                    36
                };
                let mut key_bytes = row.bytes[0..len].to_owned();

                // Currently the list rlp bytes are dropped for non-key row, restore them here
                if key_bytes[0] < RLP_LIST_SHORT
                    && row.get_type() != MptWitnessRowType::ExtensionNodeS
                {
                    for idx in 0..key_rlp_bytes.len() {
                        key_bytes[idx] = key_rlp_bytes[idx];
                    }
                }

                const RLP_LIST_LONG_1: u8 = RLP_LIST_LONG + 1;
                const RLP_LIST_LONG_2: u8 = RLP_LIST_LONG + 2;
                let mut is_short = false;
                let mut is_long = false;
                let mut is_very_long = false;
                let mut is_string = false;
                match key_bytes[0] {
                    RLP_LIST_SHORT..=RLP_LIST_LONG => is_short = true,
                    RLP_LIST_LONG_1 => is_long = true,
                    RLP_LIST_LONG_2 => is_very_long = true,
                    _ => is_string = true,
                }

                //println!("bytes: {:?}", key_bytes);

                let num_rlp_bytes = if is_short {
                    1
                } else if is_long {
                    2
                } else if is_very_long {
                    3
                } else {
                    if row.get_type() == MptWitnessRowType::ExtensionNodeS {
                        0
                    } else {
                        unreachable!()
                    }
                };

                //println!("bytes: {:?}", key_bytes);
                row.rlp_bytes = key_bytes[..num_rlp_bytes].to_vec();
                for byte in key_bytes[..num_rlp_bytes].iter_mut() {
                    *byte = 0;
                }
                key_bytes.rotate_left(num_rlp_bytes);
                row.bytes = [key_bytes.clone(), row.bytes[len..].to_owned()].concat();

                if row.get_type() == MptWitnessRowType::AccountLeafKeyS
                    || row.get_type() == MptWitnessRowType::StorageLeafSKey
                {
                    key_rlp_bytes = row.rlp_bytes.clone();
                }

                //println!("list : {:?}", row.rlp_bytes);
                //println!("key  : {:?}", row.bytes);
            }

            // Separate the RLP bytes and shift the value bytes to the start of the row
            if row.get_type() == MptWitnessRowType::AccountLeafNonceBalanceS
                || row.get_type() == MptWitnessRowType::AccountLeafNonceBalanceC
            {
                row.rlp_bytes = [row.bytes[..2].to_owned(), row.bytes[34..36].to_owned()].concat();

                let nonce = row.bytes[2..34].to_owned();
                let balance = row.bytes[36..68].to_owned();

                row.bytes = [
                    nonce,
                    vec![0; 2],
                    balance,
                    vec![0; 2],
                    row.bytes[68..].to_owned(),
                ]
                .concat();
            }

            // Shift the value bytes to the start of the row
            if row.get_type() == MptWitnessRowType::AccountLeafRootCodehashS
                || row.get_type() == MptWitnessRowType::AccountLeafRootCodehashC
            {
                let storage_root = row.bytes[1..34].to_owned();
                let codehash = row.bytes[35..68].to_owned();

                row.bytes = [
                    storage_root,
                    vec![0; 1],
                    codehash,
                    vec![0; 1],
                    row.bytes[68..].to_owned(),
                ]
                .concat();
            }

            if row.get_type() == MptWitnessRowType::InitBranch {
                // Extract the RLP bytes
                row.rlp_bytes = [row.bytes[4..7].to_owned(), row.bytes[7..10].to_owned()].concat();

                // Store a single value that the branch is an extension node or not
                row.is_extension = row.get_byte(IS_EXT_LONG_ODD_C16_POS)
                    + row.get_byte(IS_EXT_LONG_ODD_C1_POS)
                    + row.get_byte(IS_EXT_SHORT_C16_POS)
                    + row.get_byte(IS_EXT_SHORT_C1_POS)
                    + row.get_byte(IS_EXT_LONG_EVEN_C16_POS)
                    + row.get_byte(IS_EXT_LONG_EVEN_C1_POS)
                    == 1;
                row.is_placeholder = [
                    row.get_byte(IS_BRANCH_S_PLACEHOLDER_POS) == 1,
                    row.get_byte(IS_BRANCH_C_PLACEHOLDER_POS) == 1,
                ];
                row.modified_index = row.get_byte(BRANCH_0_KEY_POS) as usize;
                row.drifted_index = row.get_byte(DRIFTED_POS) as usize;
                // Move the modified branch into the init row
                row.bytes = [vec![0; 68], row.bytes[68..].to_owned()].concat();
            }

            // Shift the value bytes to the start of the row
            if row.get_type() == MptWitnessRowType::StorageLeafSValue
                || row.get_type() == MptWitnessRowType::StorageLeafCValue
            {
                row.rlp_bytes = vec![row.bytes[0]];
                row.bytes = [row.bytes[1..].to_owned()].concat();
            }
        }

        // TODO(Brecht): change this on the witness generation side
        let cached_witness = witness.to_owned();
        for (idx, row) in witness
            .iter_mut()
            .filter(|r| r.get_type() != MptWitnessRowType::HashToBeComputed)
            .enumerate()
        {
            if row.get_type() == MptWitnessRowType::InitBranch {
                // Move the modified branch into the init row
                let mod_bytes = cached_witness[idx + 1 + row.modified_index].c();
                row.bytes = [mod_bytes, row.bytes[34..].to_owned()].concat();
            }
        }

        let mut nodes = Vec::new();
        let witness = witness
            .iter()
            .filter(|r| r.get_type() != MptWitnessRowType::HashToBeComputed)
            .collect::<Vec<_>>();
        let mut offset = 0;
        while offset < witness.len() {
            //println!("offset: {}", offset);
            let mut new_proof = offset == 0;
            if offset > 0 {
                let row_prev = witness[offset - 1].clone();
                let not_first_level_prev = row_prev.not_first_level();
                let not_first_level_cur = witness[offset].not_first_level();
                if not_first_level_cur == 0 && not_first_level_prev == 1 {
                    new_proof = true;
                }
            }

            // 🌻 offset = 0 ｜ node_rows 0-2
            if new_proof {
                let mut new_row = witness[offset].clone();
                new_row.bytes = [
                    new_row.s_root_bytes().to_owned(),
                    vec![0; 2],
                    new_row.c_root_bytes().to_owned(),
                    vec![0; 2],
                ]
                .concat();

                let mut node_rows = vec![Vec::new(); StartRowType::Count as usize];
                node_rows[StartRowType::RootS as usize] = new_row.s();
                node_rows[StartRowType::RootC as usize] = new_row.c();

                let start_node = StartNode {
                    proof_type: new_row.proof_type.clone(),
                };
                let mut node = Node::default();
                node.start = Some(start_node);
                node.values = node_rows;
                nodes.push(node);
            }
            // 🌻 offset = 0 ｜ node_rows 0-21
            if witness[offset].get_type() == MptWitnessRowType::InitBranch {
                let row_init = witness[offset].to_owned();
                let is_placeholder = row_init.is_placeholder.clone();
                let is_extension = row_init.is_extension;
                let modified_index = row_init.modified_index;
                let mut drifted_index = row_init.drifted_index;
                // If no placeholder branch, we set `drifted_pos = modified_node`. This
                // is needed just to make some other constraints (`s_mod_node_hash_rlc`
                // and `c_mod_node_hash_rlc` correspond to the proper node) easier to write.
                if !is_placeholder[true.idx()] && !is_placeholder[false.idx()] {
                    drifted_index = modified_index;
                }
                let branch_list_rlp_bytes = [
                    row_init.rlp_bytes[0..3].to_owned(),
                    row_init.rlp_bytes[3..6].to_owned(),
                ];
                let child_bytes: [Vec<u8>; ARITY + 1] =
                    array_init::array_init(|i| witness[offset + i].s());
                let ext_list_rlp_bytes = witness[offset + 17].rlp_bytes.to_owned();

                let mut node_rows = vec![Vec::new(); ExtensionBranchRowType::Count as usize];
                for idx in 0..ARITY + 1 {
                    node_rows[idx] = child_bytes[idx].clone();
                }
                node_rows[ExtensionBranchRowType::KeyS as usize] = witness[offset + 17].s();
                node_rows[ExtensionBranchRowType::ValueS as usize] = witness[offset + 17].c();
                node_rows[ExtensionBranchRowType::KeyC as usize] = witness[offset + 18].s();
                node_rows[ExtensionBranchRowType::ValueC as usize] = witness[offset + 18].c();
                offset += 19;

                let extension_branch_node = ExtensionBranchNode {
                    is_extension,
                    is_placeholder,
                    extension: ExtensionNode {
                        list_rlp_bytes: ext_list_rlp_bytes,
                    },
                    branch: BranchNode {
                        modified_index,
                        drifted_index,
                        list_rlp_bytes: branch_list_rlp_bytes,
                    },
                };
                let mut node = Node::default();
                node.extension_branch = Some(extension_branch_node);
                node.values = node_rows;
                nodes.push(node);
            } 
            // 🌻 offset = 19 ｜ node_rows 22-21
            else if witness[offset].get_type() == MptWitnessRowType::StorageLeafSKey {
                let row_key = [&witness[offset + 0], &witness[offset + 2]];
                let row_value = [&witness[offset + 1], &witness[offset + 3]];
                let row_drifted = &witness[offset + 4];
                let row_wrong = &witness[offset + 5];
                offset += 6;

                let list_rlp_bytes = [
                    row_key[true.idx()].rlp_bytes.to_owned(),
                    row_key[false.idx()].rlp_bytes.to_owned(),
                ];
                let value_rlp_bytes = [
                    row_value[true.idx()].rlp_bytes.to_owned(),
                    row_value[false.idx()].rlp_bytes.to_owned(),
                ];
                let drifted_rlp_bytes = row_drifted.rlp_bytes.clone();
                let wrong_rlp_bytes = row_wrong.rlp_bytes.clone();

                let mut node_rows = vec![Vec::new(); StorageRowType::Count as usize];
                node_rows[StorageRowType::KeyS as usize] = row_key[true.idx()].s();
                node_rows[StorageRowType::ValueS as usize] = row_value[true.idx()].s();
                node_rows[StorageRowType::KeyC as usize] = row_key[false.idx()].s();
                node_rows[StorageRowType::ValueC as usize] = row_value[false.idx()].s();
                node_rows[StorageRowType::Drifted as usize] = row_drifted.s();
                node_rows[StorageRowType::Wrong as usize] = row_wrong.s();

                let storage_node = StorageNode {
                    list_rlp_bytes,
                    value_rlp_bytes,
                    drifted_rlp_bytes,
                    wrong_rlp_bytes,
                };
                let mut node = Node::default();
                node.storage = Some(storage_node);
                node.values = node_rows;
                nodes.push(node);
            } else if witness[offset].get_type() == MptWitnessRowType::AccountLeafKeyS {
                let key_s = witness[offset].to_owned();
                let key_c = witness[offset + 1].to_owned();
                let nonce_balance_s = witness[offset + 3].to_owned();
                let nonce_balance_c = witness[offset + 4].to_owned();
                let storage_codehash_s = witness[offset + 5].to_owned();
                let storage_codehash_c = witness[offset + 6].to_owned();
                let row_drifted = witness[offset + 7].to_owned();
                let row_wrong = witness[offset + 2].to_owned();
                let address = witness[offset].address_bytes().to_owned();
                offset += 8;

                let list_rlp_bytes = [key_s.rlp_bytes.to_owned(), key_c.rlp_bytes.to_owned()];
                let value_rlp_bytes = [
                    nonce_balance_s.rlp_bytes.clone(),
                    nonce_balance_c.rlp_bytes.clone(),
                ];
                let drifted_rlp_bytes = row_drifted.rlp_bytes.clone();
                let wrong_rlp_bytes = row_wrong.rlp_bytes.clone();

                let mut node_rows = vec![Vec::new(); AccountRowType::Count as usize];
                node_rows[AccountRowType::KeyS as usize] = key_s.s();
                node_rows[AccountRowType::KeyC as usize] = key_c.s();
                node_rows[AccountRowType::NonceS as usize] = nonce_balance_s.s();
                node_rows[AccountRowType::BalanceS as usize] = nonce_balance_s.c();
                node_rows[AccountRowType::StorageS as usize] = storage_codehash_s.s();
                node_rows[AccountRowType::CodehashS as usize] = storage_codehash_s.c();
                node_rows[AccountRowType::NonceC as usize] = nonce_balance_c.s();
                node_rows[AccountRowType::BalanceC as usize] = nonce_balance_c.c();
                node_rows[AccountRowType::StorageC as usize] = storage_codehash_c.s();
                node_rows[AccountRowType::CodehashC as usize] = storage_codehash_c.c();
                node_rows[AccountRowType::Drifted as usize] = row_drifted.s();
                node_rows[AccountRowType::Wrong as usize] = row_wrong.s();

                let account_node = AccountNode {
                    address,
                    list_rlp_bytes,
                    value_rlp_bytes,
                    drifted_rlp_bytes,
                    wrong_rlp_bytes,
                };
                let mut node = Node::default();
                node.account = Some(account_node);
                node.values = node_rows;
                nodes.push(node);
            }
        }

        layouter.assign_region(
            || "MPT",
            |mut region| {
                let mut pv = MPTState::new(&self.memory);
                memory.clear_witness_data();

                let power_of_randomness: [F; 31] = array_init::array_init(|i | self.r.pow(&[i as u64, 0, 0, 0]));

                let mut offset = 0;
                for (node_id, node) in nodes.iter().enumerate() {
                    // Assign bytes
                    for (idx, bytes) in node.values.iter().enumerate() {
                        for (byte, &column) in bytes.iter().zip(self.main.bytes.iter()) {
                            assign!(region, (column, offset + idx) => byte.scalar())?;
                        }
                        let idx_scalar: F = idx.scalar();
                        assign!(region, (self.q_node, offset + idx) => offset.scalar())?;
                        assign!(region, (self.q_row, offset + idx) => idx_scalar)?;
                        assign!(region, (self.q_row_inv, offset + idx) => idx_scalar.invert().unwrap_or(F::zero()))?;
                    }

                    // Assign nodes
                    if node.start.is_some() {
                        let mut cached_region = CachedRegion::new(
                            &mut region, 
                            power_of_randomness,
                            TOTAL_WIDTH,
                            StartRowType::Count as usize, 
                            0,
                            offset,
                        );
                        println!("{}: start", offset);
                        assign!(cached_region, (self.is_start, offset) => 1.scalar())?;
                        self.state_machine.start_config.assign_cached(
                            &mut cached_region,
                            self,
                            &mut pv,
                            offset,
                            node,
                        )?;
                    } else if node.extension_branch.is_some() {
                        let mut cached_region = CachedRegion::new(
                            &mut region, 
                            power_of_randomness,
                            TOTAL_WIDTH,
                            ExtensionBranchRowType::Count as usize, 
                            0,
                            offset,
                        );
                        println!("{}: branch", offset);
                        assign!(cached_region, (self.is_branch, offset) => 1.scalar())?;
                        self.state_machine.branch_config.assign_cached(
                            &mut cached_region,
                            self,
                            &mut pv,
                            offset,
                            node,
                        )?;
                    } else if node.storage.is_some() {
                        let mut cached_region = CachedRegion::new(
                            &mut region, 
                            power_of_randomness,
                            TOTAL_WIDTH,
                            StorageRowType::Count as usize, 
                            0,
                            offset,
                        );
                        assign!(cached_region, (self.is_storage, offset) => 1.scalar())?;
                        println!("{}: storage", offset);
                        self.state_machine.storage_config.assign_cached(
                            &mut cached_region,
                            self,
                            &mut pv,
                            offset,
                            node,
                        )?;
                    } else if node.account.is_some() {
                        let mut cached_region = CachedRegion::new(
                            &mut region, 
                            power_of_randomness,
                            TOTAL_WIDTH,
                            AccountRowType::Count as usize, 
                            0,
                            offset,
                        );
                        assign!(cached_region, (self.is_account, offset) => 1.scalar())?;
                        println!("{}: account", offset);
                        self.state_machine.account_config.assign_cached(
                            &mut cached_region,
                            self,
                            &mut pv,
                            offset,
                            node,
                        )?;
                    }

                    println!("height: {}", node.values.len());
                    offset += node.values.len();
                }

                height = offset;
                memory = pv.memory;

                for offset in 0..height {
                    // assignf!(region, (self.q_enable, offset) => true.scalar())?;
                    assignf!(region, (self.q_not_first, offset) => (offset == 0).scalar())?;
                }

                Ok(())
            },
        )?;

        memory.assign(layouter, height)?;

        Ok(())
    }

    pub(crate) fn region_width(&self) -> usize {
        self.mpt_table.columns().len() 
            + self.main.bytes.len() 
            + self.managed_columns.len() 
            + self.memory.columns.len()
    }

    fn load_fixed_table(
        &self,
        layouter: &mut impl Layouter<F>,
        randomness: F,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "fixed table",
            |mut region| {
                let mut offset = 0;

                // Zero lookup
                for fixed_table in self.fixed_table.iter() {
                    assignf!(region, (*fixed_table, offset) => 0.scalar())?;
                }
                offset += 1;

                // Mult table
                let mut mult = F::one();
                for ind in 0..(2 * HASH_WIDTH + 1) {
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RMult.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => mult)?;
                    mult *= randomness;
                    offset += 1;
                }

                // Byte range table
                for ind in 0..256 {
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::Range256.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    offset += 1;
                }

                // Byte range with length table
                // These fixed rows enable to easily check whether there are zeros in the unused columns (the number of unused columns vary).
                // The lookups ensure that when the unused columns start, the values in these columns are zeros -
                // when the unused columns start, the value that is used for the lookup in the last column is negative
                // and thus a zero is enforced.
                let max_length = 34i32 + 1;
                for (tag, range) in [
                    (FixedTableTag::RangeKeyLen256, 256),
                    (FixedTableTag::RangeKeyLen16, 16),
                ] {
                    for n in -512..max_length {
                        let range = if n < 0 { 1 } else { range };
                        for idx in 0..range {
                            let v = F::from(n.unsigned_abs() as u64)
                                * if n.is_negative() { -F::one() } else { F::one() };
                            assignf!(region, (self.fixed_table[0], offset) => tag.scalar())?;
                            assignf!(region, (self.fixed_table[1], offset) => idx.scalar())?;
                            assignf!(region, (self.fixed_table[2], offset) => v)?;
                            offset += 1;
                        }
                    }
                }

                // Nibble range table
                for ind in 0..16 {
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::Range16.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    offset += 1;
                }

                // Rlp prefixes table [rlp_tag, byte, is_string, is_short, is_verylong]
                for ind in 0..=127 {
                    // short string
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => true.scalar())?;
                    assignf!(region, (self.fixed_table[3], offset) => true.scalar())?;
                    assignf!(region, (self.fixed_table[4], offset) => false.scalar())?;
                    offset += 1;
                }
                for ind in 128..=183 {
                    // long string
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => true.scalar())?;
                    assignf!(region, (self.fixed_table[3], offset) => false.scalar())?;
                    assignf!(region, (self.fixed_table[4], offset) => false.scalar())?;
                    offset += 1;
                }
                for ind in 184..=191 {
                    // very long string
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => true.scalar())?;
                    assignf!(region, (self.fixed_table[3], offset) => false.scalar())?;
                    assignf!(region, (self.fixed_table[4], offset) => true.scalar())?;
                    offset += 1;
                }
                for ind in 192..=247 {
                    // short list
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ind.scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => false.scalar())?;
                    assignf!(region, (self.fixed_table[3], offset) => true.scalar())?;
                    assignf!(region, (self.fixed_table[4], offset) => false.scalar())?;
                    offset += 1;
                }
                // 248
                // long list
                assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                assignf!(region, (self.fixed_table[1], offset) => 248i32.scalar())?;
                assignf!(region, (self.fixed_table[2], offset) => false.scalar())?;
                assignf!(region, (self.fixed_table[3], offset) => false.scalar())?;
                assignf!(region, (self.fixed_table[4], offset) => false.scalar())?;
                offset += 1;
                // 249
                // very long list
                assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::RLP.scalar())?;
                assignf!(region, (self.fixed_table[1], offset) => 249i32.scalar())?;
                assignf!(region, (self.fixed_table[2], offset) => false.scalar())?;
                assignf!(region, (self.fixed_table[3], offset) => false.scalar())?;
                assignf!(region, (self.fixed_table[4], offset) => true.scalar())?;
                offset += 1;

                // Even - only the nibbles 0 0 are valid
                assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::ExtOddKey.scalar())?;
                assignf!(region, (self.fixed_table[1], offset) => 0.scalar())?;
                assignf!(region, (self.fixed_table[2], offset) => false.scalar())?;
                offset += 1;

                // Odd - First nibble is 1, the second nibble can be any value
                for idx in 0..16 {
                    assignf!(region, (self.fixed_table[0], offset) => FixedTableTag::ExtOddKey.scalar())?;
                    assignf!(region, (self.fixed_table[1], offset) => ((0b1_0000) + idx).scalar())?;
                    assignf!(region, (self.fixed_table[2], offset) => true.scalar())?;
                    offset += 1;
                }
                
                Ok(())
            },
        )
    }
}

#[derive(Default)]
struct MPTCircuit<F> {
    witness: Vec<Vec<u8>>,
    randomness: F,
}

impl<F: Field> Circuit<F> for MPTCircuit<F> {
    type Config = MPTConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let keccak_table = KeccakTable::construct(meta);
        let power_of_randomness: [Expression<F>; HASH_WIDTH] =
            power_of_randomness_from_instance(meta);
        MPTConfig::configure(meta, power_of_randomness, keccak_table)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let mut to_be_hashed = vec![];

        let mut witness_rows = vec![];
        for row in self.witness.iter() {
            if row[row.len() - 1] == 5 {
                to_be_hashed.push(row[0..row.len() - 1].to_vec());
            } else {
                let row = MptWitnessRow::new(row[0..row.len()].to_vec());
                witness_rows.push(row);
            }
        }

        config.load_fixed_table(&mut layouter, self.randomness)?;
        config.assign(&mut layouter, &mut witness_rows, self.randomness)?;

        let challenges = Challenges::mock(Value::known(self.randomness));
        config
            .keccak_table
            .dev_load(&mut layouter, &to_be_hashed, &challenges, false)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use halo2_proofs::{
        dev::MockProver,
        halo2curves::{bn256::Fr, FieldExt},
    };
    use std::{fs, env::VarError};

    #[test]
    fn test_mpt() {
        let only_run = var("ONLY_RUN")
            .and_then(|idx| idx.parse::<usize>().map_err(|e|VarError::NotPresent)
        ).ok();
        println!("ONLY_RUN={:?}", only_run);
        // for debugging:
        let path = "src/mpt_circuit/tests";
        // let path = "tests";
        let files = fs::read_dir(path).unwrap();
        files
            .filter_map(Result::ok)
            .filter(|d| {
                if let Some(e) = d.path().extension() {
                    e == "json"
                } else {
                    false
                }
            })
            .enumerate()
            .for_each(|(idx, f)| {
                let mut run = true;
                if let Some(i) = only_run {
                    if idx != i {run = false;} 
                }
                if run {
                    let path = f.path();
                    let mut parts = path.to_str().unwrap().split('-');
                    parts.next();
                    let file = std::fs::File::open(path.clone());
                    let reader = std::io::BufReader::new(file.unwrap());
                    let w: Vec<Vec<u8>> = serde_json::from_reader(reader).unwrap();
    
                    let count = w.iter().filter(|r| r[r.len() - 1] != 5).count() * 2;
                    let randomness: Fr = 123456789.scalar();
                    let instance: Vec<Vec<Fr>> = (1..HASH_WIDTH + 1)
                        .map(|exp| vec![randomness.pow(&[exp as u64, 0, 0, 0]); count])
                        .collect();
    
                    let circuit = MPTCircuit::<Fr> {
                        witness: w.clone(),
                        randomness,
                    };
    
                    println!("{} {:?}", idx, path);
                    // let prover = MockProver::run(9, &circuit, vec![pub_root]).unwrap();
                    let num_rows = w.len() * 2;
                    let prover = MockProver::run(14 /* 9 */, &circuit, instance).unwrap();
                    assert_eq!(prover.verify_at_rows(0..num_rows, 0..num_rows,), Ok(()));
                    //assert_eq!(prover.verify_par(), Ok(()));
                    //prover.assert_satisfied();       
                }
            });
    }


    #[test]
    fn test_mpt2() {
        let wit = "[[0,1,0,1,249,2,17,249,2,17,15,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,160,215,178,43,142,72,221,147,48,230,157,99,126,109,240,144,184,54,167,1,19,157,71,126,226,97,100,220,221,118,5,202,114,0,160,215,178,43,142,72,221,147,48,230,157,99,126,109,240,144,184,54,167,1,19,157,71,126,226,97,100,220,221,118,5,202,114,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,195,19,38,251,242,179,135,46,118,82,177,213,78,156,167,171,134,95,6,233,153,168,219,176,131,34,215,213,95,252,168,165,0,160,195,19,38,251,242,179,135,46,118,82,177,213,78,156,167,171,134,95,6,233,153,168,219,176,131,34,215,213,95,252,168,165,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,242,119,75,182,209,50,158,172,168,239,218,202,172,144,155,94,44,154,149,92,253,83,150,12,4,176,33,46,25,36,170,225,0,160,242,119,75,182,209,50,158,172,168,239,218,202,172,144,155,94,44,154,149,92,253,83,150,12,4,176,33,46,25,36,170,225,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,156,18,160,254,15,159,132,100,9,112,178,98,98,93,76,54,189,166,63,219,45,193,25,238,218,78,235,150,206,67,252,253,0,160,156,18,160,254,15,159,132,100,9,112,178,98,98,93,76,54,189,166,63,219,45,193,25,238,218,78,235,150,206,67,252,253,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,100,75,99,34,122,170,85,172,143,62,172,142,154,219,40,105,162,136,113,194,41,38,129,211,105,114,94,62,145,244,97,170,0,160,100,75,99,34,122,170,85,172,143,62,172,142,154,219,40,105,162,136,113,194,41,38,129,211,105,114,94,62,145,244,97,170,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,229,94,177,9,226,34,180,156,33,249,119,163,74,194,218,172,92,90,13,44,22,231,5,100,72,203,19,192,62,46,34,34,0,160,229,94,177,9,226,34,180,156,33,249,119,163,74,194,218,172,92,90,13,44,22,231,5,100,72,203,19,192,62,46,34,34,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,15,175,179,52,244,84,197,105,112,43,252,115,186,76,237,251,88,5,62,201,157,9,7,153,100,224,202,249,250,183,125,248,0,160,15,175,179,52,244,84,197,105,112,43,252,115,186,76,237,251,88,5,62,201,157,9,7,153,100,224,202,249,250,183,125,248,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,14,229,239,45,75,116,39,109,41,89,200,43,18,94,204,133,62,175,23,200,68,93,170,95,36,226,233,183,66,98,37,184,0,160,14,229,239,45,75,116,39,109,41,89,200,43,18,94,204,133,62,175,23,200,68,93,170,95,36,226,233,183,66,98,37,184,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,6,197,49,201,57,39,248,81,26,196,11,167,230,243,100,223,97,38,20,1,226,39,180,161,172,204,67,80,173,223,89,42,0,160,6,197,49,201,57,39,248,81,26,196,11,167,230,243,100,223,97,38,20,1,226,39,180,161,172,204,67,80,173,223,89,42,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,3,131,195,206,124,22,207,14,142,91,216,135,77,202,69,1,53,115,223,85,52,95,43,227,237,82,138,95,93,70,227,232,0,160,3,131,195,206,124,22,207,14,142,91,216,135,77,202,69,1,53,115,223,85,52,95,43,227,237,82,138,95,93,70,227,232,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,98,109,64,32,201,140,205,221,164,1,209,57,84,209,249,108,87,101,70,12,37,160,114,139,27,145,104,130,62,183,150,108,0,160,98,109,64,32,201,140,205,221,164,1,209,57,84,209,249,108,87,101,70,12,37,160,114,139,27,145,104,130,62,183,150,108,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,69,221,169,92,165,83,34,53,123,93,55,127,206,167,112,175,13,233,196,118,68,137,156,246,219,49,159,137,25,37,30,157,0,160,69,221,169,92,165,83,34,53,123,93,55,127,206,167,112,175,13,233,196,118,68,137,156,246,219,49,159,137,25,37,30,157,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,39,24,29,240,236,191,237,195,74,255,251,61,19,232,218,181,111,83,69,125,70,208,135,182,81,0,125,85,38,21,25,11,0,160,39,24,29,240,236,191,237,195,74,255,251,61,19,232,218,181,111,83,69,125,70,208,135,182,81,0,125,85,38,21,25,11,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,191,249,76,252,217,172,58,95,133,138,144,243,9,87,191,253,23,150,215,186,153,214,27,17,128,10,154,202,202,43,193,173,0,160,191,249,76,252,217,172,58,95,133,138,144,243,9,87,191,253,23,150,215,186,153,214,27,17,128,10,154,202,202,43,193,173,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,238,147,22,82,116,71,41,238,84,0,62,40,0,153,205,90,194,234,61,255,205,197,55,0,41,239,197,174,219,163,6,130,0,160,238,147,22,82,116,71,41,238,84,0,62,40,0,153,205,90,194,234,61,255,205,197,55,0,41,239,197,174,219,163,6,130,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,160,22,99,129,222,131,163,115,40,32,94,210,97,181,141,77,173,9,184,214,164,50,44,139,113,241,255,7,213,43,8,145,41,0,160,245,44,16,35,247,198,201,190,127,121,84,12,180,160,45,116,180,243,5,121,76,186,165,227,121,49,165,118,171,248,179,191,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,17],[0,1,0,1,249,2,17,249,2,17,12,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,62,81,246,216,50,34,109,164,244,230,118,34,30,57,143,168,201,163,53,157,138,200,83,22,217,54,9,12,142,74,113,119,0,160,62,81,246,216,50,34,109,164,244,230,118,34,30,57,143,168,201,163,53,157,138,200,83,22,217,54,9,12,142,74,113,119,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,110,58,129,42,7,242,95,48,22,117,24,235,247,115,27,159,148,240,189,82,238,245,24,104,4,88,96,153,87,62,124,87,0,160,110,58,129,42,7,242,95,48,22,117,24,235,247,115,27,159,148,240,189,82,238,245,24,104,4,88,96,153,87,62,124,87,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,23,125,132,126,57,77,125,108,183,122,223,20,200,11,140,116,8,197,125,77,164,229,34,187,130,255,11,5,123,106,18,226,0,160,23,125,132,126,57,77,125,108,183,122,223,20,200,11,140,116,8,197,125,77,164,229,34,187,130,255,11,5,123,106,18,226,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,34,142,142,168,173,99,151,78,20,223,49,167,173,193,155,22,185,57,77,94,140,81,219,8,205,119,152,79,221,53,32,207,0,160,34,142,142,168,173,99,151,78,20,223,49,167,173,193,155,22,185,57,77,94,140,81,219,8,205,119,152,79,221,53,32,207,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,47,68,110,156,219,87,42,252,153,153,146,44,71,68,241,42,115,255,196,172,147,166,193,209,18,197,101,149,78,252,80,101,0,160,47,68,110,156,219,87,42,252,153,153,146,44,71,68,241,42,115,255,196,172,147,166,193,209,18,197,101,149,78,252,80,101,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,177,72,39,32,231,101,153,80,19,87,43,118,153,180,111,107,120,208,26,121,107,122,223,78,248,252,112,146,129,121,183,249,0,160,177,72,39,32,231,101,153,80,19,87,43,118,153,180,111,107,120,208,26,121,107,122,223,78,248,252,112,146,129,121,183,249,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,166,43,253,9,132,156,69,206,240,132,245,215,140,18,136,28,76,115,232,35,55,46,97,106,111,29,136,215,243,244,104,17,0,160,166,43,253,9,132,156,69,206,240,132,245,215,140,18,136,28,76,115,232,35,55,46,97,106,111,29,136,215,243,244,104,17,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,81,197,227,235,119,136,89,180,1,44,243,14,3,35,252,32,39,239,6,187,20,67,5,160,124,3,15,223,92,185,169,242,0,160,81,197,227,235,119,136,89,180,1,44,243,14,3,35,252,32,39,239,6,187,20,67,5,160,124,3,15,223,92,185,169,242,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,149,116,118,68,110,184,206,46,175,107,154,14,16,171,116,53,96,139,243,244,119,49,149,255,105,200,203,196,178,219,6,82,0,160,149,116,118,68,110,184,206,46,175,107,154,14,16,171,116,53,96,139,243,244,119,49,149,255,105,200,203,196,178,219,6,82,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,110,155,225,108,252,161,111,115,34,161,168,254,20,210,73,55,53,84,44,62,235,227,145,125,56,152,100,115,68,140,102,72,0,160,110,155,225,108,252,161,111,115,34,161,168,254,20,210,73,55,53,84,44,62,235,227,145,125,56,152,100,115,68,140,102,72,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,215,32,96,214,100,112,201,119,187,39,102,186,145,221,83,195,0,96,163,123,49,150,62,117,25,68,3,71,226,217,71,5,0,160,215,32,96,214,100,112,201,119,187,39,102,186,145,221,83,195,0,96,163,123,49,150,62,117,25,68,3,71,226,217,71,5,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,216,77,122,185,166,60,76,175,212,143,31,218,53,223,132,60,243,170,247,163,51,217,81,184,10,173,42,95,228,91,232,94,0,160,216,77,122,185,166,60,76,175,212,143,31,218,53,223,132,60,243,170,247,163,51,217,81,184,10,173,42,95,228,91,232,94,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,102,201,63,187,90,100,67,28,66,169,235,107,142,189,159,208,34,47,59,148,229,29,242,190,206,105,91,103,217,108,220,3,0,160,215,59,125,216,248,19,201,18,250,183,66,8,213,171,232,118,28,224,37,86,154,26,45,63,50,37,179,78,95,7,56,114,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,130,214,88,212,94,182,185,119,29,142,174,223,89,224,222,59,117,224,157,226,110,51,196,90,175,27,158,67,104,153,87,154,0,160,130,214,88,212,94,182,185,119,29,142,174,223,89,224,222,59,117,224,157,226,110,51,196,90,175,27,158,67,104,153,87,154,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,32,215,128,174,120,163,233,16,26,195,200,23,79,233,122,253,170,114,110,149,85,164,70,233,70,156,107,147,254,209,174,11,0,160,32,215,128,174,120,163,233,16,26,195,200,23,79,233,122,253,170,114,110,149,85,164,70,233,70,156,107,147,254,209,174,11,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,193,92,89,242,135,228,231,76,186,23,253,187,202,250,103,245,131,178,24,164,47,129,106,19,179,50,117,153,14,62,38,242,0,160,193,92,89,242,135,228,231,76,186,23,253,187,202,250,103,245,131,178,24,164,47,129,106,19,179,50,117,153,14,62,38,242,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[0,1,0,1,249,2,17,249,2,17,14,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,243,248,117,132,135,179,242,217,170,170,147,202,41,30,49,202,235,19,91,182,154,115,189,49,71,95,213,18,134,202,205,168,0,160,243,248,117,132,135,179,242,217,170,170,147,202,41,30,49,202,235,19,91,182,154,115,189,49,71,95,213,18,134,202,205,168,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,248,214,27,163,48,89,69,124,39,200,95,223,46,31,254,156,7,133,69,242,252,178,116,213,90,11,24,2,233,210,95,159,0,160,248,214,27,163,48,89,69,124,39,200,95,223,46,31,254,156,7,133,69,242,252,178,116,213,90,11,24,2,233,210,95,159,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,71,79,93,73,42,149,57,51,225,103,235,189,110,188,70,240,193,23,63,219,116,110,243,110,149,7,153,50,68,75,57,255,0,160,71,79,93,73,42,149,57,51,225,103,235,189,110,188,70,240,193,23,63,219,116,110,243,110,149,7,153,50,68,75,57,255,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,247,151,249,133,151,231,1,67,135,51,218,198,210,129,152,142,23,144,43,153,113,15,227,167,123,200,117,134,246,144,41,89,0,160,247,151,249,133,151,231,1,67,135,51,218,198,210,129,152,142,23,144,43,153,113,15,227,167,123,200,117,134,246,144,41,89,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,64,41,80,117,176,86,114,226,155,222,42,78,189,238,210,98,213,168,109,98,43,187,53,78,43,64,239,233,108,49,103,145,0,160,64,41,80,117,176,86,114,226,155,222,42,78,189,238,210,98,213,168,109,98,43,187,53,78,43,64,239,233,108,49,103,145,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,81,130,213,232,226,141,41,38,167,145,141,254,200,67,223,12,25,155,31,46,162,105,182,222,2,233,159,55,73,58,81,8,0,160,81,130,213,232,226,141,41,38,167,145,141,254,200,67,223,12,25,155,31,46,162,105,182,222,2,233,159,55,73,58,81,8,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,26,77,61,137,205,196,204,129,210,129,10,70,241,189,76,121,69,162,6,215,188,152,126,170,249,149,72,157,147,95,113,240,0,160,26,77,61,137,205,196,204,129,210,129,10,70,241,189,76,121,69,162,6,215,188,152,126,170,249,149,72,157,147,95,113,240,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,76,38,106,162,112,134,133,37,202,39,149,12,3,108,165,104,174,60,185,97,253,218,30,38,19,121,89,102,165,245,8,216,0,160,76,38,106,162,112,134,133,37,202,39,149,12,3,108,165,104,174,60,185,97,253,218,30,38,19,121,89,102,165,245,8,216,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,52,133,159,70,24,244,195,146,2,17,195,222,204,211,129,28,126,50,191,31,4,148,37,228,107,40,143,95,15,239,188,142,0,160,52,133,159,70,24,244,195,146,2,17,195,222,204,211,129,28,126,50,191,31,4,148,37,228,107,40,143,95,15,239,188,142,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,30,36,232,235,34,123,13,63,36,1,92,189,49,255,191,152,101,81,234,45,170,78,228,224,41,77,6,235,75,41,95,228,0,160,30,36,232,235,34,123,13,63,36,1,92,189,49,255,191,152,101,81,234,45,170,78,228,224,41,77,6,235,75,41,95,228,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,53,118,183,248,110,40,9,128,63,159,4,146,88,128,57,182,207,231,204,72,18,102,249,225,183,253,26,165,204,221,133,107,0,160,53,118,183,248,110,40,9,128,63,159,4,146,88,128,57,182,207,231,204,72,18,102,249,225,183,253,26,165,204,221,133,107,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,47,211,121,43,20,152,236,53,207,92,248,102,254,75,207,136,49,232,147,125,59,184,15,14,62,136,58,5,56,132,135,139,0,160,47,211,121,43,20,152,236,53,207,92,248,102,254,75,207,136,49,232,147,125,59,184,15,14,62,136,58,5,56,132,135,139,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,146,83,193,207,181,29,203,52,47,58,114,231,161,55,66,1,75,127,145,210,118,37,82,232,135,3,183,30,255,240,248,11,0,160,146,83,193,207,181,29,203,52,47,58,114,231,161,55,66,1,75,127,145,210,118,37,82,232,135,3,183,30,255,240,248,11,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,74,111,171,71,106,196,108,204,154,49,109,206,164,6,195,104,55,35,226,133,78,86,140,154,197,163,105,253,218,72,68,52,0,160,74,111,171,71,106,196,108,204,154,49,109,206,164,6,195,104,55,35,226,133,78,86,140,154,197,163,105,253,218,72,68,52,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,221,162,147,55,34,170,139,142,218,244,84,132,181,168,39,246,188,198,8,193,144,16,119,237,138,12,69,220,76,152,153,153,0,160,142,150,91,3,151,108,7,234,211,182,207,220,110,244,217,170,240,12,90,203,195,243,210,201,172,9,15,30,76,40,83,183,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,250,100,42,85,168,208,19,121,181,167,41,37,110,73,50,34,56,59,218,49,242,70,153,106,217,4,105,151,51,36,134,125,0,160,250,100,42,85,168,208,19,121,181,167,41,37,110,73,50,34,56,59,218,49,242,70,153,106,217,4,105,151,51,36,134,125,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[0,1,0,1,249,2,17,249,2,17,13,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,161,128,1,12,47,16,128,58,172,109,97,186,101,50,211,24,116,166,152,209,189,185,191,39,125,163,235,50,169,86,158,229,0,160,161,128,1,12,47,16,128,58,172,109,97,186,101,50,211,24,116,166,152,209,189,185,191,39,125,163,235,50,169,86,158,229,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,43,180,202,90,212,89,144,118,139,227,102,232,30,186,65,236,181,5,130,247,53,26,255,110,32,164,81,96,121,240,13,252,0,160,43,180,202,90,212,89,144,118,139,227,102,232,30,186,65,236,181,5,130,247,53,26,255,110,32,164,81,96,121,240,13,252,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,222,100,177,157,75,143,240,145,16,36,58,46,51,139,107,7,196,233,64,182,153,253,203,175,129,102,22,111,153,168,150,26,0,160,222,100,177,157,75,143,240,145,16,36,58,46,51,139,107,7,196,233,64,182,153,253,203,175,129,102,22,111,153,168,150,26,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,94,231,69,12,111,229,77,99,71,17,141,11,41,112,27,177,218,61,40,30,213,193,247,27,173,123,94,162,194,11,64,110,0,160,94,231,69,12,111,229,77,99,71,17,141,11,41,112,27,177,218,61,40,30,213,193,247,27,173,123,94,162,194,11,64,110,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,148,241,5,249,211,104,221,226,140,197,193,238,210,173,105,8,129,244,154,57,13,253,109,216,177,158,110,36,172,122,110,88,0,160,148,241,5,249,211,104,221,226,140,197,193,238,210,173,105,8,129,244,154,57,13,253,109,216,177,158,110,36,172,122,110,88,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,49,101,31,195,122,182,161,106,170,190,126,247,114,74,123,53,20,100,9,186,33,38,17,167,168,229,10,220,151,18,196,241,0,160,49,101,31,195,122,182,161,106,170,190,126,247,114,74,123,53,20,100,9,186,33,38,17,167,168,229,10,220,151,18,196,241,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,73,246,226,153,120,139,128,58,10,194,85,4,186,39,18,220,239,252,50,159,22,196,125,122,103,50,247,196,37,68,58,169,0,160,73,246,226,153,120,139,128,58,10,194,85,4,186,39,18,220,239,252,50,159,22,196,125,122,103,50,247,196,37,68,58,169,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,15,132,108,63,247,99,185,92,140,54,8,64,230,186,45,30,61,193,8,165,18,74,107,200,87,45,33,232,22,58,219,43,0,160,15,132,108,63,247,99,185,92,140,54,8,64,230,186,45,30,61,193,8,165,18,74,107,200,87,45,33,232,22,58,219,43,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,171,6,213,180,15,77,228,71,174,54,254,251,111,241,218,40,233,3,107,112,164,163,132,133,85,121,0,128,188,237,176,38,0,160,171,6,213,180,15,77,228,71,174,54,254,251,111,241,218,40,233,3,107,112,164,163,132,133,85,121,0,128,188,237,176,38,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,190,202,142,180,181,1,250,241,49,215,108,185,216,23,205,142,139,158,85,162,252,156,118,150,43,152,194,183,178,218,159,221,0,160,190,202,142,180,181,1,250,241,49,215,108,185,216,23,205,142,139,158,85,162,252,156,118,150,43,152,194,183,178,218,159,221,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,116,118,116,254,111,169,77,111,65,32,203,133,193,209,164,92,7,21,222,137,239,153,10,17,202,156,229,253,242,229,50,66,0,160,116,118,116,254,111,169,77,111,65,32,203,133,193,209,164,92,7,21,222,137,239,153,10,17,202,156,229,253,242,229,50,66,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,245,182,9,212,150,185,219,26,154,17,0,141,168,125,166,152,114,219,87,156,42,77,206,233,29,211,176,18,46,29,86,118,0,160,245,182,9,212,150,185,219,26,154,17,0,141,168,125,166,152,114,219,87,156,42,77,206,233,29,211,176,18,46,29,86,118,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,125,173,37,34,63,10,10,105,35,138,170,159,170,58,203,218,96,174,159,130,118,216,137,144,59,203,221,237,109,28,197,14,0,160,125,173,37,34,63,10,10,105,35,138,170,159,170,58,203,218,96,174,159,130,118,216,137,144,59,203,221,237,109,28,197,14,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,125,205,12,44,38,14,115,188,176,89,248,149,162,236,64,246,24,91,125,70,183,125,37,100,214,54,174,74,207,71,185,190,0,160,77,232,92,90,207,144,28,209,255,114,73,30,34,22,65,146,193,168,52,246,172,111,139,107,21,220,140,195,72,109,174,142,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,9,167,144,133,57,89,194,210,118,41,249,242,60,234,105,179,15,125,163,86,11,161,61,242,89,222,67,163,239,141,115,22,0,160,9,167,144,133,57,89,194,210,118,41,249,242,60,234,105,179,15,125,163,86,11,161,61,242,89,222,67,163,239,141,115,22,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,229,254,113,96,76,247,87,54,147,166,26,241,48,108,149,89,115,6,35,119,201,191,233,239,90,99,195,93,22,222,43,126,0,160,229,254,113,96,76,247,87,54,147,166,26,241,48,108,149,89,115,6,35,119,201,191,233,239,90,99,195,93,22,222,43,126,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[0,1,0,1,249,2,17,249,2,17,3,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,1,140,240,73,75,204,201,222,156,243,213,94,33,163,61,8,206,249,37,116,57,74,97,38,98,157,136,8,8,58,80,150,0,160,1,140,240,73,75,204,201,222,156,243,213,94,33,163,61,8,206,249,37,116,57,74,97,38,98,157,136,8,8,58,80,150,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,195,199,196,181,116,185,85,110,248,215,152,211,207,168,41,60,203,5,86,141,59,163,78,219,9,213,111,185,55,120,19,233,0,160,195,199,196,181,116,185,85,110,248,215,152,211,207,168,41,60,203,5,86,141,59,163,78,219,9,213,111,185,55,120,19,233,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,140,202,218,220,242,107,140,113,118,132,7,69,53,214,70,230,137,184,171,129,43,48,107,81,80,73,247,0,177,229,219,121,0,160,140,202,218,220,242,107,140,113,118,132,7,69,53,214,70,230,137,184,171,129,43,48,107,81,80,73,247,0,177,229,219,121,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,33,36,191,50,11,87,222,33,182,77,167,63,136,123,248,241,74,182,24,11,174,247,239,125,99,202,207,255,128,35,52,165,0,160,224,226,228,152,49,191,94,18,202,42,40,43,44,135,217,234,139,112,115,185,228,196,213,42,168,79,181,143,79,201,143,96,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,149,249,75,250,81,105,43,241,150,173,198,3,252,180,149,96,0,111,180,34,118,196,43,123,93,132,160,96,250,100,217,45,0,160,149,249,75,250,81,105,43,241,150,173,198,3,252,180,149,96,0,111,180,34,118,196,43,123,93,132,160,96,250,100,217,45,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,32,39,96,173,133,195,109,50,97,77,73,185,128,89,4,150,255,132,58,164,43,120,193,117,186,32,133,65,91,116,162,173,0,160,32,39,96,173,133,195,109,50,97,77,73,185,128,89,4,150,255,132,58,164,43,120,193,117,186,32,133,65,91,116,162,173,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,98,239,63,98,146,213,134,176,5,254,159,193,14,251,162,124,237,62,243,94,97,73,108,47,3,76,184,133,162,93,214,124,0,160,98,239,63,98,146,213,134,176,5,254,159,193,14,251,162,124,237,62,243,94,97,73,108,47,3,76,184,133,162,93,214,124,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,24,110,66,31,239,73,37,228,27,69,165,214,234,132,223,109,118,39,20,166,141,25,228,24,156,85,122,60,112,195,235,154,0,160,24,110,66,31,239,73,37,228,27,69,165,214,234,132,223,109,118,39,20,166,141,25,228,24,156,85,122,60,112,195,235,154,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,124,4,254,255,41,243,241,33,206,19,170,136,141,252,149,202,221,147,172,85,213,237,197,110,71,174,111,101,127,85,205,59,0,160,124,4,254,255,41,243,241,33,206,19,170,136,141,252,149,202,221,147,172,85,213,237,197,110,71,174,111,101,127,85,205,59,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,23,49,65,202,234,196,28,65,205,115,198,37,246,143,124,72,166,37,205,232,162,25,22,39,127,188,14,26,18,214,240,152,0,160,23,49,65,202,234,196,28,65,205,115,198,37,246,143,124,72,166,37,205,232,162,25,22,39,127,188,14,26,18,214,240,152,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,51,21,215,92,255,202,104,15,118,167,53,140,39,4,142,82,127,133,147,230,204,253,47,54,99,23,226,78,113,129,89,185,0,160,51,21,215,92,255,202,104,15,118,167,53,140,39,4,142,82,127,133,147,230,204,253,47,54,99,23,226,78,113,129,89,185,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,239,123,251,65,188,11,107,22,9,46,42,104,47,193,60,78,205,118,242,12,136,145,137,46,214,157,184,26,255,37,206,38,0,160,239,123,251,65,188,11,107,22,9,46,42,104,47,193,60,78,205,118,242,12,136,145,137,46,214,157,184,26,255,37,206,38,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,179,163,103,185,250,53,96,32,14,9,248,46,117,61,151,70,245,116,155,44,163,22,115,1,102,242,244,157,45,81,102,14,0,160,179,163,103,185,250,53,96,32,14,9,248,46,117,61,151,70,245,116,155,44,163,22,115,1,102,242,244,157,45,81,102,14,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,120,51,181,75,204,140,26,229,78,145,104,6,122,193,149,189,178,100,84,118,214,32,148,10,91,248,41,39,153,51,148,250,0,160,120,51,181,75,204,140,26,229,78,145,104,6,122,193,149,189,178,100,84,118,214,32,148,10,91,248,41,39,153,51,148,250,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,152,64,28,199,229,115,92,129,39,229,199,166,105,168,252,23,227,109,56,225,3,255,171,233,92,155,115,43,225,156,231,35,0,160,152,64,28,199,229,115,92,129,39,229,199,166,105,168,252,23,227,109,56,225,3,255,171,233,92,155,115,43,225,156,231,35,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,79,205,115,234,146,184,235,250,60,154,252,244,30,28,214,37,12,114,43,159,140,167,245,162,159,65,188,1,113,43,38,143,0,160,79,205,115,234,146,184,235,250,60,154,252,244,30,28,214,37,12,114,43,159,140,167,245,162,159,65,188,1,113,43,38,143,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[0,1,0,1,249,2,17,249,2,17,4,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,245,105,73,55,130,156,85,160,31,141,126,218,15,74,121,147,147,14,234,12,31,2,207,74,132,213,9,173,180,149,183,107,0,160,245,105,73,55,130,156,85,160,31,141,126,218,15,74,121,147,147,14,234,12,31,2,207,74,132,213,9,173,180,149,183,107,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,221,130,180,81,176,155,200,100,168,4,254,92,101,171,36,147,95,202,31,177,191,39,28,78,15,253,236,77,124,115,149,137,0,160,221,130,180,81,176,155,200,100,168,4,254,92,101,171,36,147,95,202,31,177,191,39,28,78,15,253,236,77,124,115,149,137,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,8,37,3,52,123,198,42,148,211,79,179,98,105,89,161,130,151,2,137,5,198,34,114,85,180,47,176,126,179,111,60,206,0,160,8,37,3,52,123,198,42,148,211,79,179,98,105,89,161,130,151,2,137,5,198,34,114,85,180,47,176,126,179,111,60,206,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,232,217,161,8,22,169,40,66,131,228,203,23,191,255,11,201,101,138,145,67,49,60,150,125,179,56,59,152,181,26,174,138,0,160,232,217,161,8,22,169,40,66,131,228,203,23,191,255,11,201,101,138,145,67,49,60,150,125,179,56,59,152,181,26,174,138,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,52,5,132,223,20,125,125,152,77,21,29,239,159,211,65,174,156,121,107,233,188,67,44,242,54,70,100,18,159,243,207,206,0,160,46,144,237,143,190,109,159,205,233,64,197,103,36,172,203,35,39,122,25,184,212,193,136,197,175,120,207,44,140,182,105,222,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,224,250,15,5,72,187,58,246,78,193,251,188,67,94,94,63,151,23,215,194,99,44,14,23,45,34,254,220,3,94,41,58,0,160,224,250,15,5,72,187,58,246,78,193,251,188,67,94,94,63,151,23,215,194,99,44,14,23,45,34,254,220,3,94,41,58,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,6,105,60,12,5,193,169,245,176,112,146,23,69,42,0,33,177,13,230,213,165,102,152,203,58,175,135,4,16,128,172,8,0,160,6,105,60,12,5,193,169,245,176,112,146,23,69,42,0,33,177,13,230,213,165,102,152,203,58,175,135,4,16,128,172,8,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,151,152,86,204,166,248,67,223,250,77,31,100,237,11,43,191,90,23,20,54,199,92,11,215,145,50,87,90,167,159,57,165,0,160,151,152,86,204,166,248,67,223,250,77,31,100,237,11,43,191,90,23,20,54,199,92,11,215,145,50,87,90,167,159,57,165,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,1,232,26,222,47,23,75,176,90,187,251,204,93,173,132,158,36,225,142,226,147,28,202,173,168,228,182,229,123,127,49,117,0,160,1,232,26,222,47,23,75,176,90,187,251,204,93,173,132,158,36,225,142,226,147,28,202,173,168,228,182,229,123,127,49,117,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,247,78,159,238,239,170,53,113,45,18,48,98,112,234,117,104,97,108,138,230,14,76,168,84,236,172,64,67,208,57,6,73,0,160,247,78,159,238,239,170,53,113,45,18,48,98,112,234,117,104,97,108,138,230,14,76,168,84,236,172,64,67,208,57,6,73,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,46,68,248,146,220,227,97,0,41,252,210,9,44,117,251,227,165,196,13,189,174,150,34,139,203,17,200,40,245,122,167,206,0,160,46,68,248,146,220,227,97,0,41,252,210,9,44,117,251,227,165,196,13,189,174,150,34,139,203,17,200,40,245,122,167,206,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,26,216,79,208,48,103,203,251,178,213,93,58,82,104,200,119,234,228,233,252,208,91,195,35,224,229,183,69,89,175,14,229,0,160,26,216,79,208,48,103,203,251,178,213,93,58,82,104,200,119,234,228,233,252,208,91,195,35,224,229,183,69,89,175,14,229,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,246,209,205,0,213,119,26,186,142,93,94,61,153,28,165,149,49,176,155,119,213,241,208,245,15,163,38,131,125,219,108,170,0,160,246,209,205,0,213,119,26,186,142,93,94,61,153,28,165,149,49,176,155,119,213,241,208,245,15,163,38,131,125,219,108,170,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,89,143,190,130,47,255,40,170,85,219,138,46,139,251,126,68,17,241,5,216,204,86,127,71,120,116,170,149,237,137,28,227,0,160,89,143,190,130,47,255,40,170,85,219,138,46,139,251,126,68,17,241,5,216,204,86,127,71,120,116,170,149,237,137,28,227,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,79,246,250,96,218,39,3,222,92,140,84,169,44,51,184,140,136,139,201,154,119,208,207,98,29,112,62,108,254,3,142,180,0,160,79,246,250,96,218,39,3,222,92,140,84,169,44,51,184,140,136,139,201,154,119,208,207,98,29,112,62,108,254,3,142,180,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,130,179,74,86,218,213,192,18,132,24,134,63,237,50,86,187,20,97,174,221,173,83,84,97,186,105,52,78,209,101,251,138,0,160,130,179,74,86,218,213,192,18,132,24,134,63,237,50,86,187,20,97,174,221,173,83,84,97,186,105,52,78,209,101,251,138,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[1,0,1,0,248,241,0,248,241,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,160,255,151,217,75,103,5,122,115,224,137,233,146,50,189,95,178,178,247,44,237,22,101,231,39,198,40,14,249,60,251,151,15,0,160,188,253,144,87,144,251,204,78,148,203,12,141,0,77,176,70,67,92,90,100,110,40,255,28,218,97,116,184,26,121,18,49,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,60,79,85,51,115,192,158,157,93,223,211,100,62,94,72,146,251,82,116,111,190,139,246,12,252,146,211,122,66,110,206,20,0,160,60,79,85,51,115,192,158,157,93,223,211,100,62,94,72,146,251,82,116,111,190,139,246,12,252,146,211,122,66,110,206,20,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,120,190,160,200,253,109,255,226,49,189,87,112,136,160,23,77,119,59,173,185,188,145,251,156,155,144,100,217,100,114,109,106,0,160,120,190,160,200,253,109,255,226,49,189,87,112,136,160,23,77,119,59,173,185,188,145,251,156,155,144,100,217,100,114,109,106,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,69,72,113,186,79,146,63,86,46,218,1,200,131,76,71,142,217,35,30,209,101,239,91,47,163,221,136,130,249,155,236,112,0,160,69,72,113,186,79,146,63,86,46,218,1,200,131,76,71,142,217,35,30,209,101,239,91,47,163,221,136,130,249,155,236,112,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,49,65,26,94,193,156,227,78,42,198,56,211,105,254,0,33,31,96,41,208,40,13,215,156,51,173,132,112,34,192,121,49,0,160,49,65,26,94,193,156,227,78,42,198,56,211,105,254,0,33,31,96,41,208,40,13,215,156,51,173,132,112,34,192,121,49,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,244,154,252,18,232,96,245,36,84,15,253,182,157,226,247,165,106,144,166,1,2,140,228,170,110,87,112,80,140,149,162,43,0,160,244,154,252,18,232,96,245,36,84,15,253,182,157,226,247,165,106,144,166,1,2,140,228,170,110,87,112,80,140,149,162,43,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,20,103,6,95,163,140,21,238,207,84,226,60,134,0,183,217,11,213,185,123,139,201,37,22,227,234,220,30,160,20,244,115,0,160,20,103,6,95,163,140,21,238,207,84,226,60,134,0,183,217,11,213,185,123,139,201,37,22,227,234,220,30,160,20,244,115,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[1,0,1,0,248,81,0,248,81,0,8,1,0,7,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,222,45,71,217,199,68,20,55,244,206,68,197,49,191,78,208,106,209,111,87,254,9,221,230,148,86,131,219,7,121,62,140,0,160,222,45,71,217,199,68,20,55,244,206,68,197,49,191,78,208,106,209,111,87,254,9,221,230,148,86,131,219,7,121,62,140,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,160,190,214,56,80,83,126,135,17,104,48,181,30,249,223,80,59,155,70,206,67,24,6,82,98,81,246,212,143,253,181,15,180,0,160,190,214,56,80,83,126,135,17,104,48,181,30,249,223,80,59,155,70,206,67,24,6,82,98,81,246,212,143,253,181,15,180,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,16],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,17],[248,102,157,55,236,125,29,155,142,209,241,75,145,144,143,254,65,81,209,56,13,192,157,236,195,213,73,132,11,251,149,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,6],[248,102,157,32,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,4],[0,0,157,32,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,18],[184,70,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,68,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,7],[184,70,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,68,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,8],[0,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,0,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,9],[0,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,0,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,0,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,11],[248,102,157,32,236,125,29,155,142,209,241,75,145,144,143,254,65,81,209,56,13,192,157,236,195,213,73,132,11,251,149,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,92,69,153,141,251,249,206,112,188,187,128,87,78,215,166,34,146,45,44,119,94,10,35,49,254,90,139,141,204,153,244,144,242,246,191,23,44,167,166,154,14,14,27,198,200,66,149,155,102,162,36,92,147,76,227,228,141,122,139,186,245,89,5,41,252,237,52,8,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,10],[249,2,17,160,215,178,43,142,72,221,147,48,230,157,99,126,109,240,144,184,54,167,1,19,157,71,126,226,97,100,220,221,118,5,202,114,160,195,19,38,251,242,179,135,46,118,82,177,213,78,156,167,171,134,95,6,233,153,168,219,176,131,34,215,213,95,252,168,165,160,242,119,75,182,209,50,158,172,168,239,218,202,172,144,155,94,44,154,149,92,253,83,150,12,4,176,33,46,25,36,170,225,160,156,18,160,254,15,159,132,100,9,112,178,98,98,93,76,54,189,166,63,219,45,193,25,238,218,78,235,150,206,67,252,253,160,100,75,99,34,122,170,85,172,143,62,172,142,154,219,40,105,162,136,113,194,41,38,129,211,105,114,94,62,145,244,97,170,160,229,94,177,9,226,34,180,156,33,249,119,163,74,194,218,172,92,90,13,44,22,231,5,100,72,203,19,192,62,46,34,34,160,15,175,179,52,244,84,197,105,112,43,252,115,186,76,237,251,88,5,62,201,157,9,7,153,100,224,202,249,250,183,125,248,160,14,229,239,45,75,116,39,109,41,89,200,43,18,94,204,133,62,175,23,200,68,93,170,95,36,226,233,183,66,98,37,184,160,6,197,49,201,57,39,248,81,26,196,11,167,230,243,100,223,97,38,20,1,226,39,180,161,172,204,67,80,173,223,89,42,160,3,131,195,206,124,22,207,14,142,91,216,135,77,202,69,1,53,115,223,85,52,95,43,227,237,82,138,95,93,70,227,232,160,98,109,64,32,201,140,205,221,164,1,209,57,84,209,249,108,87,101,70,12,37,160,114,139,27,145,104,130,62,183,150,108,160,69,221,169,92,165,83,34,53,123,93,55,127,206,167,112,175,13,233,196,118,68,137,156,246,219,49,159,137,25,37,30,157,160,39,24,29,240,236,191,237,195,74,255,251,61,19,232,218,181,111,83,69,125,70,208,135,182,81,0,125,85,38,21,25,11,160,191,249,76,252,217,172,58,95,133,138,144,243,9,87,191,253,23,150,215,186,153,214,27,17,128,10,154,202,202,43,193,173,160,238,147,22,82,116,71,41,238,84,0,62,40,0,153,205,90,194,234,61,255,205,197,55,0,41,239,197,174,219,163,6,130,160,22,99,129,222,131,163,115,40,32,94,210,97,181,141,77,173,9,184,214,164,50,44,139,113,241,255,7,213,43,8,145,41,128,5],[249,2,17,160,215,178,43,142,72,221,147,48,230,157,99,126,109,240,144,184,54,167,1,19,157,71,126,226,97,100,220,221,118,5,202,114,160,195,19,38,251,242,179,135,46,118,82,177,213,78,156,167,171,134,95,6,233,153,168,219,176,131,34,215,213,95,252,168,165,160,242,119,75,182,209,50,158,172,168,239,218,202,172,144,155,94,44,154,149,92,253,83,150,12,4,176,33,46,25,36,170,225,160,156,18,160,254,15,159,132,100,9,112,178,98,98,93,76,54,189,166,63,219,45,193,25,238,218,78,235,150,206,67,252,253,160,100,75,99,34,122,170,85,172,143,62,172,142,154,219,40,105,162,136,113,194,41,38,129,211,105,114,94,62,145,244,97,170,160,229,94,177,9,226,34,180,156,33,249,119,163,74,194,218,172,92,90,13,44,22,231,5,100,72,203,19,192,62,46,34,34,160,15,175,179,52,244,84,197,105,112,43,252,115,186,76,237,251,88,5,62,201,157,9,7,153,100,224,202,249,250,183,125,248,160,14,229,239,45,75,116,39,109,41,89,200,43,18,94,204,133,62,175,23,200,68,93,170,95,36,226,233,183,66,98,37,184,160,6,197,49,201,57,39,248,81,26,196,11,167,230,243,100,223,97,38,20,1,226,39,180,161,172,204,67,80,173,223,89,42,160,3,131,195,206,124,22,207,14,142,91,216,135,77,202,69,1,53,115,223,85,52,95,43,227,237,82,138,95,93,70,227,232,160,98,109,64,32,201,140,205,221,164,1,209,57,84,209,249,108,87,101,70,12,37,160,114,139,27,145,104,130,62,183,150,108,160,69,221,169,92,165,83,34,53,123,93,55,127,206,167,112,175,13,233,196,118,68,137,156,246,219,49,159,137,25,37,30,157,160,39,24,29,240,236,191,237,195,74,255,251,61,19,232,218,181,111,83,69,125,70,208,135,182,81,0,125,85,38,21,25,11,160,191,249,76,252,217,172,58,95,133,138,144,243,9,87,191,253,23,150,215,186,153,214,27,17,128,10,154,202,202,43,193,173,160,238,147,22,82,116,71,41,238,84,0,62,40,0,153,205,90,194,234,61,255,205,197,55,0,41,239,197,174,219,163,6,130,160,245,44,16,35,247,198,201,190,127,121,84,12,180,160,45,116,180,243,5,121,76,186,165,227,121,49,165,118,171,248,179,191,128,5],[249,2,17,160,62,81,246,216,50,34,109,164,244,230,118,34,30,57,143,168,201,163,53,157,138,200,83,22,217,54,9,12,142,74,113,119,160,110,58,129,42,7,242,95,48,22,117,24,235,247,115,27,159,148,240,189,82,238,245,24,104,4,88,96,153,87,62,124,87,160,23,125,132,126,57,77,125,108,183,122,223,20,200,11,140,116,8,197,125,77,164,229,34,187,130,255,11,5,123,106,18,226,160,34,142,142,168,173,99,151,78,20,223,49,167,173,193,155,22,185,57,77,94,140,81,219,8,205,119,152,79,221,53,32,207,160,47,68,110,156,219,87,42,252,153,153,146,44,71,68,241,42,115,255,196,172,147,166,193,209,18,197,101,149,78,252,80,101,160,177,72,39,32,231,101,153,80,19,87,43,118,153,180,111,107,120,208,26,121,107,122,223,78,248,252,112,146,129,121,183,249,160,166,43,253,9,132,156,69,206,240,132,245,215,140,18,136,28,76,115,232,35,55,46,97,106,111,29,136,215,243,244,104,17,160,81,197,227,235,119,136,89,180,1,44,243,14,3,35,252,32,39,239,6,187,20,67,5,160,124,3,15,223,92,185,169,242,160,149,116,118,68,110,184,206,46,175,107,154,14,16,171,116,53,96,139,243,244,119,49,149,255,105,200,203,196,178,219,6,82,160,110,155,225,108,252,161,111,115,34,161,168,254,20,210,73,55,53,84,44,62,235,227,145,125,56,152,100,115,68,140,102,72,160,215,32,96,214,100,112,201,119,187,39,102,186,145,221,83,195,0,96,163,123,49,150,62,117,25,68,3,71,226,217,71,5,160,216,77,122,185,166,60,76,175,212,143,31,218,53,223,132,60,243,170,247,163,51,217,81,184,10,173,42,95,228,91,232,94,160,102,201,63,187,90,100,67,28,66,169,235,107,142,189,159,208,34,47,59,148,229,29,242,190,206,105,91,103,217,108,220,3,160,130,214,88,212,94,182,185,119,29,142,174,223,89,224,222,59,117,224,157,226,110,51,196,90,175,27,158,67,104,153,87,154,160,32,215,128,174,120,163,233,16,26,195,200,23,79,233,122,253,170,114,110,149,85,164,70,233,70,156,107,147,254,209,174,11,160,193,92,89,242,135,228,231,76,186,23,253,187,202,250,103,245,131,178,24,164,47,129,106,19,179,50,117,153,14,62,38,242,128,5],[249,2,17,160,62,81,246,216,50,34,109,164,244,230,118,34,30,57,143,168,201,163,53,157,138,200,83,22,217,54,9,12,142,74,113,119,160,110,58,129,42,7,242,95,48,22,117,24,235,247,115,27,159,148,240,189,82,238,245,24,104,4,88,96,153,87,62,124,87,160,23,125,132,126,57,77,125,108,183,122,223,20,200,11,140,116,8,197,125,77,164,229,34,187,130,255,11,5,123,106,18,226,160,34,142,142,168,173,99,151,78,20,223,49,167,173,193,155,22,185,57,77,94,140,81,219,8,205,119,152,79,221,53,32,207,160,47,68,110,156,219,87,42,252,153,153,146,44,71,68,241,42,115,255,196,172,147,166,193,209,18,197,101,149,78,252,80,101,160,177,72,39,32,231,101,153,80,19,87,43,118,153,180,111,107,120,208,26,121,107,122,223,78,248,252,112,146,129,121,183,249,160,166,43,253,9,132,156,69,206,240,132,245,215,140,18,136,28,76,115,232,35,55,46,97,106,111,29,136,215,243,244,104,17,160,81,197,227,235,119,136,89,180,1,44,243,14,3,35,252,32,39,239,6,187,20,67,5,160,124,3,15,223,92,185,169,242,160,149,116,118,68,110,184,206,46,175,107,154,14,16,171,116,53,96,139,243,244,119,49,149,255,105,200,203,196,178,219,6,82,160,110,155,225,108,252,161,111,115,34,161,168,254,20,210,73,55,53,84,44,62,235,227,145,125,56,152,100,115,68,140,102,72,160,215,32,96,214,100,112,201,119,187,39,102,186,145,221,83,195,0,96,163,123,49,150,62,117,25,68,3,71,226,217,71,5,160,216,77,122,185,166,60,76,175,212,143,31,218,53,223,132,60,243,170,247,163,51,217,81,184,10,173,42,95,228,91,232,94,160,215,59,125,216,248,19,201,18,250,183,66,8,213,171,232,118,28,224,37,86,154,26,45,63,50,37,179,78,95,7,56,114,160,130,214,88,212,94,182,185,119,29,142,174,223,89,224,222,59,117,224,157,226,110,51,196,90,175,27,158,67,104,153,87,154,160,32,215,128,174,120,163,233,16,26,195,200,23,79,233,122,253,170,114,110,149,85,164,70,233,70,156,107,147,254,209,174,11,160,193,92,89,242,135,228,231,76,186,23,253,187,202,250,103,245,131,178,24,164,47,129,106,19,179,50,117,153,14,62,38,242,128,5],[249,2,17,160,243,248,117,132,135,179,242,217,170,170,147,202,41,30,49,202,235,19,91,182,154,115,189,49,71,95,213,18,134,202,205,168,160,248,214,27,163,48,89,69,124,39,200,95,223,46,31,254,156,7,133,69,242,252,178,116,213,90,11,24,2,233,210,95,159,160,71,79,93,73,42,149,57,51,225,103,235,189,110,188,70,240,193,23,63,219,116,110,243,110,149,7,153,50,68,75,57,255,160,247,151,249,133,151,231,1,67,135,51,218,198,210,129,152,142,23,144,43,153,113,15,227,167,123,200,117,134,246,144,41,89,160,64,41,80,117,176,86,114,226,155,222,42,78,189,238,210,98,213,168,109,98,43,187,53,78,43,64,239,233,108,49,103,145,160,81,130,213,232,226,141,41,38,167,145,141,254,200,67,223,12,25,155,31,46,162,105,182,222,2,233,159,55,73,58,81,8,160,26,77,61,137,205,196,204,129,210,129,10,70,241,189,76,121,69,162,6,215,188,152,126,170,249,149,72,157,147,95,113,240,160,76,38,106,162,112,134,133,37,202,39,149,12,3,108,165,104,174,60,185,97,253,218,30,38,19,121,89,102,165,245,8,216,160,52,133,159,70,24,244,195,146,2,17,195,222,204,211,129,28,126,50,191,31,4,148,37,228,107,40,143,95,15,239,188,142,160,30,36,232,235,34,123,13,63,36,1,92,189,49,255,191,152,101,81,234,45,170,78,228,224,41,77,6,235,75,41,95,228,160,53,118,183,248,110,40,9,128,63,159,4,146,88,128,57,182,207,231,204,72,18,102,249,225,183,253,26,165,204,221,133,107,160,47,211,121,43,20,152,236,53,207,92,248,102,254,75,207,136,49,232,147,125,59,184,15,14,62,136,58,5,56,132,135,139,160,146,83,193,207,181,29,203,52,47,58,114,231,161,55,66,1,75,127,145,210,118,37,82,232,135,3,183,30,255,240,248,11,160,74,111,171,71,106,196,108,204,154,49,109,206,164,6,195,104,55,35,226,133,78,86,140,154,197,163,105,253,218,72,68,52,160,221,162,147,55,34,170,139,142,218,244,84,132,181,168,39,246,188,198,8,193,144,16,119,237,138,12,69,220,76,152,153,153,160,250,100,42,85,168,208,19,121,181,167,41,37,110,73,50,34,56,59,218,49,242,70,153,106,217,4,105,151,51,36,134,125,128,5],[249,2,17,160,243,248,117,132,135,179,242,217,170,170,147,202,41,30,49,202,235,19,91,182,154,115,189,49,71,95,213,18,134,202,205,168,160,248,214,27,163,48,89,69,124,39,200,95,223,46,31,254,156,7,133,69,242,252,178,116,213,90,11,24,2,233,210,95,159,160,71,79,93,73,42,149,57,51,225,103,235,189,110,188,70,240,193,23,63,219,116,110,243,110,149,7,153,50,68,75,57,255,160,247,151,249,133,151,231,1,67,135,51,218,198,210,129,152,142,23,144,43,153,113,15,227,167,123,200,117,134,246,144,41,89,160,64,41,80,117,176,86,114,226,155,222,42,78,189,238,210,98,213,168,109,98,43,187,53,78,43,64,239,233,108,49,103,145,160,81,130,213,232,226,141,41,38,167,145,141,254,200,67,223,12,25,155,31,46,162,105,182,222,2,233,159,55,73,58,81,8,160,26,77,61,137,205,196,204,129,210,129,10,70,241,189,76,121,69,162,6,215,188,152,126,170,249,149,72,157,147,95,113,240,160,76,38,106,162,112,134,133,37,202,39,149,12,3,108,165,104,174,60,185,97,253,218,30,38,19,121,89,102,165,245,8,216,160,52,133,159,70,24,244,195,146,2,17,195,222,204,211,129,28,126,50,191,31,4,148,37,228,107,40,143,95,15,239,188,142,160,30,36,232,235,34,123,13,63,36,1,92,189,49,255,191,152,101,81,234,45,170,78,228,224,41,77,6,235,75,41,95,228,160,53,118,183,248,110,40,9,128,63,159,4,146,88,128,57,182,207,231,204,72,18,102,249,225,183,253,26,165,204,221,133,107,160,47,211,121,43,20,152,236,53,207,92,248,102,254,75,207,136,49,232,147,125,59,184,15,14,62,136,58,5,56,132,135,139,160,146,83,193,207,181,29,203,52,47,58,114,231,161,55,66,1,75,127,145,210,118,37,82,232,135,3,183,30,255,240,248,11,160,74,111,171,71,106,196,108,204,154,49,109,206,164,6,195,104,55,35,226,133,78,86,140,154,197,163,105,253,218,72,68,52,160,142,150,91,3,151,108,7,234,211,182,207,220,110,244,217,170,240,12,90,203,195,243,210,201,172,9,15,30,76,40,83,183,160,250,100,42,85,168,208,19,121,181,167,41,37,110,73,50,34,56,59,218,49,242,70,153,106,217,4,105,151,51,36,134,125,128,5],[249,2,17,160,161,128,1,12,47,16,128,58,172,109,97,186,101,50,211,24,116,166,152,209,189,185,191,39,125,163,235,50,169,86,158,229,160,43,180,202,90,212,89,144,118,139,227,102,232,30,186,65,236,181,5,130,247,53,26,255,110,32,164,81,96,121,240,13,252,160,222,100,177,157,75,143,240,145,16,36,58,46,51,139,107,7,196,233,64,182,153,253,203,175,129,102,22,111,153,168,150,26,160,94,231,69,12,111,229,77,99,71,17,141,11,41,112,27,177,218,61,40,30,213,193,247,27,173,123,94,162,194,11,64,110,160,148,241,5,249,211,104,221,226,140,197,193,238,210,173,105,8,129,244,154,57,13,253,109,216,177,158,110,36,172,122,110,88,160,49,101,31,195,122,182,161,106,170,190,126,247,114,74,123,53,20,100,9,186,33,38,17,167,168,229,10,220,151,18,196,241,160,73,246,226,153,120,139,128,58,10,194,85,4,186,39,18,220,239,252,50,159,22,196,125,122,103,50,247,196,37,68,58,169,160,15,132,108,63,247,99,185,92,140,54,8,64,230,186,45,30,61,193,8,165,18,74,107,200,87,45,33,232,22,58,219,43,160,171,6,213,180,15,77,228,71,174,54,254,251,111,241,218,40,233,3,107,112,164,163,132,133,85,121,0,128,188,237,176,38,160,190,202,142,180,181,1,250,241,49,215,108,185,216,23,205,142,139,158,85,162,252,156,118,150,43,152,194,183,178,218,159,221,160,116,118,116,254,111,169,77,111,65,32,203,133,193,209,164,92,7,21,222,137,239,153,10,17,202,156,229,253,242,229,50,66,160,245,182,9,212,150,185,219,26,154,17,0,141,168,125,166,152,114,219,87,156,42,77,206,233,29,211,176,18,46,29,86,118,160,125,173,37,34,63,10,10,105,35,138,170,159,170,58,203,218,96,174,159,130,118,216,137,144,59,203,221,237,109,28,197,14,160,125,205,12,44,38,14,115,188,176,89,248,149,162,236,64,246,24,91,125,70,183,125,37,100,214,54,174,74,207,71,185,190,160,9,167,144,133,57,89,194,210,118,41,249,242,60,234,105,179,15,125,163,86,11,161,61,242,89,222,67,163,239,141,115,22,160,229,254,113,96,76,247,87,54,147,166,26,241,48,108,149,89,115,6,35,119,201,191,233,239,90,99,195,93,22,222,43,126,128,5],[249,2,17,160,161,128,1,12,47,16,128,58,172,109,97,186,101,50,211,24,116,166,152,209,189,185,191,39,125,163,235,50,169,86,158,229,160,43,180,202,90,212,89,144,118,139,227,102,232,30,186,65,236,181,5,130,247,53,26,255,110,32,164,81,96,121,240,13,252,160,222,100,177,157,75,143,240,145,16,36,58,46,51,139,107,7,196,233,64,182,153,253,203,175,129,102,22,111,153,168,150,26,160,94,231,69,12,111,229,77,99,71,17,141,11,41,112,27,177,218,61,40,30,213,193,247,27,173,123,94,162,194,11,64,110,160,148,241,5,249,211,104,221,226,140,197,193,238,210,173,105,8,129,244,154,57,13,253,109,216,177,158,110,36,172,122,110,88,160,49,101,31,195,122,182,161,106,170,190,126,247,114,74,123,53,20,100,9,186,33,38,17,167,168,229,10,220,151,18,196,241,160,73,246,226,153,120,139,128,58,10,194,85,4,186,39,18,220,239,252,50,159,22,196,125,122,103,50,247,196,37,68,58,169,160,15,132,108,63,247,99,185,92,140,54,8,64,230,186,45,30,61,193,8,165,18,74,107,200,87,45,33,232,22,58,219,43,160,171,6,213,180,15,77,228,71,174,54,254,251,111,241,218,40,233,3,107,112,164,163,132,133,85,121,0,128,188,237,176,38,160,190,202,142,180,181,1,250,241,49,215,108,185,216,23,205,142,139,158,85,162,252,156,118,150,43,152,194,183,178,218,159,221,160,116,118,116,254,111,169,77,111,65,32,203,133,193,209,164,92,7,21,222,137,239,153,10,17,202,156,229,253,242,229,50,66,160,245,182,9,212,150,185,219,26,154,17,0,141,168,125,166,152,114,219,87,156,42,77,206,233,29,211,176,18,46,29,86,118,160,125,173,37,34,63,10,10,105,35,138,170,159,170,58,203,218,96,174,159,130,118,216,137,144,59,203,221,237,109,28,197,14,160,77,232,92,90,207,144,28,209,255,114,73,30,34,22,65,146,193,168,52,246,172,111,139,107,21,220,140,195,72,109,174,142,160,9,167,144,133,57,89,194,210,118,41,249,242,60,234,105,179,15,125,163,86,11,161,61,242,89,222,67,163,239,141,115,22,160,229,254,113,96,76,247,87,54,147,166,26,241,48,108,149,89,115,6,35,119,201,191,233,239,90,99,195,93,22,222,43,126,128,5],[249,2,17,160,1,140,240,73,75,204,201,222,156,243,213,94,33,163,61,8,206,249,37,116,57,74,97,38,98,157,136,8,8,58,80,150,160,195,199,196,181,116,185,85,110,248,215,152,211,207,168,41,60,203,5,86,141,59,163,78,219,9,213,111,185,55,120,19,233,160,140,202,218,220,242,107,140,113,118,132,7,69,53,214,70,230,137,184,171,129,43,48,107,81,80,73,247,0,177,229,219,121,160,33,36,191,50,11,87,222,33,182,77,167,63,136,123,248,241,74,182,24,11,174,247,239,125,99,202,207,255,128,35,52,165,160,149,249,75,250,81,105,43,241,150,173,198,3,252,180,149,96,0,111,180,34,118,196,43,123,93,132,160,96,250,100,217,45,160,32,39,96,173,133,195,109,50,97,77,73,185,128,89,4,150,255,132,58,164,43,120,193,117,186,32,133,65,91,116,162,173,160,98,239,63,98,146,213,134,176,5,254,159,193,14,251,162,124,237,62,243,94,97,73,108,47,3,76,184,133,162,93,214,124,160,24,110,66,31,239,73,37,228,27,69,165,214,234,132,223,109,118,39,20,166,141,25,228,24,156,85,122,60,112,195,235,154,160,124,4,254,255,41,243,241,33,206,19,170,136,141,252,149,202,221,147,172,85,213,237,197,110,71,174,111,101,127,85,205,59,160,23,49,65,202,234,196,28,65,205,115,198,37,246,143,124,72,166,37,205,232,162,25,22,39,127,188,14,26,18,214,240,152,160,51,21,215,92,255,202,104,15,118,167,53,140,39,4,142,82,127,133,147,230,204,253,47,54,99,23,226,78,113,129,89,185,160,239,123,251,65,188,11,107,22,9,46,42,104,47,193,60,78,205,118,242,12,136,145,137,46,214,157,184,26,255,37,206,38,160,179,163,103,185,250,53,96,32,14,9,248,46,117,61,151,70,245,116,155,44,163,22,115,1,102,242,244,157,45,81,102,14,160,120,51,181,75,204,140,26,229,78,145,104,6,122,193,149,189,178,100,84,118,214,32,148,10,91,248,41,39,153,51,148,250,160,152,64,28,199,229,115,92,129,39,229,199,166,105,168,252,23,227,109,56,225,3,255,171,233,92,155,115,43,225,156,231,35,160,79,205,115,234,146,184,235,250,60,154,252,244,30,28,214,37,12,114,43,159,140,167,245,162,159,65,188,1,113,43,38,143,128,5],[249,2,17,160,1,140,240,73,75,204,201,222,156,243,213,94,33,163,61,8,206,249,37,116,57,74,97,38,98,157,136,8,8,58,80,150,160,195,199,196,181,116,185,85,110,248,215,152,211,207,168,41,60,203,5,86,141,59,163,78,219,9,213,111,185,55,120,19,233,160,140,202,218,220,242,107,140,113,118,132,7,69,53,214,70,230,137,184,171,129,43,48,107,81,80,73,247,0,177,229,219,121,160,224,226,228,152,49,191,94,18,202,42,40,43,44,135,217,234,139,112,115,185,228,196,213,42,168,79,181,143,79,201,143,96,160,149,249,75,250,81,105,43,241,150,173,198,3,252,180,149,96,0,111,180,34,118,196,43,123,93,132,160,96,250,100,217,45,160,32,39,96,173,133,195,109,50,97,77,73,185,128,89,4,150,255,132,58,164,43,120,193,117,186,32,133,65,91,116,162,173,160,98,239,63,98,146,213,134,176,5,254,159,193,14,251,162,124,237,62,243,94,97,73,108,47,3,76,184,133,162,93,214,124,160,24,110,66,31,239,73,37,228,27,69,165,214,234,132,223,109,118,39,20,166,141,25,228,24,156,85,122,60,112,195,235,154,160,124,4,254,255,41,243,241,33,206,19,170,136,141,252,149,202,221,147,172,85,213,237,197,110,71,174,111,101,127,85,205,59,160,23,49,65,202,234,196,28,65,205,115,198,37,246,143,124,72,166,37,205,232,162,25,22,39,127,188,14,26,18,214,240,152,160,51,21,215,92,255,202,104,15,118,167,53,140,39,4,142,82,127,133,147,230,204,253,47,54,99,23,226,78,113,129,89,185,160,239,123,251,65,188,11,107,22,9,46,42,104,47,193,60,78,205,118,242,12,136,145,137,46,214,157,184,26,255,37,206,38,160,179,163,103,185,250,53,96,32,14,9,248,46,117,61,151,70,245,116,155,44,163,22,115,1,102,242,244,157,45,81,102,14,160,120,51,181,75,204,140,26,229,78,145,104,6,122,193,149,189,178,100,84,118,214,32,148,10,91,248,41,39,153,51,148,250,160,152,64,28,199,229,115,92,129,39,229,199,166,105,168,252,23,227,109,56,225,3,255,171,233,92,155,115,43,225,156,231,35,160,79,205,115,234,146,184,235,250,60,154,252,244,30,28,214,37,12,114,43,159,140,167,245,162,159,65,188,1,113,43,38,143,128,5],[249,2,17,160,245,105,73,55,130,156,85,160,31,141,126,218,15,74,121,147,147,14,234,12,31,2,207,74,132,213,9,173,180,149,183,107,160,221,130,180,81,176,155,200,100,168,4,254,92,101,171,36,147,95,202,31,177,191,39,28,78,15,253,236,77,124,115,149,137,160,8,37,3,52,123,198,42,148,211,79,179,98,105,89,161,130,151,2,137,5,198,34,114,85,180,47,176,126,179,111,60,206,160,232,217,161,8,22,169,40,66,131,228,203,23,191,255,11,201,101,138,145,67,49,60,150,125,179,56,59,152,181,26,174,138,160,52,5,132,223,20,125,125,152,77,21,29,239,159,211,65,174,156,121,107,233,188,67,44,242,54,70,100,18,159,243,207,206,160,224,250,15,5,72,187,58,246,78,193,251,188,67,94,94,63,151,23,215,194,99,44,14,23,45,34,254,220,3,94,41,58,160,6,105,60,12,5,193,169,245,176,112,146,23,69,42,0,33,177,13,230,213,165,102,152,203,58,175,135,4,16,128,172,8,160,151,152,86,204,166,248,67,223,250,77,31,100,237,11,43,191,90,23,20,54,199,92,11,215,145,50,87,90,167,159,57,165,160,1,232,26,222,47,23,75,176,90,187,251,204,93,173,132,158,36,225,142,226,147,28,202,173,168,228,182,229,123,127,49,117,160,247,78,159,238,239,170,53,113,45,18,48,98,112,234,117,104,97,108,138,230,14,76,168,84,236,172,64,67,208,57,6,73,160,46,68,248,146,220,227,97,0,41,252,210,9,44,117,251,227,165,196,13,189,174,150,34,139,203,17,200,40,245,122,167,206,160,26,216,79,208,48,103,203,251,178,213,93,58,82,104,200,119,234,228,233,252,208,91,195,35,224,229,183,69,89,175,14,229,160,246,209,205,0,213,119,26,186,142,93,94,61,153,28,165,149,49,176,155,119,213,241,208,245,15,163,38,131,125,219,108,170,160,89,143,190,130,47,255,40,170,85,219,138,46,139,251,126,68,17,241,5,216,204,86,127,71,120,116,170,149,237,137,28,227,160,79,246,250,96,218,39,3,222,92,140,84,169,44,51,184,140,136,139,201,154,119,208,207,98,29,112,62,108,254,3,142,180,160,130,179,74,86,218,213,192,18,132,24,134,63,237,50,86,187,20,97,174,221,173,83,84,97,186,105,52,78,209,101,251,138,128,5],[249,2,17,160,245,105,73,55,130,156,85,160,31,141,126,218,15,74,121,147,147,14,234,12,31,2,207,74,132,213,9,173,180,149,183,107,160,221,130,180,81,176,155,200,100,168,4,254,92,101,171,36,147,95,202,31,177,191,39,28,78,15,253,236,77,124,115,149,137,160,8,37,3,52,123,198,42,148,211,79,179,98,105,89,161,130,151,2,137,5,198,34,114,85,180,47,176,126,179,111,60,206,160,232,217,161,8,22,169,40,66,131,228,203,23,191,255,11,201,101,138,145,67,49,60,150,125,179,56,59,152,181,26,174,138,160,46,144,237,143,190,109,159,205,233,64,197,103,36,172,203,35,39,122,25,184,212,193,136,197,175,120,207,44,140,182,105,222,160,224,250,15,5,72,187,58,246,78,193,251,188,67,94,94,63,151,23,215,194,99,44,14,23,45,34,254,220,3,94,41,58,160,6,105,60,12,5,193,169,245,176,112,146,23,69,42,0,33,177,13,230,213,165,102,152,203,58,175,135,4,16,128,172,8,160,151,152,86,204,166,248,67,223,250,77,31,100,237,11,43,191,90,23,20,54,199,92,11,215,145,50,87,90,167,159,57,165,160,1,232,26,222,47,23,75,176,90,187,251,204,93,173,132,158,36,225,142,226,147,28,202,173,168,228,182,229,123,127,49,117,160,247,78,159,238,239,170,53,113,45,18,48,98,112,234,117,104,97,108,138,230,14,76,168,84,236,172,64,67,208,57,6,73,160,46,68,248,146,220,227,97,0,41,252,210,9,44,117,251,227,165,196,13,189,174,150,34,139,203,17,200,40,245,122,167,206,160,26,216,79,208,48,103,203,251,178,213,93,58,82,104,200,119,234,228,233,252,208,91,195,35,224,229,183,69,89,175,14,229,160,246,209,205,0,213,119,26,186,142,93,94,61,153,28,165,149,49,176,155,119,213,241,208,245,15,163,38,131,125,219,108,170,160,89,143,190,130,47,255,40,170,85,219,138,46,139,251,126,68,17,241,5,216,204,86,127,71,120,116,170,149,237,137,28,227,160,79,246,250,96,218,39,3,222,92,140,84,169,44,51,184,140,136,139,201,154,119,208,207,98,29,112,62,108,254,3,142,180,160,130,179,74,86,218,213,192,18,132,24,134,63,237,50,86,187,20,97,174,221,173,83,84,97,186,105,52,78,209,101,251,138,128,5],[248,241,160,255,151,217,75,103,5,122,115,224,137,233,146,50,189,95,178,178,247,44,237,22,101,231,39,198,40,14,249,60,251,151,15,128,128,128,128,160,60,79,85,51,115,192,158,157,93,223,211,100,62,94,72,146,251,82,116,111,190,139,246,12,252,146,211,122,66,110,206,20,128,160,120,190,160,200,253,109,255,226,49,189,87,112,136,160,23,77,119,59,173,185,188,145,251,156,155,144,100,217,100,114,109,106,128,160,69,72,113,186,79,146,63,86,46,218,1,200,131,76,71,142,217,35,30,209,101,239,91,47,163,221,136,130,249,155,236,112,160,49,65,26,94,193,156,227,78,42,198,56,211,105,254,0,33,31,96,41,208,40,13,215,156,51,173,132,112,34,192,121,49,160,244,154,252,18,232,96,245,36,84,15,253,182,157,226,247,165,106,144,166,1,2,140,228,170,110,87,112,80,140,149,162,43,128,160,20,103,6,95,163,140,21,238,207,84,226,60,134,0,183,217,11,213,185,123,139,201,37,22,227,234,220,30,160,20,244,115,128,128,128,5],[248,241,160,188,253,144,87,144,251,204,78,148,203,12,141,0,77,176,70,67,92,90,100,110,40,255,28,218,97,116,184,26,121,18,49,128,128,128,128,160,60,79,85,51,115,192,158,157,93,223,211,100,62,94,72,146,251,82,116,111,190,139,246,12,252,146,211,122,66,110,206,20,128,160,120,190,160,200,253,109,255,226,49,189,87,112,136,160,23,77,119,59,173,185,188,145,251,156,155,144,100,217,100,114,109,106,128,160,69,72,113,186,79,146,63,86,46,218,1,200,131,76,71,142,217,35,30,209,101,239,91,47,163,221,136,130,249,155,236,112,160,49,65,26,94,193,156,227,78,42,198,56,211,105,254,0,33,31,96,41,208,40,13,215,156,51,173,132,112,34,192,121,49,160,244,154,252,18,232,96,245,36,84,15,253,182,157,226,247,165,106,144,166,1,2,140,228,170,110,87,112,80,140,149,162,43,128,160,20,103,6,95,163,140,21,238,207,84,226,60,134,0,183,217,11,213,185,123,139,201,37,22,227,234,220,30,160,20,244,115,128,128,128,5],[248,81,128,128,128,128,128,128,128,160,222,45,71,217,199,68,20,55,244,206,68,197,49,191,78,208,106,209,111,87,254,9,221,230,148,86,131,219,7,121,62,140,160,190,214,56,80,83,126,135,17,104,48,181,30,249,223,80,59,155,70,206,67,24,6,82,98,81,246,212,143,253,181,15,180,128,128,128,128,128,128,128,128,5],[248,102,157,55,236,125,29,155,142,209,241,75,145,144,143,254,65,81,209,56,13,192,157,236,195,213,73,132,11,251,149,241,184,70,248,68,1,128,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,5],[248,102,157,32,133,130,180,167,143,97,28,115,102,25,94,62,148,249,8,6,55,244,16,75,187,208,208,127,251,120,61,73,184,70,248,68,1,23,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,5],[248,102,157,32,236,125,29,155,142,209,241,75,145,144,143,254,65,81,209,56,13,192,157,236,195,213,73,132,11,251,149,241,184,70,248,68,1,128,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,5],[248,102,157,32,236,125,29,155,142,209,241,75,145,144,143,254,65,81,209,56,13,192,157,236,195,213,73,132,11,251,149,241,184,70,248,68,1,128,160,112,158,181,221,162,20,124,79,184,25,162,13,167,162,146,25,237,242,59,120,184,154,118,137,92,181,187,152,115,82,223,48,160,7,190,1,231,231,32,111,227,30,206,233,26,215,93,173,166,90,214,186,67,58,230,71,161,185,51,4,105,247,198,103,124,5]]";
        let w: Vec<Vec<u8>> = serde_json::from_str(wit).unwrap();

        let count = w.iter().filter(|r| r[r.len() - 1] != 5).count() * 2;
        let randomness: Fr = 123456789.scalar();
        let instance: Vec<Vec<Fr>> = (1..HASH_WIDTH + 1)
            .map(|exp| vec![randomness.pow(&[exp as u64, 0, 0, 0]); count])
            .collect();

        let circuit = MPTCircuit::<Fr> {
            witness: w.clone(),
            randomness,
        };

        // let prover = MockProver::run(9, &circuit, vec![pub_root]).unwrap();
        let num_rows = w.len() * 2;
        let prover = MockProver::run(14 /* 9 */, &circuit, instance).unwrap();
        assert_eq!(prover.verify_at_rows(0..num_rows, 0..num_rows,), Ok(()));
        //assert_eq!(prover.verify_par(), Ok(()));
        //prover.assert_satisfied();       
       
                
    }
}
