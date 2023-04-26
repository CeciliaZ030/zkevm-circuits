use eth_types::Field;
use gadgets::util::Scalar;
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Expression, VirtualCells},
};

use super::{
    helpers::{MPTConstraintBuilder, RLPItemView},
    param::ARITY,
    rlp_gadgets::{RLPItemWitness, RLPListDataGadget},
    witness_row::Node,
    MPTContext,
};
use crate::{
    circuit,
    circuit_tools::{cell_manager::Cell, constraint_builder::RLCChainable, gadgets::LtGadget},
    mpt_circuit::{
        helpers::{nibble_rlc, Indexable},
        param::{HASH_WIDTH, RLP_NIL},
        MPTConfig, MPTState,
    },
};

#[derive(Clone, Debug)]
pub(crate) struct BranchState<F> {
    pub(crate) key_rlc_post_branch: Expression<F>,
    pub(crate) key_mult_post_branch: Expression<F>,
    pub(crate) key_rlc_post_drifted: Expression<F>,
    pub(crate) key_mult_post_drifted: Expression<F>,
    pub(crate) num_nibbles: Expression<F>,
    pub(crate) is_key_odd: Expression<F>,
    pub(crate) mod_rlc: [Expression<F>; 2],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BranchGadget<F> {
    rlp_list: [RLPListDataGadget<F>; 2],
    is_modified: [Cell<F>; ARITY],
    is_drifted: [Cell<F>; ARITY],
    mod_rlc: [Cell<F>; 2],
    is_not_hashed: [LtGadget<F, 2>; 2],

    // Post branch state
    post_state: Option<BranchState<F>>,
}

impl<F: Field> BranchGadget<F> {
    pub fn configure(
        meta: &mut VirtualCells<'_, F>,
        cb: &mut MPTConstraintBuilder<F>,
        ctx: MPTContext<F>,
        is_placeholder: &[Cell<F>; 2],
        parent_rlc: &[Expression<F>; 2],
        is_root: &[Expression<F>; 2],
        key_rlc: Expression<F>,
        key_mult: Expression<F>,
        num_nibbles: Expression<F>,
        is_key_odd: Expression<F>,
    ) -> Self {
        let r = ctx.r.clone();

        let mut config = BranchGadget::default();

        circuit!([meta, cb.base], {
            // Data
            let children: [RLPItemView<F>; ARITY + 1] =
                array_init::array_init(|i| ctx.rlp_item(meta, &mut cb.base, i));

            let mut num_bytes_left = vec![0.expr(); 2];
            let mut node_rlc = vec![0.expr(); 2];
            let mut mult = vec![1.expr(); 2];
            for is_s in [true, false] {
                // Read the list
                config.rlp_list[is_s.idx()] = RLPListDataGadget::construct(&mut cb.base);
                // Start RLC encoding the RLP data starting with the list RLP bytes
                (node_rlc[is_s.idx()], mult[is_s.idx()]) =
                    config.rlp_list[is_s.idx()].rlp_list.rlc_rlp_only(&r);

                // Keep track of how many bytes the branch contains to make sure it's correct.
                num_bytes_left[is_s.idx()] = config.rlp_list[is_s.idx()].rlp_list.len();

                config.mod_rlc[is_s.idx()] = cb.base.query_cell();

                // Check if the branch is hashed or not
                config.is_not_hashed[is_s.idx()] = LtGadget::construct(
                    &mut cb.base,
                    config.rlp_list[is_s.idx()].rlp_list.num_bytes(),
                    HASH_WIDTH.expr(),
                );
            }

            let mut modified_index = 0.expr();
            let mut drifted_index = 0.expr();
            for node_index in 0..ARITY {
                config.is_modified[node_index] = cb.base.query_bool();
                config.is_drifted[node_index] = cb.base.query_bool();

                // Calculate the modified and drifted index from `is_modified`/`is_drifted`
                modified_index = modified_index.expr()
                    + config.is_modified[node_index].expr() * node_index.expr();
                drifted_index =
                    drifted_index.expr() + config.is_drifted[node_index].expr() * node_index.expr();
            }

            // Process the branch children
            for node_index in 0..ARITY {
                for is_s in [true, false] {
                    // Get the correct child.
                    // All s children are stored directly in the circuit, but the only modified
                    // child branch for c is stored in child 0.
                    let child = &children[node_index + 1];
                    let mod_child = &children[0];
                    let (rlc_rlp, num_bytes, length, mult_diff) = if is_s {
                        (
                            child.rlc_rlp(),
                            child.num_bytes(),
                            child.len(),
                            child.mult(),
                        )
                    } else {
                        ifx! {config.is_modified[node_index] => {
                            (mod_child.rlc_rlp(), mod_child.num_bytes(), mod_child.len(), mod_child.mult())
                        } elsex {
                            (child.rlc_rlp(), child.num_bytes(), child.len(), child.mult())
                        }}
                    };

                    // Keep track of how many bytes of the list we've processed
                    num_bytes_left[is_s.idx()] =
                        num_bytes_left[is_s.idx()].expr() - num_bytes.expr();

                    // Update the full branch node RLC with the data of this branch
                    node_rlc[is_s.idx()] = (node_rlc[is_s.idx()].expr(), mult[is_s.idx()].expr())
                        .rlc_chain(rlc_rlp.expr());

                    // Update the branch node multiplier
                    mult[is_s.idx()] = mult[is_s.idx()].expr() * mult_diff;

                    // When in a placeholder branch, both branches are the same - the placeholder
                    // branch and its parallel counterpart, which is not a
                    // placeholder, but a regular branch (newly added branch).
                    // The regular branch has only two non-nil nodes,
                    // because the placeholder branch appears only when an existing
                    // leaf drifts down into a newly added branch. Besides an
                    // existing leaf, we have a leaf that was being added and that caused
                    // a new branch to be added. So we need to check that there are exactly two
                    // non-nil nodes (otherwise the attacker could add more
                    // new leaves at the same time). The non-nil nodes need to be at
                    // `is_modified` and `is_drifted`, elsewhere there have
                    // to be zeros.
                    ifx! {is_placeholder[is_s.idx()] => {
                        ifx! {or::expr(&[config.is_modified[node_index].expr(), config.is_drifted[node_index].expr()]) => {
                            require!(length => HASH_WIDTH);
                        } elsex {
                            require!(length => 0);
                        }}
                        // Make sure that `modified_index != drifted_index`
                        require!(config.is_modified[node_index].expr() + config.is_drifted[node_index].expr() => bool);
                    }}
                }
            }
            for is_s in [true, false] {
                // Number of bytes left needs to be 1 because ValueNode which occupies 1 byte
                require!(num_bytes_left[is_s.idx()] => 1);
                // TODO: acc currently doesn'thave branch ValueNode info
                node_rlc[is_s.idx()] = (node_rlc[is_s.idx()].expr(), mult[is_s.idx()].expr())
                    .rlc_chain(RLP_NIL.expr());
            }

            // `is_modified` needs to be set to 1 at exactly 1 branch child
            let is_modified_values = (0..ARITY)
                .map(|rot| config.is_modified[rot].expr())
                .collect::<Vec<_>>();
            require!(sum::expr(&is_modified_values) => 1);
            // When there's a placeholder, `is_drifted` needs to be set to 1 at exactly 1
            // branch child
            ifx! {or::expr(&[is_placeholder[true.idx()].expr(), is_placeholder[false.idx()].expr()]) => {
                let is_drifted_values = (0..ARITY).map(|rot| config.is_drifted[rot].expr()).collect::<Vec<_>>();
                require!(sum::expr(&is_drifted_values) => 1);
            }}

            // Check if the branch is in its parent
            for is_s in [true, false] {
                let (rlc, num_bytes, is_not_hashed) = {
                    (
                        node_rlc[is_s.idx()].expr(),
                        config.rlp_list[is_s.idx()].rlp_list.num_bytes(),
                        config.is_not_hashed[is_s.idx()].expr(),
                    )
                };

                ifx! {not!(is_placeholder[is_s.idx()]) => {
                    ifx!{or::expr(&[is_root[is_s.idx()].expr(), not!(is_not_hashed)]) => {
                        // Hashed branch hash in parent branch
                        require!((1, rlc, num_bytes, parent_rlc[is_s.idx()].expr()) => @"keccak");
                    } elsex {
                        // Non-hashed branch hash in parent branch
                        require!(rlc => parent_rlc[is_s.idx()].expr());
                    }}
                }}
            }

            // Update the key RLC and multiplier for the branch nibble.
            let (key_rlc_post_branch, key_mult_post_branch) = nibble_rlc(
                &mut cb.base,
                key_rlc.expr(),
                key_mult.expr(),
                is_key_odd.expr(),
                modified_index.expr(),
                &r,
            );
            // Also calculate the key RLC and multiplier for the drifted nibble.
            let (key_rlc_post_drifted, key_mult_post_drifted) = nibble_rlc(
                &mut cb.base,
                key_rlc.expr(),
                key_mult.expr(),
                is_key_odd.expr(),
                drifted_index.expr(),
                &r,
            );

            // Update the nibble counter
            let num_nibbles = num_nibbles + 1.expr();
            // Update key parity
            let is_key_odd = not!(is_key_odd);

            // Set the branch we'll take
            for is_s in [true, false] {
                ifx! {is_placeholder[is_s.idx()] => {
                    for node_index in 0..ARITY {
                        ifx!{config.is_drifted[node_index].expr() => {
                            require!(config.mod_rlc[is_s.idx()] =>
                                children[node_index + 1].rlc_content());
                        }}
                    }
                } elsex {
                    if is_s {
                        for node_index in 0..ARITY {
                            ifx!{config.is_modified[node_index].expr() => {
                                require!(config.mod_rlc[is_s.idx()] =>
                                    children[node_index + 1].rlc_content());
                            }}
                        }
                    } else {
                        require!(config.mod_rlc[is_s.idx()] => children[0].rlc_content());
                    }
                }}
            }

            // Store the post ext state
            config.post_state = Some(BranchState {
                key_rlc_post_branch,
                key_mult_post_branch,
                key_rlc_post_drifted,
                key_mult_post_drifted,
                num_nibbles,
                is_key_odd,
                mod_rlc: [
                    config.mod_rlc[true.idx()].expr(),
                    config.mod_rlc[false.idx()].expr(),
                ],
            });
        });

        config
    }

    pub(crate) fn get_post_state(&self) -> BranchState<F> {
        self.post_state.as_ref().unwrap().clone()
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        mpt_config: &MPTConfig<F>,
        pv: &mut MPTState<F>,
        offset: usize,
        is_placeholder: &[bool; 2],
        key_rlc: &mut F,
        key_mult: &mut F,
        num_nibbles: &mut usize,
        is_key_odd: &mut bool,
        node: &Node,
        rlp_values: &[RLPItemWitness],
    ) -> Result<(F, F, F, [F; 2]), Error> {
        let branch = &node.extension_branch.clone().unwrap().branch;

        for is_s in [true, false] {
            let rlp_list_witness = self.rlp_list[is_s.idx()].assign(
                region,
                offset,
                &branch.list_rlp_bytes[is_s.idx()],
            )?;
            self.is_not_hashed[is_s.idx()].assign(
                region,
                offset,
                rlp_list_witness.num_bytes().scalar(),
                HASH_WIDTH.scalar(),
            )?;
        }

        for node_index in 0..ARITY {
            self.is_modified[node_index].assign(
                region,
                offset,
                (node_index == branch.modified_index).scalar(),
            )?;
            self.is_drifted[node_index].assign(
                region,
                offset,
                (node_index == branch.drifted_index).scalar(),
            )?;
        }

        // one nibble is used for position in branch
        *num_nibbles += 1;
        // Update key parity
        *is_key_odd = !*is_key_odd;

        // Update the key RLC and multiplier for the branch nibble.
        let (nibble_mult, mult): (F, F) = if *is_key_odd {
            // The nibble will be added as the most significant nibble using the same
            // multiplier
            (16.scalar(), 1.scalar())
        } else {
            // The nibble will be added as the least significant nibble, the multiplier
            // needs to advance
            (1.scalar(), pv.r)
        };
        let key_rlc_post_branch =
            *key_rlc + F::from(branch.modified_index as u64) * nibble_mult * *key_mult;
        let key_rlc_post_drifted =
            *key_rlc + F::from(branch.drifted_index as u64) * nibble_mult * *key_mult;
        let key_mult_post_branch = *key_mult * mult;

        // Set the branch we'll take
        let mut mod_node_hash_rlc = [0.scalar(); 2];
        for is_s in [true, false] {
            mod_node_hash_rlc[is_s.idx()] = if is_placeholder[is_s.idx()] {
                rlp_values[1 + branch.drifted_index].rlc_content(pv.r)
            } else {
                if is_s {
                    rlp_values[1 + branch.modified_index].rlc_content(pv.r)
                } else {
                    rlp_values[0].rlc_content(pv.r)
                }
            };
            self.mod_rlc[is_s.idx()].assign(region, offset, mod_node_hash_rlc[is_s.idx()])?;
        }

        Ok((
            key_rlc_post_branch,
            key_rlc_post_drifted,
            key_mult_post_branch,
            mod_node_hash_rlc,
        ))
    }
}
