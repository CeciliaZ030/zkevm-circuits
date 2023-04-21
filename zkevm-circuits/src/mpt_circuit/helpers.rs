use std::any::Any;

use crate::{
    _cb, circuit,
    circuit_tools::{
        cell_manager::{Cell, CellManager_, Trackable, CellType},
        constraint_builder::{
            Conditionable, ConstraintBuilder, RLCChainable, RLCChainableValue, RLCable,
            RLCableValue,
        },
        gadgets::IsEqualGadget,
        memory::MemoryBank,
    },
    matchr, matchw,
    mpt_circuit::{
        param::{EMPTY_TRIE_HASH, KEY_LEN_IN_NIBBLES, KEY_PREFIX_EVEN, KEY_TERMINAL_PREFIX_EVEN},
        rlp_gadgets::{get_ext_odd_nibble, get_terminal_odd_nibble},
    },
    util::Expr,
    evm_circuit::util::{CachedRegion, rlc}, bytecode_circuit::param::HASH_WIDTH,
};
use eth_types::Field;
use gadgets::util::{or, Scalar};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{Error, Expression},
};

use super::{
    rlp_gadgets::{
        get_ext_odd_nibble_value, RLPListGadget, RLPListWitness, RLPValueGadget, RLPValueWitness,
    },
    FixedTableTag,

    table::Lookup,
};

// Max degree allowed in all expressions passing through the ConstraintBuilder.
// It aims to cap `extended_k` to 2, which allows constraint degree to 2^2+1,
// but each ExecutionGadget has implicit selector degree 3, so here it only
// allows 2^2+1-3 = 2.
const MAX_DEGREE: usize = 5;
const IMPLICIT_DEGREE: usize = 3;

/// Indexable object
pub trait Indexable {
    /// Convert to index
    fn idx(&self) -> usize;
}

impl Indexable for bool {
    fn idx(&self) -> usize {
        if *self {
            0
        } else {
            1
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LeafKeyGadget<F> {
    has_no_nibbles: IsEqualGadget<F>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LeafKeyWitness {
    has_no_nibble: bool,
}

impl<F: Field> LeafKeyGadget<F> {
    pub(crate) fn construct(cb: &mut ConstraintBuilder<F>, rlp_key: RLPValueGadget<F>) -> Self {
        circuit!([meta, cb], {
            let has_no_nibbles = IsEqualGadget::<F>::construct(
                cb,
                rlp_key.bytes[0].expr(),
                KEY_TERMINAL_PREFIX_EVEN.expr(),
            );
            LeafKeyGadget { has_no_nibbles }
        })
    }

    pub(crate) fn expr(
        &self,
        cb: &mut ConstraintBuilder<F>,
        rlp_key: RLPValueGadget<F>,
        key_mult_prev: Expression<F>,
        is_key_odd: Expression<F>,
        r: &[Expression<F>],
    ) -> Expression<F> {
        circuit!([meta, cb], {
            let calc_rlc = |cb: &mut ConstraintBuilder<F>,
                            bytes: &[Expression<F>],
                            is_key_odd: Expression<F>| {
                leaf_key_rlc(cb, bytes, key_mult_prev.expr(), is_key_odd.expr(), r)
            };
            matchx! {
                rlp_key.is_short() => {
                    // When no nibbles: only terminal prefix at `bytes[1]`.
                    // Else: Terminal prefix + single nibble  at `bytes[1]`
                    let is_odd = not!(self.has_no_nibbles);
                    calc_rlc(cb, &rlp_key.bytes[0..1], is_odd)
                },
                rlp_key.is_long() => {
                    // First key byte is at `bytes[2]`.
                    calc_rlc(cb, &rlp_key.bytes[1..34], is_key_odd.expr())
                },
            }
        })
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        rlp_key: RLPValueWitness,
    ) -> Result<LeafKeyWitness, Error> {
        let has_no_nibble = self.has_no_nibbles.assign(
            region,
            offset,
            F::from(rlp_key.bytes[0] as u64),
            F::from(KEY_TERMINAL_PREFIX_EVEN as u64),
        )?;
        Ok(LeafKeyWitness {
            has_no_nibble: has_no_nibble != 0.scalar(),
        })
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        rlp_key: RLPValueWitness,
    ) -> Result<LeafKeyWitness, Error> {
        let has_no_nibble = self.has_no_nibbles.assign_cached(
            region,
            offset,
            F::from(rlp_key.bytes[0] as u64),
            F::from(KEY_TERMINAL_PREFIX_EVEN as u64),
        )?;
        Ok(LeafKeyWitness {
            has_no_nibble: has_no_nibble != 0.scalar(),
        })
    }

}

impl LeafKeyWitness {
    pub(crate) fn key<F: Field>(
        &self,
        rlp_key: RLPValueWitness,
        key_rlc: F,
        key_mult: F,
        r: F,
    ) -> (F, F) {
        if rlp_key.len() <= 1 {
            return (key_rlc, key_mult);
        }

        let start = 0;
        let len = rlp_key.len();
        let even_num_of_nibbles = rlp_key.bytes[start + 1] == 32;

        let mut key_rlc = key_rlc;
        let mut key_mult = key_mult;
        if !even_num_of_nibbles {
            // If odd number of nibbles, we have nibble+48 in s_advices[0].
            key_rlc += F::from((rlp_key.bytes[start + 1] - 48) as u64) * key_mult;
            key_mult *= r;
        }
        (key_rlc, key_mult).rlc_chain_value(&rlp_key.bytes[start + 2..start + 2 + len - 1], r)
    }
}

pub(crate) fn ext_key_rlc_expr<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    key_value: RLPValueGadget<F>,
    key_mult_prev: Expression<F>,
    is_key_part_odd: Expression<F>,
    is_key_odd: Expression<F>,
    data: [Vec<Expression<F>>; 2],
    r: &[Expression<F>],
) -> Expression<F> {
    circuit!([meta, cb], {
        let (is_short, is_long) = (key_value.is_short(), key_value.is_long());
        let mult_first_odd = ifx! {is_key_odd => { 1.expr() } elsex { 16.expr() }};
        let calc_rlc = |cb: &mut ConstraintBuilder<F>,
                        bytes: &[Expression<F>],
                        key_mult_first_even: Expression<F>| {
            ext_key_rlc(
                cb,
                bytes,
                key_mult_prev.expr(),
                is_key_part_odd.expr(),
                mult_first_odd.expr(),
                key_mult_first_even,
                r,
            )
        };
        matchx! {
            and::expr(&[is_long.expr(), not!(is_key_odd)]) => {
                // Here we need to multiply nibbles over bytes with different r's so we need to rlc over separate nibbles.
                // Note that there can be at max 31 key bytes because 32 same bytes would mean
                // the two keys being the same - update operation, not splitting into extension node.
                // So, we do not need to look further than `s_main.bytes` even if `s_main.bytes[0]`
                // is not used (when even number of nibbles).
                let mut key_bytes = vec![data[0][1].expr()];
                key_bytes.append(&mut data[0][1..].iter().skip(1).zip(data[1][2..].iter()).map(|(byte, nibble_hi)| {
                    let nibble_lo = (byte.expr() - nibble_hi.expr()) * invert!(16);
                    // Check that `nibble_hi` is correct.
                    require!(byte => nibble_lo.expr() * 16.expr() + nibble_hi.expr());
                    // Collect bytes
                    (nibble_hi.expr() * 16.expr() * r[0].expr()) + nibble_lo.expr()
                }).collect::<Vec<_>>());
                calc_rlc(cb, &key_bytes, 1.expr())
            },
            and::expr(&[is_long.expr(), is_key_odd.expr()]) => {
                let additional_mult = ifx! {is_key_part_odd => { r[0].expr() } elsex { 1.expr() }};
                calc_rlc(cb, &data[0][1..], additional_mult)
            },
            is_short => {
                calc_rlc(cb, &data[0][..1], 1.expr())
            },
        }
    })
}

pub(crate) fn ext_key_rlc_calc_value<F: Field>(
    key_value: RLPValueWitness,
    key_mult_prev: F,
    is_key_part_odd: bool,
    is_key_odd: bool,
    data: [Vec<u8>; 2],
    r: F,
) -> (F, F) {
    let (is_short, is_long) = (key_value.is_short(), key_value.is_long());
    let mult_first_odd = if is_key_odd { 1.scalar() } else { 16.scalar() };
    let calc_rlc = |bytes: &[F], key_mult_first_even: F| {
        ext_key_rlc_value(
            bytes,
            key_mult_prev,
            is_key_part_odd,
            mult_first_odd,
            key_mult_first_even,
            r,
        )
    };
    matchr! {
        is_long && !is_key_odd => {
            // Here we need to multiply nibbles over bytes with different r's so we need to rlc over separate nibbles.
            // Note that there can be at max 31 key bytes because 32 same bytes would mean
            // the two keys being the same - update operation, not splitting into extension node.
            // So, we do not need to look further than `s_main.bytes` even if `s_main.bytes[0]`
            // is not used (when even number of nibbles).
            let mut key_bytes = vec![data[0][1].scalar()];
            key_bytes.append(&mut data[0][1..].iter().skip(1).zip(data[1][2..].iter()).map(|(byte, nibble_hi)| {
                let nibble_lo = (byte - nibble_hi) >> 4;
                // Check that `nibble_hi` is correct.
                assert!(*byte == nibble_lo * 16 + nibble_hi);
                // Collect bytes
                (F::from(*nibble_hi as u64) * F::from(16 as u64) * r) + F::from(nibble_lo as u64)
            }).collect::<Vec<_>>());
            calc_rlc(&key_bytes, 1.scalar())
        },
        is_long && is_key_odd => {
            let additional_mult = if is_key_part_odd { r } else { 1.scalar() };
            calc_rlc(&data[0][1..].iter().map(|byte| byte.scalar()).collect::<Vec<_>>(), additional_mult)
        },
        is_short => {
            calc_rlc(&data[0][..1].iter().map(|byte| byte.scalar()).collect::<Vec<_>>(), 1.scalar())
        },
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ListKeyGadget<F> {
    pub(crate) rlp_list_bytes: [Cell<F>; 3],
    pub(crate) rlp_list: RLPListGadget<F>,
    pub(crate) key_value: RLPValueGadget<F>,
    pub(crate) key: LeafKeyGadget<F>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ListKeyWitness {
    pub(crate) rlp_list: RLPListWitness,
    pub(crate) key_value: RLPValueWitness,
    pub(crate) key: LeafKeyWitness,
}

impl<F: Field> ListKeyGadget<F> {
    pub(crate) fn construct(cb: &mut ConstraintBuilder<F>, bytes: &[Expression<F>]) -> Self {
        let rlp_list_bytes = cb.query_bytes();
        let rlp_list_bytes_expr = rlp_list_bytes.iter().map(|c| c.expr()).collect::<Vec<_>>();
        let key_value = RLPValueGadget::construct(cb, &bytes);
        let key = LeafKeyGadget::construct(cb, key_value.clone());
        ListKeyGadget {
            rlp_list: RLPListGadget::construct(cb, &rlp_list_bytes_expr),
            key_value,
            rlp_list_bytes,
            key,
        }
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        list_bytes: &[u8],
        bytes: &[u8],
    ) -> Result<ListKeyWitness, Error> {
        for (cell, byte) in self.rlp_list_bytes.iter().zip(list_bytes.iter()) {
            cell.assign(region, offset, byte.scalar())?;
        }
        let rlp_list = self.rlp_list.assign(region, offset, list_bytes)?;
        let key_value = self.key_value.assign(region, offset, bytes)?;
        let key = self.key.assign(region, offset, key_value.clone())?;

        Ok(ListKeyWitness {
            rlp_list,
            key_value,
            key,
        })
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        list_bytes: &[u8],
        bytes: &[u8],
    ) -> Result<ListKeyWitness, Error> {
        for (cell, byte) in self.rlp_list_bytes.iter().zip(list_bytes.iter()) {
            cell.assign_cached(region, offset, byte.scalar())?;
        }
        let rlp_list = self.rlp_list.assign_cached(region, offset, list_bytes)?;
        let key_value = self.key_value.assign_cached(region, offset, bytes)?;
        let key = self.key.assign_cached(region, offset, key_value.clone())?;

        Ok(ListKeyWitness {
            rlp_list,
            key_value,
            key,
        })
    }

    pub(crate) fn rlc(&self, r: &[Expression<F>]) -> Expression<F> {
        self.rlp_list
            .rlc_rlp_only(&r)
            .rlc_chain(self.key_value.rlc(&r).1)
    }

    /// Number of bytes of RLP (including list RLP bytes) and key
    pub(crate) fn num_bytes_on_key_row(&self) -> Expression<F> {
        self.rlp_list.num_rlp_bytes() + self.key_value.num_bytes()
    }
}

impl ListKeyWitness {
    /// Number of bytes of RLP (including list RLP bytes) and key
    pub(crate) fn num_bytes_on_key_row(&self) -> usize {
        self.rlp_list.num_rlp_bytes() + self.key_value.num_bytes()
    }

    /// Number of bytes of RLP (including list RLP bytes) and key
    pub(crate) fn rlc_leaf<F: Field>(&self, r: F) -> (F, F) {
        self.rlp_list
            .rlc_rlp_only(r)
            .rlc_chain_value(&self.key_value.bytes[..self.key_value.num_bytes()], r)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct KeyData<F> {
    pub(crate) rlc: Cell<F>,
    pub(crate) mult: Cell<F>,
    pub(crate) num_nibbles: Cell<F>,
    pub(crate) is_odd: Cell<F>,
    pub(crate) drifted_is_odd: Cell<F>,
    pub(crate) drifted_rlc: Cell<F>,
    pub(crate) drifted_mult: Cell<F>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct KeyDataWitness<F> {
    pub(crate) rlc: F,
    pub(crate) mult: F,
    pub(crate) num_nibbles: usize,
    pub(crate) is_odd: bool,
    pub(crate) drifted_is_odd: bool,
    pub(crate) drifted_rlc: F,
    pub(crate) drifted_mult: F,
}

impl<F: Field> Trackable for KeyData<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Trackable> {
        Box::new(self.clone())
    }
}

impl<F: Field> KeyData<F> {
    pub(crate) fn load(
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        offset: Expression<F>,
    ) -> Self {
        let key_data = KeyData {
            rlc: cb.query_cell(),
            mult: cb.query_cell(),
            num_nibbles: cb.query_cell(),
            is_odd: cb.query_cell(),
            drifted_is_odd: cb.query_cell(),
            drifted_rlc: cb.query_cell(),
            drifted_mult: cb.query_cell(),
        };
        circuit!([meta, cb], {
            memory.load(
                "key load",
                cb,
                offset,
                &[
                    key_data.rlc.expr(),
                    key_data.mult.expr(),
                    key_data.num_nibbles.expr(),
                    key_data.is_odd.expr(),
                    key_data.drifted_is_odd.expr(),
                    key_data.drifted_rlc.expr(),
                    key_data.drifted_mult.expr(),
                ],
            );
        });
        key_data
    }

    pub(crate) fn store(
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        values: [Expression<F>; 7],
    ) {
        memory.store(cb, &values);
    }

    pub(crate) fn default_values() -> [F; 7] {
        [
            0.scalar(),
            1.scalar(),
            0.scalar(),
            false.scalar(),
            false.scalar(),
            0.scalar(),
            1.scalar(),
        ]
    }

    // TODO(Brecht): fix
    pub(crate) fn default_values_expr() -> [Expression<F>; 7] {
        [
            0.expr(),
            1.expr(),
            0.expr(),
            false.expr(),
            false.expr(),
            0.expr(),
            1.expr(),
        ]
    }

    pub(crate) fn witness_store(
        _region: &mut Region<'_, F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        rlc: F,
        mult: F,
        num_nibbles: usize,
        placeholder_is_odd: bool,
        parent_rlc: F,
        parent_mult: F,
    ) -> Result<(), Error> {
        let values = [
            rlc,
            mult,
            num_nibbles.scalar(),
            (num_nibbles % 2 == 1).scalar(),
            placeholder_is_odd.scalar(),
            parent_rlc,
            parent_mult,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }
    
    pub(crate) fn witness_store_cached(
        _region: &mut CachedRegion<F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        rlc: F,
        mult: F,
        num_nibbles: usize,
        placeholder_is_odd: bool,
        parent_rlc: F,
        parent_mult: F,
    ) -> Result<(), Error> {
        let values = [
            rlc,
            mult,
            num_nibbles.scalar(),
            (num_nibbles % 2 == 1).scalar(),
            placeholder_is_odd.scalar(),
            parent_rlc,
            parent_mult,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }

    pub(crate) fn witness_load(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<KeyDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.rlc.assign(region, offset, values[0])?;
        self.mult.assign(region, offset, values[1])?;
        self.num_nibbles.assign(region, offset, values[2])?;
        self.is_odd.assign(region, offset, values[3])?;
        self.drifted_is_odd.assign(region, offset, values[4])?;
        self.drifted_rlc.assign(region, offset, values[5])?;
        self.drifted_mult.assign(region, offset, values[6])?;

        Ok(KeyDataWitness {
            rlc: values[0],
            mult: values[1],
            num_nibbles: values[2].get_lower_32() as usize,
            is_odd: values[3] != F::zero(),
            drifted_is_odd: values[4] != F::zero(),
            drifted_rlc: values[5],
            drifted_mult: values[6],
        })
    }

    pub(crate) fn witness_load_cached(
        &self,
        region:  &mut CachedRegion<F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<KeyDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.rlc.assign_cached(region, offset, values[0])?;
        self.mult.assign_cached(region, offset, values[1])?;
        self.num_nibbles.assign_cached(region, offset, values[2])?;
        self.is_odd.assign_cached(region, offset, values[3])?;
        self.drifted_is_odd.assign_cached(region, offset, values[4])?;
        self.drifted_rlc.assign_cached(region, offset, values[5])?;
        self.drifted_mult.assign_cached(region, offset, values[6])?;

        Ok(KeyDataWitness {
            rlc: values[0],
            mult: values[1],
            num_nibbles: values[2].get_lower_32() as usize,
            is_odd: values[3] != F::zero(),
            drifted_is_odd: values[4] != F::zero(),
            drifted_rlc: values[5],
            drifted_mult: values[6],
        })
    }

}

#[derive(Clone, Debug, Default)]
pub(crate) struct ParentData<F> {
    pub(crate) rlc: Cell<F>,
    pub(crate) is_root: Cell<F>,
    pub(crate) is_placeholder: Cell<F>,
    pub(crate) placeholder_rlc: Cell<F>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ParentDataWitness<F> {
    pub(crate) rlc: F,
    pub(crate) is_root: bool,
    pub(crate) is_placeholder: bool,
    pub(crate) placeholder_rlc: F,
}

impl<F: Field> ParentData<F> {
    pub(crate) fn load(
        description: &'static str,
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        offset: Expression<F>,
    ) -> Self {
        let parent_data = ParentData {
            rlc: cb.query_cell(),
            is_root: cb.query_cell(),
            is_placeholder: cb.query_cell(),
            placeholder_rlc: cb.query_cell(),
        };
        circuit!([meta, cb], {
            memory.load(
                description,
                cb,
                offset,
                &[
                    parent_data.rlc.expr(),
                    parent_data.is_root.expr(),
                    parent_data.is_placeholder.expr(),
                    parent_data.placeholder_rlc.expr(),
                ],
            );
        });
        parent_data
    }

    pub(crate) fn store(
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        values: [Expression<F>; 4],
    ) {
        memory.store(cb, &values);
    }

    pub(crate) fn witness_store(
        _region: &mut Region<'_, F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        rlc: F,
        force_hashed: bool,
        is_placeholder: bool,
        placeholder_rlc: F,
    ) -> Result<(), Error> {
        let values = [
            rlc,
            force_hashed.scalar(),
            is_placeholder.scalar(),
            placeholder_rlc,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }

    pub(crate) fn witness_store_cached(
        _region: &mut CachedRegion<F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        rlc: F,
        force_hashed: bool,
        is_placeholder: bool,
        placeholder_rlc: F,
    ) -> Result<(), Error> {
        let values = [
            rlc,
            force_hashed.scalar(),
            is_placeholder.scalar(),
            placeholder_rlc,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }

    pub(crate) fn witness_load(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<ParentDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.rlc.assign(region, offset, values[0])?;
        self.is_root.assign(region, offset, values[1])?;
        self.is_placeholder.assign(region, offset, values[2])?;
        self.placeholder_rlc.assign(region, offset, values[3])?;

        Ok(ParentDataWitness {
            rlc: values[0],
            is_root: values[1] == 1.scalar(),
            is_placeholder: values[2] == 1.scalar(),
            placeholder_rlc: values[3],
        })
    }

    pub(crate) fn witness_load_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<ParentDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.rlc.assign_cached(region, offset, values[0])?;
        self.is_root.assign_cached(region, offset, values[1])?;
        self.is_placeholder.assign_cached(region, offset, values[2])?;
        self.placeholder_rlc.assign_cached(region, offset, values[3])?;

        Ok(ParentDataWitness {
            rlc: values[0],
            is_root: values[1] == 1.scalar(),
            is_placeholder: values[2] == 1.scalar(),
            placeholder_rlc: values[3],
        })
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MainData<F> {
    pub(crate) proof_type: Cell<F>,
    pub(crate) is_below_account: Cell<F>,
    pub(crate) address_rlc: Cell<F>,
    pub(crate) root_prev: Cell<F>,
    pub(crate) root: Cell<F>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MainDataWitness<F> {
    pub(crate) proof_type: usize,
    pub(crate) is_below_account: bool,
    pub(crate) address_rlc: F,
    pub(crate) root_prev: F,
    pub(crate) root: F,
}

impl<F: Field> MainData<F> {
    pub(crate) fn load(
        description: &'static str,
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        offset: Expression<F>,
    ) -> Self {
        let main_data = MainData {
            proof_type: cb.query_cell(),
            is_below_account: cb.query_cell(),
            address_rlc: cb.query_cell(),
            root_prev: cb.query_cell(),
            root: cb.query_cell(),
        };
        circuit!([meta, cb], {
            memory.load(
                description,
                cb,
                offset,
                &[
                    main_data.proof_type.expr(),
                    main_data.is_below_account.expr(),
                    main_data.address_rlc.expr(),
                    main_data.root_prev.expr(),
                    main_data.root.expr(),
                ],
            );
        });
        main_data
    }

    pub(crate) fn store(
        cb: &mut ConstraintBuilder<F>,
        memory: &MemoryBank<F>,
        values: [Expression<F>; 5],
    ) {
        memory.store(cb, &values);
    }

    pub(crate) fn witness_store(
        _region: &mut Region<'_, F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        proof_type: usize,
        is_below_account: bool,
        address_rlc: F,
        root_prev: F,
        root: F,
    ) -> Result<(), Error> {
        let values = [
            proof_type.scalar(),
            is_below_account.scalar(),
            address_rlc,
            root_prev,
            root,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }

    pub(crate) fn witness_store_cached(
        _region:  &mut CachedRegion<F>,
        offset: usize,
        memory: &mut MemoryBank<F>,
        proof_type: usize,
        is_below_account: bool,
        address_rlc: F,
        root_prev: F,
        root: F,
    ) -> Result<(), Error> {
        let values = [
            proof_type.scalar(),
            is_below_account.scalar(),
            address_rlc,
            root_prev,
            root,
        ];
        memory.witness_store(offset, &values);

        Ok(())
    }

    pub(crate) fn witness_load(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<MainDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.proof_type.assign(region, offset, values[0])?;
        self.is_below_account.assign(region, offset, values[1])?;
        self.address_rlc.assign(region, offset, values[2])?;
        self.root_prev.assign(region, offset, values[3])?;
        self.root.assign(region, offset, values[4])?;

        Ok(MainDataWitness {
            proof_type: values[0].get_lower_32() as usize,
            is_below_account: values[1] == 1.scalar(),
            address_rlc: values[2],
            root_prev: values[3],
            root: values[4],
        })
    }

    pub(crate) fn witness_load_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        memory: &MemoryBank<F>,
        load_offset: usize,
    ) -> Result<MainDataWitness<F>, Error> {
        let values = memory.witness_load(load_offset);

        self.proof_type.assign_cached(region, offset, values[0])?;
        self.is_below_account.assign_cached(region, offset, values[1])?;
        self.address_rlc.assign_cached(region, offset, values[2])?;
        self.root_prev.assign_cached(region, offset, values[3])?;
        self.root.assign_cached(region, offset, values[4])?;

        Ok(MainDataWitness {
            proof_type: values[0].get_lower_32() as usize,
            is_below_account: values[1] == 1.scalar(),
            address_rlc: values[2],
            root_prev: values[3],
            root: values[4],
        })
    }

}

/// Add the nibble from the drifted branch
pub(crate) fn nibble_rlc<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    key_rlc: Expression<F>,
    key_mult_prev: Expression<F>,
    is_key_odd: Expression<F>,
    nibble: Expression<F>,
    r: &[Expression<F>],
) -> (Expression<F>, Expression<F>) {
    circuit!([meta, cb], {
        let (nibble_mult, mult) = ifx! {is_key_odd => {
            // The nibble will be added as the least significant nibble, the multiplier needs to advance
            (1.expr(), r[0].expr())
        } elsex {
            // The nibble will be added as the most significant nibble, the multiplier needs to stay the same
            (16.expr(), 1.expr())
        }};
        (
            key_rlc + nibble * nibble_mult * key_mult_prev.expr(),
            key_mult_prev * mult,
        )
    })
}

pub(crate) fn leaf_key_rlc<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    bytes: &[Expression<F>],
    key_mult_prev: Expression<F>,
    is_key_odd: Expression<F>,
    r: &[Expression<F>],
) -> Expression<F> {
    circuit!([meta, cb], {
        // Add the odd nibble first if we have one.
        let (rlc, mult) = ifx! {is_key_odd => {
            (get_terminal_odd_nibble(bytes[0].expr()) * key_mult_prev.expr(), r[0].expr())
        } elsex {
            require!(bytes[0] => KEY_TERMINAL_PREFIX_EVEN);
            (0.expr(), 1.expr())
        }};
        (rlc, key_mult_prev * mult).rlc_chain(bytes[1..].rlc(r))
    })
}

pub(crate) fn ext_key_rlc<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    bytes: &[Expression<F>],
    key_mult_prev: Expression<F>,
    is_odd: Expression<F>,
    rlc_mult_first_odd: Expression<F>,
    key_mult_first_odd: Expression<F>,
    r: &[Expression<F>],
) -> Expression<F> {
    circuit!([meta, cb], {
        // Add the odd nibble first if we have one.
        let (rlc, mult) = ifx! {is_odd => {
            (get_ext_odd_nibble(bytes[0].expr()) * key_mult_prev.expr() * rlc_mult_first_odd, key_mult_first_odd.expr())
        } elsex {
            require!(bytes[0] => KEY_PREFIX_EVEN);
            (0.expr(), 1.expr())
        }};
        (rlc, key_mult_prev * mult).rlc_chain(bytes[1..].rlc(r))
    })
}

pub(crate) fn ext_key_rlc_value<F: Field>(
    bytes: &[F],
    key_mult_prev: F,
    is_odd: bool,
    rlc_mult_first_odd: F,
    key_mult_first_odd: F,
    r: F,
) -> (F, F) {
    // Add the odd nibble first if we have one.
    let (rlc, mult) = if is_odd {
        (
            get_ext_odd_nibble_value(bytes[0]) * key_mult_prev * rlc_mult_first_odd,
            key_mult_first_odd,
        )
    } else {
        assert!(bytes[0] == KEY_PREFIX_EVEN.scalar());
        (0.scalar(), 1.scalar())
    };
    (rlc, key_mult_prev * mult).rlc_chain_value_f(&bytes[1..], r)
}

// Returns the number of nibbles stored in a key value
pub(crate) mod num_nibbles {
    use crate::circuit_tools::constraint_builder::ConstraintBuilder;
    use crate::{_cb, circuit};
    use eth_types::Field;
    use halo2_proofs::plonk::Expression;

    pub(crate) fn expr<F: Field>(
        key_len: Expression<F>,
        is_key_odd: Expression<F>,
    ) -> Expression<F> {
        circuit!([meta, _cb!()], {
            ifx! {is_key_odd => {
                key_len.expr() * 2.expr() - 1.expr()
            } elsex {
                (key_len.expr() - 1.expr()) * 2.expr()
            }}
        })
    }
    pub(crate) fn value(key_len: usize, is_key_odd: bool) -> usize {
        if is_key_odd {
            key_len * 2 - 1
        } else {
            (key_len - 1) * 2
        }
    }
}

pub(crate) fn extend_rand<F: Field>(r: &[Expression<F>]) -> Vec<Expression<F>> {
    [
        r.to_vec(),
        r.iter()
            .map(|v| r.last().unwrap().expr() * v.clone())
            .collect::<Vec<_>>(),
    ]
    .concat()
}

pub(crate) fn parent_memory(is_s: bool) -> String {
    (if is_s { "parent_s" } else { "parent_c" }).to_string()
}

pub(crate) fn key_memory(is_s: bool) -> String {
    (if is_s { "key_s" } else { "key_c" }).to_string()
}

pub(crate) fn main_memory() -> String {
    "main".to_string()
}

/// MPTConstraintBuilder
#[derive(Clone)]
pub struct MPTConstraintBuilder<F> {
    pub base: ConstraintBuilder<F>,
    /// Number of non-zero s bytes
    pub length_s: Vec<(Expression<F>, Expression<F>)>,
    /// Number of non-zero s bytes in c bytes (when only using s length)
    pub length_sc: Expression<F>,
    /// Number of non-zero c bytes
    pub length_c: Vec<(Expression<F>, Expression<F>)>,
    /// The range to check in s bytes
    pub range_s: Vec<(Expression<F>, Expression<F>)>,

    // /// Store expression over max degree
    // pub stored_expressions: Vec<StoredExpression<F>>,
    // /// Store randomness over max degree
    // power_of_randomness: [Expression<F>; HASH_WIDTH],
}

// #[derive(Debug, Clone)]
// pub struct StoredExpression<F> {
//     pub(crate) name: String,
//     cell: Cell<F>,
//     cell_type: CellType,
//     expr: Expression<F>,
//     expr_id: String,
// }

// impl<F: Field> StoredExpression<F> {
//     pub fn assign(
//         &self,
//         region: &mut CachedRegion<'_, '_, F>,
//         offset: usize,
//     ) -> Result<AssignedCell<F, F>, Error> {
//         let value = self.expr.evaluate(
//             &|scalar| scalar,
//             &|_| unimplemented!("selector column"),
//             &|fixed_query| {
//                 region.get_fixed(offset, fixed_query.column_index(), fixed_query.rotation())
//             },
//             &|advide_query| {
//                 region.get_advice(offset, advide_query.column_index(), advide_query.rotation())
//             },
//             &|instance_query| {
//                 region.get_instance(
//                     offset,
//                     instance_query.column_index(),
//                     instance_query.rotation(),
//                 )
//             },
//             &|_| unimplemented!(),
//             &|a| -a,
//             &|a, b| a + b,
//             &|a, b| a * b,
//             &|a, scalar| a * scalar,
//         );
//         self.cell.assign_cached(region, offset, value)
//     }
// }

/// Length is set in the configuration of the rows where unused columns might
/// appear (and which need to be checked to be zeros). From the vector of
/// lengths, the expression is computed that returns the length of the used
/// columns for each row. This enables a generalised constraint (for all rows)
/// for the values in the unused columns being zeros in mpt.rs.
impl<F: Field> MPTConstraintBuilder<F> {
    const DEFAULT_LENGTH_S: usize = 34;
    const DEFAULT_LENGTH_C: usize = 32;
    const NUM_BYTES_SKIP: usize = 2; // RLP bytes never need to be zero checked
    const DEFAULT_RANGE: FixedTableTag = FixedTableTag::RangeKeyLen256;

    pub(crate) fn new(
        max_degree: usize, 
        cell_manager: Option<CellManager_<F>>,
        power_of_randomness: [Expression<F>; HASH_WIDTH],
    ) -> Self {
        let mut base = ConstraintBuilder::new(max_degree, cell_manager);
        base.set_power_of_randomness(power_of_randomness);
        MPTConstraintBuilder {
            base,
            length_s: Vec::new(),
            length_sc: 0.expr(),
            length_c: Vec::new(),
            range_s: Vec::new(),
            // stored_expressions: Vec::new(),
            // power_of_randomness
        }
    }

    pub(crate) fn set_length_s(&mut self, length: Expression<F>) {
        self.length_s.push((
            self.base.get_condition_expr(),
            Self::DEFAULT_LENGTH_S.expr() - (length - Self::NUM_BYTES_SKIP.expr()),
        ));
    }

    pub(crate) fn set_length_c(&mut self, length: Expression<F>) {
        self.length_c.push((
            self.base.get_condition_expr(),
            Self::DEFAULT_LENGTH_C.expr() - (length - Self::NUM_BYTES_SKIP.expr()),
        ));
    }

    pub(crate) fn set_length_sc(&mut self, is_s: bool, length: Expression<F>) {
        if is_s {
            self.set_length_s(length);
        } else {
            self.set_length_c(length);
        }
    }

    pub(crate) fn set_length(&mut self, length: Expression<F>) {
        self.set_length_s(length);
        self.length_sc = self.length_sc.expr() + self.base.get_condition_expr();
    }

    pub(crate) fn get_length_s(&self) -> Expression<F> {
        Self::DEFAULT_LENGTH_S.expr() - self.length_s.apply_conditions()
    }

    pub(crate) fn get_length_c(&self) -> Expression<F> {
        Self::DEFAULT_LENGTH_C.expr() - self.length_c.apply_conditions()
    }

    pub(crate) fn set_range_s(&mut self, range: Expression<F>) {
        self.range_s.push((
            self.base.get_condition_expr(),
            Self::DEFAULT_RANGE.expr() - range,
        ));
    }

    pub(crate) fn get_range_s(&self) -> Expression<F> {
        Self::DEFAULT_RANGE.expr() - self.range_s.apply_conditions()
    }

    pub(crate) fn query_cell_with_type(&mut self, cell_type: CellType) -> Cell<F> {
        self.query_cells(cell_type, 1).first().unwrap().clone()
    }

    fn query_cells(&mut self, cell_type: CellType, count: usize) -> Vec<Cell<F>> {
        self.base.cell_manager.clone().unwrap().query_cells(cell_type, count)
    }

    // pub(crate) fn add_constraint(&mut self, name: &'static str, constraint: Expression<F>) {
    //     let constraint = self.split_expression(
    //         name,
    //         constraint * self.base.get_condition_expr(),
    //         MAX_DEGREE - IMPLICIT_DEGREE,
    //     );

    //     self.validate_degree(constraint.degree(), name);
    //     self.base.add_constraint(name, constraint);
    // }

    // pub(crate) fn validate_degree(&self, degree: usize, name: &'static str) {
    //     // We need to subtract IMPLICIT_DEGREE from MAX_DEGREE because all expressions
    //     // will be multiplied by state selector and q_step/q_step_first
    //     // selector.
    //     debug_assert!(
    //         degree <= MAX_DEGREE - IMPLICIT_DEGREE,
    //         "Expression {} degree too high: {} > {}",
    //         name,
    //         degree,
    //         MAX_DEGREE - IMPLICIT_DEGREE,
    //     );
    // }
    
    // pub(crate) fn add_lookup(&mut self, name: &str, lookup: Lookup<F>) {
    //     let lookup = match self.base.get_condition() {
    //         Some(condition) => lookup.conditional(condition),
    //         None => lookup,
    //     };
    //     let compressed_expr = self.split_expression(
    //         "Lookup compression",
    //         rlc::expr(&lookup.input_exprs(), &self.power_of_randomness),
    //         MAX_DEGREE - IMPLICIT_DEGREE,
    //     );
    //     self.store_expression(name, compressed_expr, CellType::Lookup(lookup.table()));
    // }

    // pub(crate) fn store_expression(
    //     &mut self,
    //     name: &str,
    //     expr: Expression<F>,
    //     cell_type: CellType,
    // ) -> Expression<F> {
    //     // Check if we already stored the expression somewhere
    //     let stored_expression = self.find_stored_expression(expr.clone(), cell_type);
    //     match stored_expression {
    //         Some(stored_expression) => {
    //             debug_assert!(
    //                 !matches!(cell_type, CellType::Lookup(_)),
    //                 "The same lookup is done multiple times",
    //             );
    //             stored_expression.cell.expr()
    //         }
    //         None => {
    //             // Even if we're building expressions for the next step,
    //             // these intermediate values need to be stored in the current step.
    //             // let in_next_step = self.in_next_step;
    //             // self.in_next_step = false;
    //             let cell = self.query_cell_with_type(cell_type);
    //             // self.in_next_step = in_next_step;

    //             // Require the stored value to equal the value of the expression
    //             let name = format!("{} (stored expression)", name);
    //             self.base.add_constraint(
    //                 Box::leak(name.clone().into_boxed_str()),
    //                 cell.expr() - expr.clone(),
    //             );

    //             self.stored_expressions.push(StoredExpression {
    //                 name,
    //                 cell: cell.clone(),
    //                 cell_type,
    //                 expr_id: expr.identifier(),
    //                 expr,
    //             });
    //             cell.expr()
    //         }
    //     }
    // }

    // pub(crate) fn find_stored_expression(
    //     &self,
    //     expr: Expression<F>,
    //     cell_type: CellType,
    // ) -> Option<&StoredExpression<F>> {
    //     let expr_id = expr.identifier();
    //     self.stored_expressions
    //         .iter()
    //         .find(|&e| e.cell_type == cell_type && e.expr_id == expr_id)
    // }

    // fn split_expression(
    //     &mut self,
    //     name: &'static str,
    //     expr: Expression<F>,
    //     max_degree: usize,
    // ) -> Expression<F> {
    //     if expr.degree() > max_degree {
    //         match expr {
    //             Expression::Negated(poly) => {
    //                 Expression::Negated(Box::new(self.split_expression(name, *poly, max_degree)))
    //             }
    //             Expression::Scaled(poly, v) => {
    //                 Expression::Scaled(Box::new(self.split_expression(name, *poly, max_degree)), v)
    //             }
    //             Expression::Sum(a, b) => {
    //                 let a = self.split_expression(name, *a, max_degree);
    //                 let b = self.split_expression(name, *b, max_degree);
    //                 a + b
    //             }
    //             Expression::Product(a, b) => {
    //                 let (mut a, mut b) = (*a, *b);
    //                 while a.degree() + b.degree() > max_degree {
    //                     let mut split = |expr: Expression<F>| {
    //                         if expr.degree() > max_degree {
    //                             self.split_expression(name, expr, max_degree)
    //                         } else {
    //                             self.store_expression(name, expr, CellType::Storage)
    //                         }
    //                     };
    //                     if a.degree() >= b.degree() {
    //                         a = split(a);
    //                     } else {
    //                         b = split(b);
    //                     }
    //                 }
    //                 a * b
    //             }
    //             _ => expr.clone(),
    //         }
    //     } else {
    //         expr.clone()
    //     }
    // }


}

/// Returns `1` when `value == 0`, and returns `0` otherwise.
#[derive(Clone, Debug, Default)]
pub struct IsEmptyTreeGadget<F> {
    is_in_empty_trie: IsEqualGadget<F>,
    is_in_empty_branch: IsEqualGadget<F>,
}

impl<F: Field> IsEmptyTreeGadget<F> {
    pub(crate) fn construct(
        cb: &mut ConstraintBuilder<F>,
        parent_rlc: Expression<F>,
        r: &[Expression<F>],
    ) -> Self {
        circuit!([meta, cb], {
            let empty_root_rlc = EMPTY_TRIE_HASH
                .iter()
                .map(|v| v.expr())
                .collect::<Vec<_>>()
                .rlc(&r);
            let is_in_empty_trie =
                IsEqualGadget::construct(cb, parent_rlc.expr(), empty_root_rlc.expr());
            let is_in_empty_branch = IsEqualGadget::construct(cb, parent_rlc.expr(), 0.expr());

            Self {
                is_in_empty_trie,
                is_in_empty_branch,
            }
        })
    }

    pub(crate) fn expr(&self) -> Expression<F> {
        or::expr(&[self.is_in_empty_trie.expr(), self.is_in_empty_branch.expr()])
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        parent_rlc: F,
        r: F,
    ) -> Result<(), Error> {
        self.is_in_empty_trie
            .assign(region, offset, parent_rlc, EMPTY_TRIE_HASH.rlc_value(r))?;
        self.is_in_empty_branch
            .assign(region, offset, parent_rlc, 0.scalar())?;
        Ok(())
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        parent_rlc: F,
        r: F,
    ) -> Result<(), Error> {
        self.is_in_empty_trie
            .assign_cached(region, offset, parent_rlc, EMPTY_TRIE_HASH.rlc_value(r))?;
        self.is_in_empty_branch
            .assign_cached(region, offset, parent_rlc, 0.scalar())?;
        Ok(())
    }

}

/// Handles drifted leaves
#[derive(Clone, Debug, Default)]
pub struct DriftedGadget<F> {
    drifted_rlp_key: ListKeyGadget<F>,
    drifted_mult: Cell<F>,
}

impl<F: Field> DriftedGadget<F> {
    pub(crate) fn construct(
        cb: &mut MPTConstraintBuilder<F>,
        parent_data: &[ParentData<F>],
        key_data: &[KeyData<F>],
        expected_key_rlc: &[Expression<F>],
        leaf_no_key_rlc: &[Expression<F>],
        drifted_bytes: &[Expression<F>],
        r: &[Expression<F>],
    ) -> Self {
        let mut config = DriftedGadget::default();
        circuit!([meta, cb.base], {
            ifx! {parent_data[true.idx()].is_placeholder.expr() + parent_data[false.idx()].is_placeholder.expr() => {
                config.drifted_rlp_key = ListKeyGadget::construct(&mut cb.base, drifted_bytes);
                config.drifted_mult = cb.base.query_cell();
                for is_s in [true, false] {
                    ifx! {parent_data[is_s.idx()].is_placeholder.expr() => {
                        // Check that the drifted leaf is unchanged and is stored at `drifted_index`.
                        // TODO(Brecht): Length can change so need to add RLP consistency checks?

                        // Calculate the drifted key RLC
                        // Get the key RLC for the drifted branch
                        let (key_rlc, key_mult, is_key_odd) = (
                            key_data[is_s.idx()].drifted_rlc.expr(),
                            key_data[is_s.idx()].drifted_mult.expr(),
                            key_data[is_s.idx()].drifted_is_odd.expr(),
                        );
                        let key_rlc = key_rlc.expr() + config.drifted_rlp_key.key.expr(&mut cb.base, config.drifted_rlp_key.key_value.clone(), key_mult.expr(), is_key_odd, &r);
                        // The key of the drifted leaf needs to match the key of the leaf
                        require!(key_rlc => expected_key_rlc[is_s.idx()]);

                        // Total number of nibbles needs to be KEY_LEN_IN_NIBBLES
                        // TODO(Brecht): RLC encoding would be the same for some addresses without checking the length
                        //let num_nibbles = num_nibbles::expr(config.drifted_rlp_key.key_value.len(), is_key_odd.expr());
                        //require!(key_data.num_nibbles.expr() + num_nibbles => KEY_LEN_IN_NIBBLES);

                        // Complete the drifted leaf rlc by adding the bytes on the value row
                        let leaf_rlc = (config.drifted_rlp_key.rlc(&r), config.drifted_mult.expr()).rlc_chain(leaf_no_key_rlc[is_s.idx()].expr());
                        // The drifted leaf needs to be stored in the branch at `drifted_index`.
                        require!((1, leaf_rlc, config.drifted_rlp_key.rlp_list.num_bytes(), parent_data[is_s.idx()].placeholder_rlc.expr()) => @"keccak");

                        // Check zero bytes and mult_diff
                        require!((FixedTableTag::RMult, config.drifted_rlp_key.num_bytes_on_key_row(), config.drifted_mult.expr()) => @"fixed");
                        cb.set_length(config.drifted_rlp_key.num_bytes_on_key_row());
                    }
                }}
            }}
            config
        })
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        parent_data: &[ParentDataWitness<F>],
        drifted_list_bytes: &[u8],
        drifted_bytes: &[u8],
        r: F,
    ) -> Result<(), Error> {
        if parent_data[true.idx()].is_placeholder || parent_data[false.idx()].is_placeholder {
            let drifted_key_witness =
                self.drifted_rlp_key
                    .assign(region, offset, drifted_list_bytes, drifted_bytes)?;
            let (_, leaf_mult) = drifted_key_witness.rlc_leaf(r);
            self.drifted_mult.assign(region, offset, leaf_mult)?;
        }
        Ok(())
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        parent_data: &[ParentDataWitness<F>],
        drifted_list_bytes: &[u8],
        drifted_bytes: &[u8],
        r: F,
    ) -> Result<(), Error> {
        if parent_data[true.idx()].is_placeholder || parent_data[false.idx()].is_placeholder {
            let drifted_key_witness =
                self.drifted_rlp_key
                    .assign_cached(region, offset, drifted_list_bytes, drifted_bytes)?;
            let (_, leaf_mult) = drifted_key_witness.rlc_leaf(r);
            self.drifted_mult.assign_cached(region, offset, leaf_mult)?;
        }
        Ok(())
    }

}

/// Handles wrong leaves
#[derive(Clone, Debug, Default)]
pub struct WrongGadget<F> {
    wrong_rlp_key: ListKeyGadget<F>,
    wrong_mult: Cell<F>,
    is_key_equal: IsEqualGadget<F>,
    wrong_key: Option<Expression<F>>,
}

impl<F: Field> WrongGadget<F> {
    pub(crate) fn construct(
        cb: &mut MPTConstraintBuilder<F>,
        expected_address: Expression<F>,
        is_non_existing: Expression<F>,
        key_value: &RLPValueGadget<F>,
        key_rlc: &Expression<F>,
        wrong_bytes: &[Expression<F>],
        is_in_empty_tree: Expression<F>,
        key_data: KeyData<F>,
        r: &[Expression<F>],
    ) -> Self {
        let mut config = WrongGadget::default();
        circuit!([meta, cb.base], {
            // Get the previous key data
            ifx! {is_non_existing, not!(is_in_empty_tree) => {
                // Calculate the key
                config.wrong_rlp_key = ListKeyGadget::construct(&mut cb.base, &wrong_bytes);
                let key_rlc_wrong = key_data.rlc.expr() + config.wrong_rlp_key.key.expr(
                    &mut cb.base,
                    config.wrong_rlp_key.key_value.clone(),
                    key_data.mult.expr(),
                    key_data.is_odd.expr(),
                    r,
                );
                // Check that it's the key as expected
                require!(key_rlc_wrong => expected_address);

                // Now make sure this address is different than the one of the leaf
                config.is_key_equal = IsEqualGadget::construct(
                    &mut cb.base,
                    key_rlc.expr(),
                    expected_address,
                );
                require!(config.is_key_equal => false);
                // Make sure the lengths of the keys are the same
                require!(config.wrong_rlp_key.key_value.len() => key_value.len());
                // RLC bytes zero check
                cb.set_length(config.wrong_rlp_key.num_bytes_on_key_row());
            }}
            config
        })
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        is_non_existing: bool,
        key_rlc: &[F],
        list_bytes: &[u8],
        wrong_bytes: &[u8],
        for_placeholder_s: bool,
        key_data: KeyDataWitness<F>,
        r: F,
    ) -> Result<F, Error> {
        if is_non_existing {
            let wrong_witness =
                self.wrong_rlp_key
                    .assign(region, offset, list_bytes, wrong_bytes)?;
            let (key_rlc_wrong, _) = wrong_witness.key.key(
                wrong_witness.key_value.clone(),
                key_data.rlc,
                key_data.mult,
                r,
            );

            self.is_key_equal.assign(
                region,
                offset,
                key_rlc[for_placeholder_s.idx()],
                key_rlc_wrong,
            )?;
            Ok(key_rlc_wrong)
        } else {
            Ok(key_rlc[for_placeholder_s.idx()])
        }
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<F>,
        offset: usize,
        is_non_existing: bool,
        key_rlc: &[F],
        list_bytes: &[u8],
        wrong_bytes: &[u8],
        for_placeholder_s: bool,
        key_data: KeyDataWitness<F>,
        r: F,
    ) -> Result<F, Error> {
        if is_non_existing {
            let wrong_witness =
                self.wrong_rlp_key
                    .assign_cached(region, offset, list_bytes, wrong_bytes)?;
            let (key_rlc_wrong, _) = wrong_witness.key.key(
                wrong_witness.key_value.clone(),
                key_data.rlc,
                key_data.mult,
                r,
            );

            self.is_key_equal.assign_cached(
                region,
                offset,
                key_rlc[for_placeholder_s.idx()],
                key_rlc_wrong,
            )?;
            Ok(key_rlc_wrong)
        } else {
            Ok(key_rlc[for_placeholder_s.idx()])
        }
    }

}
