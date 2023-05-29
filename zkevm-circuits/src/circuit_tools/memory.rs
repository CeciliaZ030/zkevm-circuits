//! Memory
use crate::util::{query_expression, Expr};
use eth_types::Field;
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression},
    poly::Rotation,
};
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use super::constraint_builder::{merge_lookups, ConstraintBuilder};
use super::cell_manager::Table;

#[derive(Clone, Debug, Default)]
pub(crate) struct Memory<F> {
    pub(crate) columns: Vec<Column<Advice>>,
    banks: Vec<MemoryBank<F>>,
}

impl<F: Field> Memory<F> {
    pub(crate) fn new(columns: Vec<Column<Advice>>) -> Self {
        Self {
            columns,
            banks: Vec::new(),
        }
    }

    pub(crate) fn allocate<S: AsRef<str>>(
        &mut self,
        meta: &mut ConstraintSystem<F>,
        tag: S,
    ) -> &MemoryBank<F> {
        self.banks
            .push(MemoryBank::new(meta, self.columns[self.banks.len()], tag));
        self.banks.last().unwrap()
    }

    pub(crate) fn get<S: AsRef<str>>(&self, tag: S) -> &MemoryBank<F> {
        for bank in self.banks.iter() {
            if bank.tag() == tag.as_ref() {
                return bank;
            }
        }
        unreachable!()
    }

    pub(crate) fn get_mut<S: AsRef<str>>(&mut self, tag: S) -> &mut MemoryBank<F> {
        for bank in self.banks.iter_mut() {
            if bank.tag() == tag.as_ref() {
                return bank;
            }
        }
        unreachable!()
    }

    pub(crate) fn generate_constraints(
        &self,
        cb: &mut ConstraintBuilder<F>,
        is_first_row: Expression<F>,
    ) {
        for bank in self.banks.iter() {
            // 处理 lookup_tables
            // 通过纵向压缩的table.merged_cond 是否为1，来做 bank.next == bank.cur + merged_cound
            bank.generate_constraints(cb, is_first_row.expr());
            // require 但单个和 merged_cond 都为 bool
            cb.generate_lookup_table_checks(bank.tag());

            // 拿出需要的 lookups（不是table）并在牌堆中删掉
            let lookups = cb.consume_lookups(&[bank.tag()]);
            // 就是每个 bank 在这一个 gate 中只允许一个 lookup
            /// 如 main_bank: [
            ///         s1 * (proof_type1, addr_rlc1), 
            ///         s2 * (proof_type2, addr_rlc2)
            /// ] 
            /// 最后变成 ==> meta.lookup_any[
            ///                     value_expr        |  table_expr
            ///             (s1*proof_type1+s2*proof_type2, proof_type)
            ///             (s1*addr_rlc1+s2*addr_rlc2, addr_rlc)
            ///         ]
            /// s1, s2 只有一个为 1，因为跑 memory.generate_constraints() 是结束一个 MPT Gate 
            /// 而一个 MPT Gate 对应一个 node，那么上下级的 prent, child, proof_type 都是唯一的
            /// 如果有 !ifx {} elsex {}, 此时保证只有一个branch taken
            if !lookups.is_empty() {
                println!("{}: {}", bank.tag, lookups.len());
                // table = [sel, lrc_a, lrc_b, ...] 纵向压缩
                let (_, values) = merge_lookups(cb, lookups);
                crate::circuit!([meta, cb], {
                    // 展开成 cb.lookup(descr, tag, val) 
                    // 然后又push进 cb 的 lookups 里面
                    require!(values => @bank.tag());
                })
            }
        }
    }

    pub(crate) fn clear_witness_data(&mut self) {
        for bank in self.banks.iter_mut() {
            bank.clear_witness_data();
        }
    }

    pub(crate) fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        height: usize,
    ) -> Result<(), Error> {
        for bank in self.banks.iter() {
            bank.assign(layouter, height)?;
        }
        Ok(())
    }

    pub(crate) fn tags(&self) -> Vec<String> {
        self.banks.iter().map(|bank| bank.tag()).collect()
    }
}

impl<F: Field, S: AsRef<str>> Index<S> for Memory<F> {
    type Output = MemoryBank<F>;

    fn index(&self, tag: S) -> &Self::Output {
        for bank in self.banks.iter() {
            if bank.tag() == tag.as_ref() {
                return bank;
            }
        }
        unreachable!()
    }
}

impl<F: Field, S: AsRef<str>> IndexMut<S> for Memory<F> {
    fn index_mut(&mut self, tag: S) -> &mut Self::Output {
        for bank in self.banks.iter_mut() {
            if bank.tag() == tag.as_ref() {
                return bank;
            }
        }
        unreachable!()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryBank<F> {
    column: Column<Advice>,
    tag: String,
    cur: Expression<F>,
    next: Expression<F>,
    store_offsets: Vec<usize>,
    stored_values: Vec<Vec<F>>,
    _marker: PhantomData<F>,
}

impl<F: Field> MemoryBank<F> {
    pub(crate) fn new<S: AsRef<str>>(
        meta: &mut ConstraintSystem<F>,
        column: Column<Advice>,
        tag: S,
    ) -> Self {
        let mut cur = 0.expr();
        let mut next = 0.expr();
        query_expression(meta, |meta| {
            cur = meta.query_advice(column, Rotation::cur());
            next = meta.query_advice(column, Rotation::next());
        });
        Self {
            column,
            tag: tag.as_ref().to_owned(),
            cur,
            next,
            store_offsets: Vec::new(),
            stored_values: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub(crate) fn key(&self) -> Expression<F> {
        self.cur.expr()
    }

    pub(crate) fn load(
        &self,
        description: &'static str,
        cb: &mut ConstraintBuilder<F>,
        offset: Expression<F>,
        values: &[Expression<F>],
    ) {
        self.load_with_key(description, cb, self.key() - offset, values);
    }

    pub(crate) fn load_with_key(
        &self,
        description: &'static str,
        cb: &mut ConstraintBuilder<F>,
        key: Expression<F>,
        values: &[Expression<F>],
    ) {
        // Insert the key in the front
        let mut key_and_values = values.to_vec();
        key_and_values.insert(0, key);
        cb.lookup(description, self.tag(), key_and_values);
    }

    pub(crate) fn store(&self, cb: &mut ConstraintBuilder<F>, values: &[Expression<F>]) {
        self.store_with_key(cb, self.key() + 1.expr(), values);
    }

    pub(crate) fn store_with_key(
        &self,
        cb: &mut ConstraintBuilder<F>,
        key: Expression<F>,
        values: &[Expression<F>],
    ) {
        // Insert the key in the front
        let mut key_and_values = values.to_vec();
        key_and_values.insert(0, key);
        cb.lookup_table("memory store", self.tag(), key_and_values.clone());
    }

    pub(crate) fn witness_store(&mut self, offset: usize, values: &[F]) {
        self.stored_values.push(values.to_vec());
        self.store_offsets.push(offset);
    }

    pub(crate) fn witness_load(&self, offset: usize) -> Vec<F> {
        self.stored_values[self.stored_values.len() - 1 - offset].clone()
    }

    pub(crate) fn clear_witness_data(&mut self) {
        self.store_offsets.clear();
    }

    pub(crate) fn generate_constraints(
        &self,
        cb: &mut ConstraintBuilder<F>,
        is_first_row: Expression<F>,
    ) {
        // table = [sel, lrc_a, lrc_b, ...] 纵向压缩
        let lookup_table = cb.get_lookup_table(self.tag());
        crate::circuit!([meta, cb], {
            ifx! {is_first_row => {
                // 一个箭头左到右就是 // cb.require_equal 
                require!(self.cur.expr() => 0); 
            }}
            // next == cur + 如果cond=F就为0即跳过，否则为1
            require!(self.tag(), self.next => self.cur.expr() + lookup_table.0);
        });
    }

    pub(crate) fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        height: usize,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "memory bank",
            |mut region| {
                // Pad to the full circuit (necessary for reads)
                let mut store_offsets = self.store_offsets.clone();
                store_offsets.push(height);
                
                // 如 store_offsets = [3,4,1]
                // assign 出 self.column 为 = [0,0,0,0, 1,1,1,1,1, 2,2]
                let mut store_index = 0;
                let mut offset = 0;
                for &stored_offset in store_offsets.iter() {
                    while offset <= stored_offset {
                        region.assign_advice(
                            || "assign memory index".to_string(),
                            self.column,
                            offset,
                            || Value::known(F::from(store_index as u64)),
                        )?;
                        offset += 1;
                    }
                    store_index += 1;
                }
                Ok(())
            },
        )
    }

    pub(crate) fn tag(&self) -> String {
        self.tag.clone()
    }
}
