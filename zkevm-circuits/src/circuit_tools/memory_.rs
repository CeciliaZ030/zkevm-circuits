//! Memory
use crate::{util::{query_expression, Expr}, evm_circuit::util::rlc};
use eth_types::{Field, Hash};
use halo2_proofs::{
    circuit::Value,
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Phase, FirstPhase, SecondPhase, ThirdPhase},
    poly::Rotation,
};
use itertools::Itertools;
use std::{ops::{Index, IndexMut}, collections::HashMap};

use super::{
    cached_region::CachedRegion, cell_manager::{CellType, Cell}, constraint_builder::{ConstraintBuilder, DynamicData},
};

#[derive(Clone, Debug, Default)]
pub(crate) struct Memory<F, C> {
    height: usize,
    banks: Vec<MemoryBank<F, C>>,
    rw_records: HashMap<C, (Column<Advice>, Column<Advice>)>
}

impl<F: Field, C: CellType> Memory<F, C> {
    pub(crate) fn new(meta: &mut ConstraintSystem<F>, tags: Vec<(C, usize)>, offset: usize, height: usize) -> Self {
        let mut rw_records = HashMap::new();
        let banks = tags.iter()
            .map(|(tag, phase)| {
                let [key, reads, writes] = match phase {
                    1 => [(); 3].map(|_| meta.advice_column_in(FirstPhase)),
                    2 => [(); 3].map(|_| meta.advice_column_in(SecondPhase)),
                    3 => [(); 3].map(|_| meta.advice_column_in(ThirdPhase)),
                    _ => unreachable!(),
                };
                rw_records.insert(tag.clone(), (reads, writes));
                MemoryBank::new(meta, tag.clone(), height, offset, key, reads, writes)
            })
            .collect::<Vec<MemoryBank<F, C>>>();
        Self {
            banks,
            height,
            rw_records,
        }
    }


    pub(crate) fn get_bank(&self, tag: C) -> &MemoryBank<F, C> {
        for bank in self.banks.iter() {
            if bank.tag() == tag {
                return bank;
            }
        }
        unreachable!()
    }

    pub(crate) fn get_mut_bank(&mut self, tag: C) -> &mut MemoryBank<F, C> {
        for bank in self.banks.iter_mut() {
            if bank.tag() == tag {
                return bank;
            }
        }
        unreachable!()
    }

    pub(crate) fn get_records(&self) -> Vec<(Column<Advice>, Column<Advice>)> {
        self.rw_records.clone().into_values().collect()
    }

    pub(crate) fn build_constraints(
        &self,
        cb: &mut ConstraintBuilder<F, C>,
        is_first_row: Expression<F>,
    ) {
        for bank in self.banks.iter() {
            bank.build_constraints(cb, is_first_row.expr());
        }
    }

    pub(crate) fn build_lookups(&self, meta: &mut ConstraintSystem<F>,) {
        for (cell_type, (reads, writes)) in &self.rw_records {
            let name = format!("{:?}", cell_type);
            meta.lookup_any(Box::leak(name.into_boxed_str()), |meta| {
                vec![(
                    meta.query_advice(*reads, Rotation(0)), 
                    meta.query_advice(*writes, Rotation(0)),
                )]
            });
        }
    }

    pub(crate) fn clear_witness_data(&mut self) {
        for bank in self.banks.iter_mut() {
            bank.clear_witness_data();
        }
    }

    pub(crate) fn assign(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        height: usize,
    ) -> Result<(), Error> {
        for bank in self.banks.iter() {
            bank.assign(region, height)?;
        }
        Ok(())
    }

    pub(crate) fn tags(&self) -> Vec<C> {
        self.banks.iter().map(|bank| bank.tag()).collect()
    }
}

impl<F: Field, C: CellType> Index<C> for Memory<F, C> {
    type Output = MemoryBank<F, C>;

    fn index(&self, tag: C) -> &Self::Output {
        for bank in self.banks.iter() {
            if bank.tag() == tag {
                return bank;
            }
        }
        unreachable!()
    }
}

impl<F: Field, C: CellType> IndexMut<C> for Memory<F, C> {
    fn index_mut(&mut self, tag: C) -> &mut Self::Output {
        for bank in self.banks.iter_mut() {
            if bank.tag() == tag {
                return bank;
            }
        }
        unreachable!()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryBank<F, C> {
    tag: C,
    key: Column<Advice>,
    reads: (Vec<Cell<F>>, usize),
    writes: (Vec<Cell<F>>, usize),
    cur: Expression<F>,
    next: Expression<F>,
    table_conditions: Vec<(usize, Expression<F>)>,
    store_offsets: Vec<usize>,
    stored_values: Vec<Vec<F>>,
}

impl<F: Field, C: CellType> MemoryBank<F, C> {
    pub(crate) fn new(
        meta: &mut ConstraintSystem<F>,
        tag: C,
        height: usize,
        offset: usize,
        key: Column<Advice>,
        read_col: Column<Advice>,
        write_col: Column<Advice>,
    ) -> Self {
        let mut cur = 0.expr();
        let mut next = 0.expr();
        query_expression(meta, |meta| {
            cur = meta.query_advice(key, Rotation::cur());
            next = meta.query_advice(key, Rotation::next());
        });
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for h in 0..height {
            query_expression(meta, |meta| {
                reads.push(Cell::new(meta, read_col, offset + h));
                writes.push(Cell::new(meta, write_col, offset + h));
            });
        }
        Self {
            tag,
            key,
            reads: (reads, 0),
            writes: (writes, 0),
            cur,
            next,
            table_conditions: Vec::new(),
            store_offsets: Vec::new(),
            stored_values: Vec::new(),
        }
    }

    pub(crate) fn key(&self) -> Expression<F> {
        self.cur.expr()
    }

    fn query_write(&mut self) -> Cell<F> {
        let cell = self.writes.0[self.writes.1].clone();
        self.writes.1 += 1;
        cell
    }

    fn query_read(&mut self) -> Cell<F> {
        let cell = self.reads.0[self.reads.1].clone();
        self.reads.1 += 1;
        cell
    }


    pub(crate) fn store(
        &mut self,
        cb: &mut ConstraintBuilder<F, C>,
        values: &[Expression<F>],
    ) -> Expression<F> {
        let key = self.key() + 1.expr();
        let condition = cb.get_condition_expr();
        let values = self.prepend_key(key.clone(), values)
            .iter()
            .map(|value| condition.expr() * value.expr())
            .collect_vec();
        let compressed_expr = cb.split_expression(
            "compression",
            rlc::expr(&values, cb.lookup_challenge.clone().unwrap().expr()),
            None,
        );
        let name = format!("{:?} write #{:?}", self.tag, self.writes.1);
        // cb.add_constraint(
        //     Box::leak(name.clone().into_boxed_str()), 
        //     self.query_write().expr() - compressed_expr
        // );
        cb.store_expression(name.as_str(), compressed_expr.expr(), C::default(), Some(self.query_write()));
        // let lookup = DynamicData {
        //     description: Box::leak(name.to_string().into_boxed_str()),
        //     condition,
        //     values,
        //     region_id: cb.region_id,
        //     is_fixed: true,
        //     compress: true,
        // };
        self.table_conditions.push((cb.region_id, condition));
        key
    }

    pub(crate) fn load(
        &mut self,
        description: &'static str,
        cb: &mut ConstraintBuilder<F, C>,
        load_offset: Expression<F>,
        values: &[Expression<F>],
    ) {
        let key = self.key() - load_offset;
        let condition = cb.get_condition_expr();
        let values = self.prepend_key(key, values)
            .iter()
            .map(|value| condition.expr() * value.expr())
            .collect_vec();
        let compressed_expr = cb.split_expression(
            "compression",
            rlc::expr(&values, cb.lookup_challenge.clone().unwrap().expr()),
            None
        );
        let name = format!("{:?} write #{:?}", self.tag, self.writes.1);
        cb.store_expression(name.as_str(), compressed_expr.expr(), C::default(), Some(self.query_read()));

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

    pub(crate) fn build_constraints(
        &self,
        cb: &mut ConstraintBuilder<F, C>,
        is_first_row: Expression<F>,
    ) {
        // let lookups = self.stored_table
        //     .iter()
        //     .filter(|l| l.region_id == cb.region_id)
        //     .collect::<Vec<_>>();
        // let condition = lookups
        //     .iter()
        //     .fold(0.expr(), |acc, l| acc + l.condition.expr());
        let condition = self.table_conditions
        .iter()
        .filter(|tc| tc.0 == cb.region_id)
        .fold(0.expr(), |acc, tc| acc + tc.1.expr());
        crate::circuit!([meta, cb], {
            ifx! {is_first_row => {
                require!(self.cur.expr() => 0);
            }}
            let description = format!("Dynamic lookup table {:?}", self.tag());
            require!(condition => bool);
            // require!(description, self.next => self.cur.expr() + condition.expr());
            // TODO(Brecht): add constraint that makes sure the table value remains the same when
            // not written
            // ifx!(not!(condition) => {
            //     // Allign the allocation of cell data with key
            //     // meaning if not written then current cell should be same as last
            //     // TODO(Cecilia): fix assignment
            //     require!(peek(&self.writes, 0) => peek(&self.writes, 1));
            // });
        });
    }

    pub(crate) fn assign(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        height: usize,
    ) -> Result<(), Error> {
        // Pad to the full circuit (necessary for reads)
        let mut store_offsets = self.store_offsets.clone();
        store_offsets.push(height);

        // TODO(Brecht): partial updates
        let mut offset = 0;
        for (store_index, &stored_offset) in store_offsets.iter().enumerate() {
            while offset <= stored_offset {
                region.assign_advice(
                    || "assign memory index".to_string(),
                    self.key,
                    offset,
                    || Value::known(F::from(store_index as u64)),
                )?;
                offset += 1;
            }
        }
        Ok(())
    }

    pub(crate) fn tag(&self) -> C {
        self.tag
    }

    pub(crate) fn prepend_key<V: Clone>(&self, key: V, values: &[V]) -> Vec<V> {
        [vec![key], values.to_owned()].concat().to_vec()
    }
}

fn peek<F: Field>(rw_stack: &(Vec<Cell<F>>, usize), offset: usize) -> Expression<F> {
    rw_stack.0[rw_stack.1 -1 - offset].expr()
}