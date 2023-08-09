//! Memory
use crate::{
    evm_circuit::util::{rlc},
    util::{query_expression, Expr},
};
use eth_types::{Field, Hash};
use halo2_proofs::{
    circuit::Value,
    plonk::{
        Advice, Column, ConstraintSystem, Error, Expression, FirstPhase, SecondPhase, ThirdPhase,
    },
    poly::Rotation, dev::metadata::VirtualCell,
};
use itertools::Itertools;
use std::{
    collections::HashMap,
    ops::{Index, IndexMut}, marker::PhantomData,
};

use super::{
    cached_region::CachedRegion,
    cell_manager::{Cell, CellType, CellConfig, CellManager},
    constraint_builder::ConstraintBuilder,
};

#[derive(Clone, Debug, Default)]
pub(crate) struct Memory<F: Field, C: CellType, MB: MemoryBank<F, C>> {
    // TODO(Cecilia): want to use dynamic dispatch
    // i.e. dyn MemoryBank<F, C> but trait with generic param is not object safe
    banks: HashMap<C, MB>,
    _phantom: PhantomData<F>
}

impl<F: Field, C: CellType, MB: MemoryBank<F, C>> Memory<F, C, MB> {
    pub(crate) fn new(
        cb: &mut ConstraintBuilder<F, C>,
        meta: &mut ConstraintSystem<F>,
        tags: Vec<(C, u8)>,
        offset: usize,
    ) -> Self {
        let mut banks = HashMap::new();
        tags
            .into_iter()
            .for_each(|(tag, phase)| {
                banks.insert(
                    tag, 
                    MB::new(meta, cb, tag, phase, offset)
                );
            });
        Self { 
            banks,
            _phantom: PhantomData
        }
    }

    pub(crate) fn get_bank(&self, tag: C) -> &MB {
        self.banks.get(&tag).unwrap()
    }

    pub(crate) fn get_mut_bank(&mut self, tag: C) -> &mut MB {
        self.banks.get_mut(&tag).unwrap()
    }

    pub(crate) fn get_columns(&self) -> Vec<Column<Advice>> {
        self.banks
        .values()
        .fold( Vec::new(),|mut acc, bank| {
            acc.extend(bank.columns().iter());
            acc
        })
    }

    pub(crate) fn build_constraints(
        &self,
        cb: &mut ConstraintBuilder<F, C>,
        q_start: Expression<F>,
    ) {
        for (_, bank) in self.banks.iter() {
            bank.build_constraints(cb, q_start.expr());
        }
    }

    pub(crate) fn assign(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        height: usize,
    ) -> Result<(), Error> {
        for (_, bank) in self.banks.iter() {
            bank.assign(region, height)?;
        }
        Ok(())
    }

    pub(crate) fn tags(&self) -> Vec<C> {
        self.banks.iter().map(|(_, bank)| bank.tag()).collect()
    }
}


pub(crate) trait MemoryBank<F: Field, C: CellType>: Clone {
    fn new(meta: &mut ConstraintSystem<F>, cb: &mut ConstraintBuilder<F, C>, tag: C, phase: u8, offset: usize) -> Self;
    fn store(&mut self, cb: &mut ConstraintBuilder<F, C>, values: &[Expression<F>]) -> Expression<F>;
    fn load(&mut self, cb: &mut ConstraintBuilder<F, C>, load_offset: Expression<F>, values: &[Expression<F>]);
    fn columns(&self) -> Vec<Column<Advice>>;
    fn tag(&self) -> C;
    fn witness_store(&mut self, offset: usize, values: &[F]);
    fn witness_load(&self, offset: usize) -> Vec<F>;
    fn build_constraints(&self, cb: &mut ConstraintBuilder<F, C>, q_start: Expression<F>);
    fn assign(&self, region: &mut CachedRegion<'_, '_, F>, height: usize) -> Result<(), Error>;
}

#[derive(Clone, Debug)]
pub(crate) struct RWBank<F, C> {
    tag: C,
    key: Column<Advice>,
    reads: Column<Advice>,
    writes: Column<Advice>,
    store_offsets: Vec<usize>,
    stored_values: Vec<Vec<F>>,
    cur: Expression<F>,
    next: Expression<F>,
    // TODO(Cecilia): get rid of this when we kill regions
    local_conditions: Vec<(usize, Expression<F>)>,
}

impl<F: Field, C: CellType> RWBank<F, C> {
    pub(crate) fn prepend_key(&self, values: &[Expression<F>]) -> Vec<Expression<F>> {
        [&[self.cur.expr() + 1.expr()], values].concat().to_vec()
    }

    pub(crate) fn prepend_offset(&self, values: &[Expression<F>], offset: Expression<F>) -> Vec<Expression<F>> {
        [&[self.cur.expr() - offset], values].concat().to_vec()
    }
}

impl<F: Field, C: CellType> MemoryBank<F, C> for RWBank<F, C> {
    fn new(
        meta: &mut ConstraintSystem<F>,
        cb: &mut ConstraintBuilder<F, C>, 
        tag: C,
        phase: u8,
        offset: usize,
    ) -> Self {
        let cm = cb.cell_manager.as_mut().unwrap();
        let config = (tag, 2, phase, false);
        cm.add_celltype(meta, config, offset);
        let rw_cols = cm.get_typed_columns(tag);
        let key = meta.advice_column();
        let (cur, next) = query_expression(meta, |meta| {
            (
                meta.query_advice(key, Rotation(0)),
                meta.query_advice(key, Rotation(1))
            )   
        });
        Self { 
            tag, 
            key, 
            reads: rw_cols[0].column, 
            writes: rw_cols[1].column, 
            store_offsets: Vec::new(), 
            stored_values: Vec::new(), 
            cur,
            next,
            local_conditions: Vec::new(),
        }
    }

    fn store(
        &mut self, 
        cb: &mut ConstraintBuilder<F, C>, 
        values: &[Expression<F>]
    ) -> Expression<F> {
        let values = self.prepend_key(values);
        cb.store_table(
            Box::leak(format!("{:?} store", self.tag).into_boxed_str()),
            self.tag, 
            values.clone(), 
            true, 
            true, 
            false
        );
        values[0].expr()
    }

    fn load(
        &mut self, 
        cb: &mut ConstraintBuilder<F, C>, 
        load_offset: Expression<F>, 
        values: &[Expression<F>]
    ) {
        let values = self.prepend_offset(values, load_offset);
        cb.add_lookup(
            Box::leak(format!("{:?} load", self.tag).into_boxed_str()), 
            self.tag, 
            values, 
            false, 
            true, 
            true, 
            true
        );
    }

    fn tag(&self) -> C {
        self.tag
    }

    fn columns(&self) -> Vec<Column<Advice>> {
        vec![self.key, self.reads, self.writes]
    }

    fn build_constraints(
        &self, 
        cb: &mut ConstraintBuilder<F, C>, 
        q_start: Expression<F>
    ) {
        let condition = self
            .local_conditions
            .iter()
            .filter(|tc| tc.0 == cb.region_id)
            .fold(0.expr(), |acc, tc| acc + tc.1.expr());
        crate::circuit!([meta, cb], {
            ifx! {q_start => {
                require!(self.cur.expr() => 0);
            }}
            let description = format!("Dynamic lookup table {:?}", self.tag());
            require!(condition => bool);
            require!(description, self.next => self.cur.expr() + condition.expr());
        });    
    }

    fn witness_store(&mut self, offset: usize, values: &[F]) {
        self.stored_values.push(values.to_vec());
        self.store_offsets.push(offset);
    }

    fn witness_load(&self, offset: usize) -> Vec<F> {
        self.stored_values[self.stored_values.len() - 1 - offset].clone()
    }

    fn assign(
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
}