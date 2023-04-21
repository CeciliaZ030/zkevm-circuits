//! Cell manager
use crate::{util::Expr};
use crate::evm_circuit::util::CachedRegion;
use eth_types::{Field, Hash};
use halo2_proofs::{
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Column, Error, Expression, VirtualCells},
    poly::Rotation,
};
use lazy_static::__Deref;
use strum::EnumIter;
use std::collections::HashMap;
use std::cmp::{max, Ordering};
use std::{any::Any, collections::BTreeMap};

// Todo(Cecilia): config this number somewhere
pub(crate) const N_BYTE_LOOKUPS: usize = 4;


#[derive(Clone)]
pub(crate) struct DataTransition<F> {
    prev: Expression<F>,
    cur: Expression<F>,
}

impl<F: Field> DataTransition<F> {
    pub(crate) fn new(meta: &mut VirtualCells<F>, column: Column<Advice>) -> DataTransition<F> {
        DataTransition {
            prev: meta.query_advice(column, Rotation::prev()),
            cur: meta.query_advice(column, Rotation::cur()),
        }
    }

    pub(crate) fn new_with_rot(
        meta: &mut VirtualCells<F>,
        column: Column<Advice>,
        rot_prev: i32,
        rot_cur: i32,
    ) -> DataTransition<F> {
        DataTransition {
            prev: meta.query_advice(column, Rotation(rot_prev)),
            cur: meta.query_advice(column, Rotation(rot_cur)),
        }
    }

    pub(crate) fn from(prev: Expression<F>, cur: Expression<F>) -> DataTransition<F> {
        DataTransition { prev, cur }
    }

    pub(crate) fn cur(&self) -> Expression<F> {
        self.cur.clone()
    }

    pub(crate) fn prev(&self) -> Expression<F> {
        self.prev.clone()
    }

    pub(crate) fn delta(&self) -> Expression<F> {
        self.prev() - self.cur()
    }
}

impl<F: Field> Expr<F> for DataTransition<F> {
    fn expr(&self) -> Expression<F> {
        self.cur.clone()
    }
}

/// Trackable object
pub trait Trackable {
    /// To allow downcasting
    fn as_any(&self) -> &dyn Any;

    /// Cloning
    fn clone_box(&self) -> Box<dyn Trackable>;
}

// We can now implement Clone manually by forwarding to clone_box.
impl Clone for Box<dyn Trackable> {
    fn clone(&self) -> Box<dyn Trackable> {
        self.clone_box()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Cell<F> {
    // expression for constraint
    expression: Option<Expression<F>>,
    column: Option<Column<Advice>>,
    // relative position to selector for synthesis
    rotation: usize,
}

impl<F: Field> Trackable for Cell<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Trackable> {
        Box::new(self.clone())
    }
}

impl<F: Field> Cell<F> {
    pub(crate) fn new(meta: &mut VirtualCells<F>, column: Column<Advice>, rotation: usize) -> Self {
        Self {
            expression: Some(meta.query_advice(column, Rotation(rotation as i32))),
            column: Some(column),
            rotation,
        }
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        value: F,
    ) -> Result<AssignedCell<F, F>, Error> {
        region.assign_advice(
            || {
                format!(
                    "Cell column: {:?} and rotation: {}",
                    self.column, self.rotation
                )
            },
            self.column.unwrap(),
            offset + self.rotation,
            || Value::known(value),
        )
    }

    pub(crate) fn assign_cached(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        offset: usize,
        value: F,
    ) -> Result<AssignedCell<F, F>, Error> {
        region.assign_advice(
            || {
                format!(
                    "Cell column: {:?} and rotation: {}",
                    self.column, self.rotation
                )
            },
            self.column.unwrap(),
            offset + self.rotation,
            || Value::known(value),
        )
    }
}

impl<F: Field> Expr<F> for Cell<F> {
    fn expr(&self) -> Expression<F> {
        self.expression.as_ref().unwrap().clone()
    }
}

impl<F: Field> Expr<F> for &Cell<F> {
    fn expr(&self) -> Expression<F> {
        self.expression.as_ref().unwrap().clone()
    }
}

/// CellType
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CellType {
    /// Storage type
    Storage,
    /// Lookup Byte
    LookupByte,
    /// Lookup outter tables
    Lookup(Table), 
}

/// Table being lookuped by cell which stores the RLC of lookup tuples
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, EnumIter)]
pub enum Table {
    /// The cell contains RLC of a Fixed lookup
    Fixed,
    /// The cell contains RLC of a Tx lookup
    Tx,
    /// The cell contains RLC of a Rw lookup
    Rw,
    /// The cell contains RLC of a Bytecode lookup
    Bytecode,
    /// The cell contains RLC of a Block lookup
    Block,
    /// The cell contains RLC of a Byte lookup
    Byte,
    /// The cell contains RLC of a Copy lookup
    Copy,
    /// The cell contains RLC of a Keccak lookup
    Keccak,
}


/// CellColumn
#[derive(Clone, Debug)]
pub struct CellColumn<F> {
    pub(crate) index: usize,
    pub(crate) cell_type: CellType,
    pub(crate) height: usize,
    pub(crate) expr: Expression<F>,
}

impl<F: Field> PartialEq for CellColumn<F> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.cell_type == other.cell_type && self.height == other.height 
    }
}

impl<F: Field> Eq for CellColumn<F> {}

impl<F: Field> PartialOrd for CellColumn<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.height.partial_cmp(&other.height)
    }
}

impl<F: Field> Ord for CellColumn<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.height.cmp(&other.height)
    }
}

impl<F: Field> Expr<F> for CellColumn<F> {
    fn expr(&self) -> Expression<F> {
        self.expr.clone()
    }
}

/// CellManager with context
#[derive(Clone, Debug)]
pub struct CellManager_<F> {
    width: usize,
    height: usize,
    // current ctx
    cells: Vec<Cell<F>>,
    columns: Vec<CellColumn<F>>,
    // branch ctxs
    branch_ctxs: HashMap<String, CmContext<F>>,
    parent_ctx: Option<CmContext<F>>,
}


#[derive(Default, Clone, Debug)]
struct CmContext<F>{
    parent: Box<Option<CmContext<F>>>,
    columns: Vec<CellColumn<F>>,
}

impl<F: Field>  CellManager_<F> {
    
    pub(crate) fn cur_to_parent(&mut self) {
        let new_parent = match self.parent_ctx.clone() {
            // if parent context exists, meaning we are deep in a callstack
            // we set it as the parent of new parent
            Some(ctx) => CmContext {
                parent: Box::new(Some(ctx.clone())),
                columns: self.columns.clone(),
            },
            // otherwise, this is the fist level of callstack
            // the parent of new parent is None
            None => CmContext {
                parent: Box::new(None),
                columns: self.columns.clone(),
            }
        };
        self.parent_ctx = Some(new_parent);
        self.reset();
    }

    pub(crate) fn cur_to_branch(&mut self, name: &str) {
        let new_branch = match self.parent_ctx.clone() {
            // if parent context exists, meaning we are deep in a callstack
            // we set it as the parent of new branch
            Some(ctx) => CmContext {
                parent: Box::new(Some(ctx.clone())),
                columns: self.columns.clone(),
            },
            // otherwise, this is the fist level of callstack
            // the parent of new branch is None
            None => CmContext {
                parent: Box::new(None),
                columns: self.columns.clone(),
            }
        };
        self.branch_ctxs.insert(name.to_string(), new_branch);
        self.reset();
    }

    pub(crate) fn recover_parent(&mut self) {
        assert!(self.parent_ctx.is_some(), "No parent context to recover");
        self.columns = self.parent_ctx.clone().unwrap().columns.clone();
        self.parent_ctx
            .clone()
            .map(|ctx| self.parent_ctx = ctx.parent.deref().clone())
            .unwrap();
        self.branch_ctxs.clear();
    }

    pub(crate) fn recover_branch(&mut self, name: &str) {
        self.branch_ctxs.get(name).map(|ctx| {
            assert!(ctx.parent.is_some(), "Cannot have sibling without parent");
            self.columns = ctx.columns.clone();
        }).expect("CellManager has no specified context.");
        self.branch_ctxs.remove(name);
    }

    pub(crate) fn recover_max_branch(&mut self) {
        let mut new_cols = self.columns.clone();
        let parent = self.parent_ctx.clone().expect("Retruning context needs parent");
        self.branch_ctxs
            .iter()
            .map(|(name, ctx)| {
                for c in 0..self.width {
                    new_cols[c] = max(&new_cols[c], &ctx.columns[c]).clone();
                    new_cols[c] = max(&new_cols[c], &parent.columns[c]).clone();
                }
            });
        self.columns = new_cols;
        self.branch_ctxs.clear();
        self.parent_ctx = self.parent_ctx
            .clone()
            .map(|ctx| ctx.parent.deref().clone())
            .unwrap();
    }

    pub(crate) fn new(meta: &mut VirtualCells<F>, advice_columns: &[Column<Advice>]) -> Self {
        // Setup the columns and query the cells
        let width = advice_columns.len();
        let height = 32;
        let mut cells = Vec::with_capacity(height * width);
        let mut columns = Vec::with_capacity(width);
        for c in 0..width {
            for r in 0..height {
                cells.push(Cell::new(meta, advice_columns[c], r));
            }
            columns.push(CellColumn {
                index: c,
                cell_type: CellType::Storage,
                height: 0,
                expr: cells[c * height].expr(),
            });
        }
        let mut column_idx = 0;

    
        for i in 0usize..N_BYTE_LOOKUPS {
            columns[i].cell_type = CellType::LookupByte;
            assert_eq!(advice_columns[column_idx].column_type().phase(), 0);
            column_idx += 1;
        }


        Self {
            width,
            height,
            cells,
            columns,
            branch_ctxs: HashMap::new(),
            parent_ctx: None,
        }
    }

    pub(crate) fn query_cells(&mut self, cell_type: CellType, count: usize) -> Vec<Cell<F>> {
        let mut cells = Vec::with_capacity(count);
        while cells.len() < count {
            let column_idx = self.next_column(cell_type);
            let column = &mut self.columns[column_idx];
            cells.push(self.cells[column_idx * self.height + column.height].clone());
            column.height += 1;
        }
        cells
    }

    pub(crate) fn query_cell(&mut self, cell_type: CellType) -> Cell<F> {
        self.query_cells(cell_type, 1)[0].clone()
    }


    pub(crate) fn reset(&mut self) {
        for column in self.columns.iter_mut() {
            column.height = 0;
        }
    }

    fn next_column(&self, cell_type: CellType) -> usize {
        let mut best_index: Option<usize> = None;
        let mut best_height = self.height;
        for column in self.columns.iter() {
            // if cell_type == CellType::LookupByte {
            //     println!("column.cell_type: {:?}, column.index: {:?}, cell_type: {:?}", column.cell_type, column.index, cell_type);
            // }
            if column.cell_type == cell_type && column.height < best_height {
                best_index = Some(column.index);
                best_height = column.height;
            }
        }
        match best_index {
            Some(index) => index,
            None => unreachable!("not enough cells for query: {:?}", cell_type),
        }
    }

    pub(crate) fn get_height(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.height)
            .max()
            .unwrap()
    }

    /// Returns a map of CellType -> (width, height, num_cells)
    pub(crate) fn get_stats(&self) -> BTreeMap<CellType, (usize, usize, usize)> {
        let mut data = BTreeMap::new();
        for column in self.columns.iter() {
            let (mut count, mut height, mut num_cells) =
                data.get(&column.cell_type).unwrap_or(&(0, 0, 0));
            count += 1;
            height = height.max(column.height);
            num_cells += column.height;
            data.insert(column.cell_type, (count, height, num_cells));
        }
        data
    }

    pub(crate) fn columns(&self) -> &[CellColumn<F>] {
        &self.columns
    }
}

/// CellManager
#[derive(Clone, Debug)]
pub struct _CellManager<F> {
    width: usize,
    height: usize,
    cells: Vec<Cell<F>>,
    columns: Vec<CellColumn<F>>,
}

impl<F: Field> _CellManager<F> {
    pub(crate) fn new(meta: &mut VirtualCells<F>, advice_columns: &[Column<Advice>]) -> Self {
        // Setup the columns and query the cells
        let width = advice_columns.len();
        let height = 32;
        let mut cells = Vec::with_capacity(height * width);
        let mut columns = Vec::with_capacity(width);
        for c in 0..width {
            for r in 0..height {
                cells.push(Cell::new(meta, advice_columns[c], r));
            }
            columns.push(CellColumn {
                index: c,
                cell_type: CellType::Storage,
                height: 0,
                expr: cells[c * height].expr(),
            });
        }
        let mut column_idx = 0;

    
        for i in 0usize..N_BYTE_LOOKUPS {
            columns[i].cell_type = CellType::LookupByte;
            assert_eq!(advice_columns[column_idx].column_type().phase(), 0);
            column_idx += 1;
        }


        Self {
            width,
            height,
            cells,
            columns,
        }
    }

    pub(crate) fn query_cells(&mut self, cell_type: CellType, count: usize) -> Vec<Cell<F>> {
        let mut cells = Vec::with_capacity(count);
        while cells.len() < count {
            let column_idx = self.next_column(cell_type);
            let column = &mut self.columns[column_idx];
            cells.push(self.cells[column_idx * self.height + column.height].clone());
            column.height += 1;
        }
        cells
    }

    pub(crate) fn query_cell(&mut self, cell_type: CellType) -> Cell<F> {
        self.query_cells(cell_type, 1)[0].clone()
    }


    pub(crate) fn reset(&mut self) {
        for column in self.columns.iter_mut() {
            column.height = 0;
        }
    }

    fn next_column(&self, cell_type: CellType) -> usize {
        let mut best_index: Option<usize> = None;
        let mut best_height = self.height;
        for column in self.columns.iter() {
            if cell_type == CellType::LookupByte {
                println!("column.cell_type: {:?}, column.index: {:?}, cell_type: {:?}", column.cell_type, column.index, cell_type);
            }
            if column.cell_type == cell_type && column.height < best_height {
                best_index = Some(column.index);
                best_height = column.height;
            }
        }
        match best_index {
            Some(index) => index,
            None => unreachable!("not enough cells for query: {:?}", cell_type),
        }
    }

    pub(crate) fn get_height(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.height)
            .max()
            .unwrap()
    }

    /// Returns a map of CellType -> (width, height, num_cells)
    pub(crate) fn get_stats(&self) -> BTreeMap<CellType, (usize, usize, usize)> {
        let mut data = BTreeMap::new();
        for column in self.columns.iter() {
            let (mut count, mut height, mut num_cells) =
                data.get(&column.cell_type).unwrap_or(&(0, 0, 0));
            count += 1;
            height = height.max(column.height);
            num_cells += column.height;
            data.insert(column.cell_type, (count, height, num_cells));
        }
        data
    }

    pub(crate) fn columns(&self) -> &[CellColumn<F>] {
        &self.columns
    }
}
