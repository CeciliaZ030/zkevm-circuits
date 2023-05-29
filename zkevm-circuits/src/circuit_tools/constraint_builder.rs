//! Circuit utilities
use crate::{
    bytecode_circuit::param::HASH_WIDTH,
    evm_circuit::{
        table::Lookup,
        util::{rlc, CachedRegion},
    },
    util::Expr,
};
use eth_types::Field;
use gadgets::util::{and, select, sum, Scalar};
use halo2_proofs::{
    circuit::AssignedCell,
    plonk::{Column, ConstraintSystem, Error, Expression, Fixed},
    poly::Rotation,
};
use itertools::Itertools;

use super::cell_manager::{Cell, CellManager_, CellType, DataTransition, Table, Trackable};

/// Lookup data
#[derive(Clone)]
pub struct LookupData<F> {
    /// Desciption
    pub description: &'static str,
    /// Lookup tag
    pub tag: String,
    /// Condition under which the lookup needs to be done
    pub condition: Expression<F>,
    /// The values to lookup
    pub values: Vec<Expression<F>>,
}

/// Constraint builder
#[derive(Clone)]
pub struct ConstraintBuilder<F> {
    constraints: Vec<(&'static str, Expression<F>)>,
    max_degree: usize,
    conditions: Vec<Expression<F>>,

    /// The lookups
    /// 需要查的数据（descr，tag, condition: Expression<F>, values:
    /// Vec<Expression<F>>） 在memory load_with_key 里面调，load 的
    /// Vec<LookupData<F>> 是 lookup 这个行为
    pub lookups: Vec<LookupData<F>>,
    /// The lookup tables
    /// 在memory store_with_key 里面调，store 的 Vec<LookupData<F>> 是形成这张表
    pub lookup_tables: Vec<LookupData<F>>,

    /// Query offset
    pub query_offset: i32,
    /// CellManager_
    pub cell_manager: Option<CellManager_<F>>,
    /// Tracked objects
    objects: Vec<Box<dyn Trackable>>,

    /// Store expression over max degree
    pub stored_expressions: Vec<StoredExpression<F>>,
    /// Store randomness over max degree
    power_of_randomness: Option<[Expression<F>; HASH_WIDTH]>,
}

/// Stored expression for lookup
#[derive(Debug, Clone)]
pub struct StoredExpression<F> {
    pub(crate) name: String,
    cell: Cell<F>,
    cell_type: CellType,
    expr: Expression<F>,
    expr_id: String,
}

impl<F: Field> StoredExpression<F> {
    /// Assign cell with lookup expression
    pub fn assign(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        offset: usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        let value = self.expr.evaluate(
            &|scalar| scalar,
            &|_| unimplemented!("selector column"),
            &|fixed_query| {
                region.get_fixed(offset, fixed_query.column_index(), fixed_query.rotation())
            },
            &|advide_query| {
                region.get_advice(offset, advide_query.column_index(), advide_query.rotation())
            },
            &|instance_query| {
                region.get_instance(
                    offset,
                    instance_query.column_index(),
                    instance_query.rotation(),
                )
            },
            &|_| unimplemented!(),
            &|a| -a,
            &|a, b| a + b,
            &|a, b| a * b,
            &|a, scalar| a * scalar,
        );
        self.cell.assign_cached(region, offset, value)
    }
}

impl<F: Field> ConstraintBuilder<F> {
    pub(crate) fn new(max_degree: usize, cell_manager: Option<CellManager_<F>>) -> Self {
        ConstraintBuilder {
            constraints: Vec::new(),
            max_degree,
            conditions: Vec::new(),
            lookups: Vec::new(),
            lookup_tables: Vec::new(),
            query_offset: 0,
            cell_manager,
            objects: Vec::new(),

            stored_expressions: Vec::new(),
            power_of_randomness: None,
        }
    }

    pub(crate) fn set_cell_manager(&mut self, cell_manager: CellManager_<F>) {
        println!("set_cell_manager");
        self.cell_manager = Some(cell_manager);
    }

    pub(crate) fn set_power_of_randomness(
        &mut self,
        power_of_randomness: [Expression<F>; HASH_WIDTH],
    ) {
        self.power_of_randomness = Some(power_of_randomness);
    }

    pub(crate) fn enter_branch_context(&mut self) {
        println!("=====>");
        match self.cell_manager.as_mut() {
            Some(cm) => cm.cur_to_parent(),
            None => (),
        };
    }

    pub(crate) fn switch_branch_context(&mut self, branch_name: &str) {
        match self.cell_manager.as_mut() {
            Some(cm) => cm.cur_to_branch(branch_name),
            None => (),
        };
    }

    pub(crate) fn exit_branch_context(&mut self) {
        println!("^*****");
        match self.cell_manager.as_mut() {
            Some(cm) => cm.recover_max_branch(),
            None => (),
        };
    }

    pub(crate) fn require_zero(&mut self, name: &'static str, constraint: Expression<F>) {
        self.add_constraint(name, constraint);
    }

    pub(crate) fn require_equal(
        &mut self,
        name: &'static str,
        lhs: Expression<F>,
        rhs: Expression<F>,
    ) {
        self.add_constraint(name, lhs - rhs);
    }

    pub(crate) fn require_boolean(&mut self, name: &'static str, value: Expression<F>) {
        self.add_constraint(name, value.clone() * (1.expr() - value));
    }

    pub(crate) fn require_in_set(
        &mut self,
        name: &'static str,
        value: Expression<F>,
        set: Vec<Expression<F>>,
    ) {
        self.add_constraint(
            name,
            set.iter()
                .fold(1.expr(), |acc, item| acc * (value.clone() - item.clone())),
        );
    }

    pub(crate) fn condition<R>(
        &mut self,
        condition: Expression<F>,
        constraint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.push_condition(condition);
        let ret = constraint(self);
        self.pop_condition();
        ret
    }

    pub(crate) fn push_condition(&mut self, condition: Expression<F>) {
        self.conditions.push(condition);
    }

    pub(crate) fn pop_condition(&mut self) {
        self.conditions.pop();
    }

    pub(crate) fn add_constraints(&mut self, constraints: Vec<(&'static str, Expression<F>)>) {
        for (name, constraint) in constraints {
            self.add_constraint(name, constraint);
        }
    }

    pub(crate) fn add_constraint(&mut self, name: &'static str, constraint: Expression<F>) {
        let constraint = match self.get_condition() {
            Some(condition) => condition * constraint,
            None => constraint,
        };
        self.validate_degree(constraint.degree(), name);
        self.constraints.push((name, constraint));
    }

    // Query

    pub(crate) fn query_bool(&mut self) -> Cell<F> {
        let cell = self.query_cell();
        self.require_boolean("Constrain cell to be a bool", cell.expr());
        cell
    }

    pub(crate) fn query_byte(&mut self) -> Cell<F> {
        self.query_cell_with_type(CellType::LookupByte)
    }

    pub(crate) fn query_bytes<const N: usize>(&mut self) -> [Cell<F>; N] {
        self.query_bytes_dyn(N).try_into().unwrap()
    }

    pub(crate) fn query_bytes_dyn(&mut self, count: usize) -> Vec<Cell<F>> {
        self.query_cells(CellType::LookupByte, count)
    }

    pub(crate) fn query_cell(&mut self) -> Cell<F> {
        self.query_cell_with_type(CellType::Storage)
    }

    pub(crate) fn query_cell_with_type(&mut self, cell_type: CellType) -> Cell<F> {
        self.query_cells(cell_type, 1).first().unwrap().clone()
    }

    fn query_cells(&mut self, cell_type: CellType, count: usize) -> Vec<Cell<F>> {
        self.cell_manager
            .as_mut()
            .unwrap()
            .query_cells(cell_type, count)
    }

    pub(crate) fn validate_degree(&self, degree: usize, name: &'static str) {
        if self.max_degree > 0 {
            debug_assert!(
                degree <= self.max_degree,
                "Expression {} degree too high: {} > {}",
                name,
                degree,
                self.max_degree,
            );
        }
    }

    pub(crate) fn generate_constraints(&self) -> Vec<(&'static str, Expression<F>)> {
        self.constraints.clone()
    }

    pub(crate) fn generate_lookups<S: AsRef<str>>(
        &mut self,
        meta: &mut ConstraintSystem<F>,
        byte_table: [Column<Fixed>; 1],
        lookup_names: &[S],
        // "keccek", "fixed", "parent_s" ,"parent_c","parent_s", "key_s", "key_c"
    ) {
        if let Some(cm) = self.cell_manager.clone() {
            for column in cm.columns() {
                match column.cell_type {
                    CellType::Storage => (),
                    CellType::LookupByte => {
                        meta.lookup_any("Byte lookup", |meta| {
                            let byte_table_expression =
                                meta.query_fixed(byte_table[0], Rotation::cur());
                            vec![(column.expr(), byte_table_expression)]
                        });
                    }
                    CellType::Lookup(..) => (),
                }
            }
        }

        // 表有：[ "parent_s" ,"parent_c","parent_s", "key_s", "key_c"]
        for lookup_name in lookup_names.iter() {
            // 拿对应的表 【sel*rlc_a, sel*rlc_b, ...】
            let table = self.get_lookup_table_values(lookup_name);
            // 把当前的表所对应的lookup data拿出来
            let lookups = self
                .lookups
                .iter()
                .cloned()
                .filter(|lookup| lookup.tag == lookup_name.as_ref())
                .collect::<Vec<_>>();
            // 比如 “parent_s" 有十个lookup data，逐个轮训
            for lookup in lookups.iter() {
                // 终于到了重点👇，调Halo2的lookup API
                meta.lookup_any(lookup.description, |_meta| {
                    // 拿要查的值
                    let mut values: Vec<_> = lookup
                        .values
                        .iter()
                        .map(|value| lookup.condition.expr() * value.expr())
                        .collect();
                    // 如果设计的Table 有四列 = (A,B,C,D)，而 Lookup = (A,B)，保证后者短于前者
                    assert!(table.len() >= values.len());
                    // Lookup 补零 => (A,B,0,0)，因为去找的这部分 Table 一定只用了前两列
                    while values.len() < table.len() {
                        values.push(0.expr());
                    }
                    // 形成符合Halo2的Tuples，[(value0, table0),(value1, table1),...]
                    table
                        .iter()
                        .zip(values.iter())
                        .map(|(table, value)| (value.expr(), table.expr()))
                        .collect()
                });
            }
        }
    }

    pub(crate) fn get_condition(&self) -> Option<Expression<F>> {
        if self.conditions.is_empty() {
            None
        } else {
            Some(and::expr(self.conditions.iter()))
        }
    }

    pub(crate) fn get_condition_expr(&self) -> Expression<F> {
        self.get_condition().unwrap_or_else(|| 1.expr())
    }

    pub(crate) fn lookup_table<S: AsRef<str>>(
        &mut self,
        description: &'static str,
        tag: S,
        values: Vec<Expression<F>>,
    ) {
        // 上层：在 memory里面调 cb.lookup_table("memory store", self.tag(),
        // key_and_values); 存：
        let condition = self.get_condition_expr();
        self.lookup_tables.push(LookupData {
            description,
            tag: tag.as_ref().to_owned(),
            condition,
            values,
        });
    }

    pub(crate) fn add_lookup_rlc(
        &mut self,
        name: &str,
        lookup: Vec<Expression<F>>,
        table_type: Table,
    ) {
        if let Some(power_of_randomness) = self.power_of_randomness.as_ref() {
            let lookup = match self.get_condition() {
                Some(condition) => lookup
                    .into_iter()
                    .map(|expr| condition.clone() * expr)
                    .collect(),
                None => lookup,
            };
            let compressed_expr = self.split_expression(
                "Lookup compression",
                rlc::expr(&lookup, power_of_randomness),
                self.max_degree,
            );
            self.store_expression(name, compressed_expr, CellType::Lookup(table_type));
        } else {
            panic!("Can's lookup with rlc without randomness");
        }
    }

    pub(crate) fn store_expression(
        &mut self,
        name: &str,
        expr: Expression<F>,
        cell_type: CellType,
    ) -> Expression<F> {
        // Check if we already stored the expression somewhere
        let stored_expression = self.find_stored_expression(expr.clone(), cell_type);
        match stored_expression {
            Some(stored_expression) => {
                debug_assert!(
                    !matches!(cell_type, CellType::Lookup(_)),
                    "The same lookup is done multiple times",
                );
                stored_expression.cell.expr()
            }
            None => {
                // Even if we're building expressions for the next step,
                // these intermediate values need to be stored in the current step.
                // let in_next_step = self.in_next_step;
                // self.in_next_step = false;
                let cell = self.query_cell_with_type(cell_type);
                // self.in_next_step = in_next_step;

                // Require the stored value to equal the value of the expression
                let name = format!("{} (stored expression)", name);
                self.add_constraint(
                    Box::leak(name.clone().into_boxed_str()),
                    cell.expr() - expr.clone(),
                );

                self.stored_expressions.push(StoredExpression {
                    name,
                    cell: cell.clone(),
                    cell_type,
                    expr_id: expr.identifier(),
                    expr,
                });
                cell.expr()
            }
        }
    }

    pub(crate) fn find_stored_expression(
        &self,
        expr: Expression<F>,
        cell_type: CellType,
    ) -> Option<&StoredExpression<F>> {
        let expr_id = expr.identifier();
        self.stored_expressions
            .iter()
            .find(|&e| e.cell_type == cell_type && e.expr_id == expr_id)
    }

    fn split_expression(
        &mut self,
        name: &'static str,
        expr: Expression<F>,
        max_degree: usize,
    ) -> Expression<F> {
        if expr.degree() > max_degree {
            match expr {
                Expression::Negated(poly) => {
                    Expression::Negated(Box::new(self.split_expression(name, *poly, max_degree)))
                }
                Expression::Scaled(poly, v) => {
                    Expression::Scaled(Box::new(self.split_expression(name, *poly, max_degree)), v)
                }
                Expression::Sum(a, b) => {
                    let a = self.split_expression(name, *a, max_degree);
                    let b = self.split_expression(name, *b, max_degree);
                    a + b
                }
                Expression::Product(a, b) => {
                    let (mut a, mut b) = (*a, *b);
                    while a.degree() + b.degree() > max_degree {
                        let mut split = |expr: Expression<F>| {
                            if expr.degree() > max_degree {
                                self.split_expression(name, expr, max_degree)
                            } else {
                                self.store_expression(name, expr, CellType::Storage)
                            }
                        };
                        if a.degree() >= b.degree() {
                            a = split(a);
                        } else {
                            b = split(b);
                        }
                    }
                    a * b
                }
                _ => expr.clone(),
            }
        } else {
            expr.clone()
        }
    }

    pub(crate) fn lookup<S: AsRef<str>>(
        &mut self,
        description: &'static str,
        tag: S,
        values: Vec<Expression<F>>,
    ) {
        // 在memory load_with_key 里面调
        // cb.lookup(description, self.tag(), key_and_values);
        let condition = self.get_condition_expr();
        self.lookups.push(LookupData {
            description,
            tag: tag.as_ref().to_owned(),
            condition,
            values,
        });
    }

    pub(crate) fn get_lookups<S: AsRef<str>>(&self, tags: &[S]) -> Vec<LookupData<F>> {
        self.lookups
            .iter()
            .cloned()
            .filter(|lookup| tags.iter().any(|tag| lookup.tag == tag.as_ref()))
            .collect::<Vec<_>>()
    }

    pub(crate) fn consume_lookups<S: AsRef<str>>(&mut self, tags: &[S]) -> Vec<LookupData<F>> {
        // 拿出需要的
        let lookups = self.get_lookups(tags);
        // 把需要的从所有 loolups 中删掉
        self.lookups
            .retain(|lookup| tags.iter().any(|tag| lookup.tag != tag.as_ref()));
        lookups
    }

    ///
    pub(crate) fn get_lookup_table<S: AsRef<str>>(
        &self,
        tag: S,
    ) -> (Expression<F>, Vec<Expression<F>>) {
        let lookups: Vec<&LookupData<F>> = self
            .lookup_tables
            .iter()
            .filter(|lookup| lookup.tag == tag.as_ref())
            .collect::<Vec<_>>();
        // table = [sel, lrc_a, lrc_b, ...] 纵向压缩
        merge_values_unsafe(
            lookups
                .iter()
                .map(|lookup| (lookup.condition.clone(), lookup.values.clone()))
                .collect::<Vec<_>>(),
        )
    }

    ///
    pub(crate) fn get_lookup_table_values<S: AsRef<str>>(&self, tag: S) -> Vec<Expression<F>> {
        // store_with_key 储存了零散的 lookup data 去形成 table
        // 如 (Tx, 1,2,3), (Kecceck, 4), (Tx, 5,6,7), (Block, 8,9)
        // 这里取比如说其中的 (cond -> Tx, 1,2,3)，(cond -> Tx, 5,6,7), ...
        // 纵向形成 rlc     (sel, Tx_rlc, rlc_1, rlc_2, rlc_3)
        let lookup_table = self.get_lookup_table(tag);

        // 每项和sel乘 【sel*Tx_rlc ,sel*rlc_1, sel*rlc_2, sel*rlc_3】
        // Combine with the merged selector as well
        lookup_table
            .1
            .iter()
            .map(|v| v.expr() * lookup_table.0.expr())
            .collect::<Vec<_>>()
    }

    pub(crate) fn generate_lookup_table_checks<S: AsRef<str>>(&mut self, tag: S) {
        // 把要查的 table 拿出来
        let lookups = self
            .lookup_tables
            .iter()
            .filter(|lookup| lookup.tag == tag.as_ref())
            .collect::<Vec<_>>();
        // 把所有sel拿出来
        let selectors = lookups
            .iter()
            .map(|lookup| lookup.condition.expr())
            .collect::<Vec<_>>();
        // 要求每个是 bool
        for selector in selectors.iter() {
            self.require_boolean(
                "lookup table condition needs to be boolean",
                selector.expr(),
            );
        }
        // 要求加起来也是 bool
        let selector = sum::expr(&selectors);
        self.require_boolean(
            "lookup table conditions sum needs to be boolean",
            selector.expr(),
        );
    }

    pub(crate) fn print_stats(&self) {
        let mut expressions = self.constraints.clone();
        expressions.sort_by(|a, b| a.1.degree().cmp(&b.1.degree()));
        for (name, expr) in expressions.iter() {
            println!("'{}': {}", name, expr.degree());
        }
    }

    pub(crate) fn get_query_offset(&self) -> i32 {
        self.query_offset
    }

    pub(crate) fn set_query_offset(&mut self, query_offset: i32) {
        self.query_offset = query_offset;
    }
}

/// Merge lookups that with mutually exclusive conditions
pub(crate) fn merge_lookups<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    lookups: Vec<LookupData<F>>,
) -> (Expression<F>, Vec<Expression<F>>) {
    // table = [sel, lrc_a, lrc_b, ...] 纵向压缩
    merge_values(
        cb,
        lookups
            .iter()
            .map(|lookup| (lookup.condition.clone(), lookup.values.clone()))
            .collect::<Vec<_>>(),
    )
}

/// Merge lookup values that with mutually exclusive conditions
pub(crate) fn merge_values<F: Field>(
    cb: &mut ConstraintBuilder<F>,
    values: Vec<(Expression<F>, Vec<Expression<F>>)>,
) -> (Expression<F>, Vec<Expression<F>>) {
    let selector = sum::expr(values.iter().map(|(condition, _)| condition.expr()));
    // Sanity checks (can be removed, here for safety)
    crate::circuit!([meta, cb], {
        require!(selector => bool);
    });
    // table = [sel, lrc_a, lrc_b, ...]
    merge_values_unsafe(values)
}

pub(crate) fn merge_values_unsafe<F: Field>(
    values: Vec<(Expression<F>, Vec<Expression<F>>)>,
) -> (Expression<F>, Vec<Expression<F>>) {
    if values.is_empty() {
        return (0.expr(), Vec::new());
    }
    // 总 select，如果所有cond都是0就算了
    // selector = 1+0+1+1+0+... = bool
    let selector = sum::expr(values.iter().map(|(condition, _)| condition.expr()));
    // Merge
    // 找表最宽的宽度
    let max_length = values.iter().map(|(_, values)| values.len()).max().unwrap();
    let mut merged_values = vec![0.expr(); max_length];
    let default_value = 0.expr();
    // values = [{cond, (a,b,c)}, {cond, (a,b,c)}, {cond, (a,b,c)}]
    // 对每一列进行合并
    for (idx, value) in merged_values.iter_mut().enumerate() {
        *value = sum::expr(values.iter().map(|(condition, values)| {
            // 逐行的每个val 乘上 cond，
            // cond1*a1 + cond2*a2 + cond3*a3
            condition.expr() * values.get(idx).unwrap_or_else(|| &default_value).expr()
        }));
    }
    // 就是最后每个 table 会纵向压扁成列的 lrc
    // table = [sel, lrc_a, lrc_b, ...]
    (selector, merged_values)
}

pub(crate) fn select<F: Field>(
    condition: Expression<F>,
    when_true: &[Expression<F>],
    when_false: &[Expression<F>],
) -> Vec<Expression<F>> {
    when_true
        .into_iter()
        .zip(when_false.into_iter())
        .map(|(when_true, when_false)| {
            select::expr(condition.expr(), when_true.expr(), when_false.expr())
        })
        .collect()
}

/// Trait that generates a vector of expressions
pub trait Expressable<F> {
    /// Returns a vector of the expressions from itself
    fn to_expr_vec(&self) -> Vec<Expression<F>>;
}

impl<F: Field> Expressable<F> for std::ops::Range<isize> {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        self.clone()
            .map(|e| e.to_expr_vec()[0].expr())
            .collect::<Vec<_>>()
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for Vec<E> {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        self.iter()
            .map(|e| e.to_expr_vec()[0].expr())
            .collect::<Vec<_>>()
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for [E] {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        self.iter()
            .map(|e| e.to_expr_vec()[0].expr())
            .collect::<Vec<_>>()
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for &[E] {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        self.iter()
            .map(|e| e.to_expr_vec()[0].expr())
            .collect::<Vec<_>>()
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for (E, E) {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        let mut res = self.0.to_expr_vec();
        res.append(&mut self.1.to_expr_vec());
        res
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for (E, E, E) {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        let mut res = self.0.to_expr_vec();
        res.append(&mut self.1.to_expr_vec());
        res.append(&mut self.2.to_expr_vec());
        res
    }
}

impl<F: Field, E: Expressable<F>> Expressable<F> for (E, E, E, E) {
    fn to_expr_vec(&self) -> Vec<Expression<F>> {
        let mut res = self.0.to_expr_vec();
        res.append(&mut self.1.to_expr_vec());
        res.append(&mut self.2.to_expr_vec());
        res.append(&mut self.3.to_expr_vec());
        res
    }
}

/// Implementation trait `Expressable` for type able to be casted to an
/// Expression
#[macro_export]
macro_rules! impl_expressable {
    ($type:ty) => {
        impl<F: eth_types::Field> Expressable<F> for $type {
            #[inline]
            fn to_expr_vec(&self) -> Vec<Expression<F>> {
                vec![self.expr()]
            }
        }
    };
}

impl_expressable!(bool);
impl_expressable!(u8);
impl_expressable!(i32);
impl_expressable!(u64);
impl_expressable!(usize);
impl_expressable!(isize);
impl_expressable!(Expression<F>);
impl_expressable!(DataTransition<F>);
impl_expressable!(Cell<F>);

/// Trait around select
pub trait Selectable<F> {
    /// Selects between itself and another value using the given condition
    fn select(&self, condition: Expression<F>, other: &Self) -> Self;
    /// Returns itself if the condition holds, else zero
    fn conditional(&self, condition: Expression<F>) -> Self;
    /// Adds 2 Selectables together
    fn add_expr(&self, other: &Self) -> Self;
    /// Creates a vector of Expressions representing itself
    fn to_vec(&self) -> Vec<Expression<F>>;
}

impl<F: Field> Selectable<F> for () {
    fn select(&self, _condition: Expression<F>, _when_false: &Self) -> Self {
        ()
    }
    fn conditional(&self, _condition: Expression<F>) -> Self {
        ()
    }
    fn add_expr(&self, _other: &Self) -> Self {
        ()
    }
    fn to_vec(&self) -> Vec<Expression<F>> {
        vec![]
    }
}

impl<F: Field> Selectable<F> for Expression<F> {
    fn select(&self, condition: Expression<F>, when_false: &Self) -> Self {
        gadgets::util::select::expr(condition, self.expr(), when_false.expr())
    }
    fn conditional(&self, condition: Expression<F>) -> Self {
        condition * self.expr()
    }
    fn add_expr(&self, other: &Self) -> Self {
        self.expr() + other.expr()
    }
    fn to_vec(&self) -> Vec<Expression<F>> {
        vec![self.expr()]
    }
}

/// Implementation trait `Selectable` for type able to be casted to an
/// expression
#[macro_export]
macro_rules! impl_selectable {
    ($type:ty, $v:expr) => {
        impl<F: eth_types::Field> Selectable<F> for $type {
            fn select(&self, condition: Expression<F>, when_false: &Self) -> Self {
                select(condition, &self.to_vec(), &when_false.to_vec())
                    .into_iter()
                    .collect_tuple()
                    .unwrap()
            }
            fn conditional(&self, condition: Expression<F>) -> Self {
                self.to_vec()
                    .into_iter()
                    .map(|when_true| condition.expr() * when_true.expr())
                    .collect_tuple()
                    .unwrap()
            }
            fn add_expr(&self, other: &Self) -> Self {
                self.to_vec()
                    .iter()
                    .zip(other.to_vec().iter())
                    .map(|(a, b)| a.expr() + b.expr())
                    .collect_tuple()
                    .unwrap()
            }
            fn to_vec(&self) -> Vec<Expression<F>> {
                $v(self)
            }
        }
    };
}

impl_selectable!((Expression<F>, Expression<F>), |t: &(
    Expression<F>,
    Expression<F>
)| {
    vec![t.0.expr(), t.1.expr()]
});
impl_selectable!((Expression<F>, Expression<F>, Expression<F>), |t: &(
    Expression<F>,
    Expression<F>,
    Expression<F>
)| {
    vec![t.0.expr(), t.1.expr(), t.2.expr()]
});
impl_selectable!(
    (Expression<F>, Expression<F>, Expression<F>, Expression<F>),
    |t: &(Expression<F>, Expression<F>, Expression<F>, Expression<F>)| {
        vec![t.0.expr(), t.1.expr(), t.2.expr(), t.3.expr()]
    }
);
impl_selectable!(
    (
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>
    ),
    |t: &(
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>
    )| { vec![t.0.expr(), t.1.expr(), t.2.expr(), t.3.expr(), t.4.expr()] }
);
impl_selectable!(
    (
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    ),
    |t: &(
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    )| {
        vec![
            t.0.expr(),
            t.1.expr(),
            t.2.expr(),
            t.3.expr(),
            t.4.expr(),
            t.5.expr(),
        ]
    }
);
impl_selectable!(
    (
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    ),
    |t: &(
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    )| {
        vec![
            t.0.expr(),
            t.1.expr(),
            t.2.expr(),
            t.3.expr(),
            t.4.expr(),
            t.5.expr(),
            t.6.expr(),
        ]
    }
);
impl_selectable!(
    (
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    ),
    |t: &(
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
        Expression<F>,
    )| {
        vec![
            t.0.expr(),
            t.1.expr(),
            t.2.expr(),
            t.3.expr(),
            t.4.expr(),
            t.5.expr(),
            t.6.expr(),
            t.7.expr(),
        ]
    }
);

/// Trait that conditionally combines multiple types
pub trait Conditionable<F, E> {
    /// Conditionally combines multiple values
    fn apply_conditions(&self) -> E;
    /// Gets the list of all conditions
    fn get_conditions(&self) -> Vec<Expression<F>>;
    /// Gets the sum of all conditions
    fn sum_conditions(&self) -> Expression<F>;
}

impl<F: Field, E: Selectable<F>> Conditionable<F, E> for Vec<(Expression<F>, E)> {
    fn apply_conditions(&self) -> E {
        let mut res = self[0].1.conditional(self[0].0.expr());
        for pair in self.iter().skip(1) {
            res = res.add_expr(&pair.1.conditional(pair.0.expr()));
        }
        res
    }

    fn get_conditions(&self) -> Vec<Expression<F>> {
        self.iter().map(|v| v.0.expr()).collect()
    }

    fn sum_conditions(&self) -> Expression<F> {
        sum::expr(&self.get_conditions())
    }
}

/// Trait around RLC
pub trait RLCable<F> {
    /// Returns the RLC of itself
    fn rlc(&self, r: &[Expression<F>]) -> Expression<F>;
}

impl<F: Field, E: Expressable<F>> RLCable<F> for Vec<E> {
    fn rlc(&self, r: &[Expression<F>]) -> Expression<F> {
        rlc::expr(&self.to_expr_vec(), r)
    }
}

impl<F: Field, E: Expressable<F>> RLCable<F> for [E] {
    fn rlc(&self, r: &[Expression<F>]) -> Expression<F> {
        rlc::expr(&self.to_expr_vec(), r)
    }
}

/// Trait around RLC
pub trait RLCChainable<F> {
    /// Returns the RLC of itself with a starting rlc/multiplier
    fn rlc_chain(&self, other: Expression<F>) -> Expression<F>;
}

impl<F: Field> RLCChainable<F> for (Expression<F>, Expression<F>) {
    fn rlc_chain(&self, other: Expression<F>) -> Expression<F> {
        self.0.expr() + self.1.expr() * other.expr()
    }
}

pub(crate) fn rlc_acc<F: Field>(values: &[F], rlc: F, mult: F, r: F) -> (F, F) {
    let mut rlc = rlc;
    let mut mult = mult;
    for &value in values.iter() {
        rlc = rlc + value * mult;
        mult *= r;
    }
    (rlc, mult)
}

/// Trait around RLC
pub trait RLCableValue<F> {
    /// Returns the RLC of itself
    fn rlc_value(&self, r: F) -> F;
}

impl<F: Field> RLCableValue<F> for Vec<u8> {
    fn rlc_value(&self, r: F) -> F {
        rlc::value(self, r)
    }
}

impl<F: Field> RLCableValue<F> for [u8] {
    fn rlc_value(&self, r: F) -> F {
        rlc::value(self, r)
    }
}

/// Trait around RLC
pub trait RLCChainableValue<F> {
    /// Returns the RLC of itself with a starting rlc/multiplier
    fn rlc_chain_value(&self, values: &[u8], r: F) -> (F, F);

    /// Returns the RLC of itself with a starting rlc/multiplier
    fn rlc_chain_value_f(&self, values: &[F], r: F) -> (F, F);
}

impl<F: Field> RLCChainableValue<F> for (F, F) {
    fn rlc_chain_value(&self, values: &[u8], r: F) -> (F, F) {
        let values = values.iter().map(|byte| byte.scalar()).collect::<Vec<F>>();
        rlc_acc(&values, self.0, self.1, r)
    }

    fn rlc_chain_value_f(&self, values: &[F], r: F) -> (F, F) {
        rlc_acc(values, self.0, self.1, r)
    }
}

/// require_parser
#[macro_export]
macro_rules! require_parser {
    {
        $cb:expr,
        lhs = ($($lhs:tt)*)
        rest = (== $($rhs:tt)*)
    } => {
        let description = $crate::concat_with_preamble!(
            stringify!($($lhs)*),
            " == ",
            stringify!($($rhs)*)
        );
        $crate::_require!($cb, description, $($lhs)* => $($rhs)*)
    };

    {
        $cb:expr,
        lhs = ($($lhs:tt)*)
        rest = ($next:tt $($rest:tt)*)
    } => {
        $crate::require_parser! {
            $cb,
            lhs = ($($lhs)* $next)
            rest = ($($rest)*)
        }
    };
}

/// _require2
#[macro_export]
macro_rules! _require2 {
    ($cb:expr, $($rest:tt)*) => {{
        $crate::require_parser! {
            $cb,
            lhs = ()
            rest = ($($rest)*)
        }
    }};
}

/// Creates a dummy constraint builder that cannot be used to add constraints.
#[macro_export]
macro_rules! _cb {
    () => {{
        ConstraintBuilder::<F>::new(0, None)
    }};
}

/// Concats arguments with preamble consisting of the originating file and line.
#[macro_export]
macro_rules! concat_with_preamble {
    ($($args:expr),* $(,)?) => {{
        concat!(
            file!(),
            ":",
            line!(),
            ": ",
            $(
                $args,
            )*
        )
    }};
}

/// Can be used to mark a specific branch as unreachable
#[macro_export]
macro_rules! _unreachablex {
    ($cb:expr $(,$descr:expr)?) => {{
        let descr = concat_with_preamble!(
            "unreachable executed",
            $(
                ": ",
                $descr,
            )*
        );
        _require!($cb, descr, true => false)
    }};
}

/// _require
#[macro_export]
macro_rules! _require {
    ($cb:expr, $lhs:expr => bool) => {{
        $cb.require_boolean(
            concat_with_preamble!(
                stringify!($lhs),
                " => ",
                "bool",
            ),
            $lhs.expr(),
        );
    }};

    ($cb:expr, $lhs:expr => $rhs:expr) => {{
        let description = concat_with_preamble!(
            stringify!($lhs),
            " => ",
            stringify!($rhs)
        );
        _require!($cb, description, $lhs => $rhs)
    }};

    ($cb:expr, $descr:expr, $lhs:expr => $rhs:expr) => {{
        let rhs = $rhs.to_expr_vec();
        if rhs.len() == 1 {
            $cb.require_equal(
                Box::leak($descr.to_string().into_boxed_str()),
                $lhs.expr(),
                rhs[0].expr(),
            );
        } else {
            $cb.require_in_set(
                Box::leak($descr.to_string().into_boxed_str()),
                $lhs.expr(),
                rhs.clone(),
            );
        }
    }};

    // Lookup using a tuple
    ($cb:expr, ($($v:expr),+) => @$tag:expr) => {{
        $cb.lookup(
            concat_with_preamble!(
                "(",
                $(
                    stringify!($v),
                    ", ",
                )*
                ") => @",
                stringify!($tag),
            ),
            $tag.to_string(),
            vec![$($v.expr(),)*],
        );
    }};
    ($cb:expr, $descr:expr, ($($v:expr),+)  => @$tag:expr) => {{
        $cb.lookup(
            Box::leak($descr.into_boxed_str()),
            $tag.to_string(),
            vec![$($v.expr(),)*],
        );
    }};

    // Lookup using an array
    ($cb:expr, $values:expr => @$tag:expr) => {{
        $cb.lookup(
            concat_with_preamble!(
                stringify!($values),
                " => @",
                stringify!($tag),
            ),
            $tag.to_string(),
            $values.clone(),
        );
    }};
    ($cb:expr, $descr:expr, $values:expr => @$tag:expr) => {{
        $cb.lookup(
            Box::leak($descr.to_string().into_boxed_str()),
            $tag.to_string(),
            $values.clone(),
        );
    }};

    // Put values in a lookup table using a tuple
    ($cb:expr, @$tag:expr => ($($v:expr),+)) => {{
        $cb.lookup_table(
            concat_with_preamble!(
                "@",
                stringify!($tag),
                " => (",
                $(
                    stringify!($v),
                    ", ",
                )*
                ")",
            ),
            $tag.to_string(),
            vec![$($v.expr(),)*],
        );
    }};
    // Put values in a lookup table using an array
    ($cb:expr, @$tag:expr => $values:expr) => {{
        $cb.lookup_table(
            concat_with_preamble!(
                "@",
                stringify!($tag),
                " => (",
                stringify!($values),
                ")",
            ),
            $tag.to_string(),
            $values,
        );
    }};
}

/// matchx
/// Supports `_` which works the same as in the normal `match`: if none of the
/// other arms are active the `_` arm will be executed and so can be used to
/// return some default values or could also be marked as unreachable (using the
/// unreachablex! macro).
#[macro_export]
macro_rules! _matchx {
    ($cb:expr, $($condition:expr => $when:expr),* $(, _ => $catch_all:expr)? $(,)?)  => {{
        let mut conditions = Vec::new();
        let mut cases = Vec::new();
        $cb.enter_branch_context();
        println!("$cb.enter_branch_context -- _matchx");

        $(
            $cb.push_condition($condition.expr());
            let ret = $when.clone();
            $cb.pop_condition();

            println!("$cb.switch_branch_context: {:?}", stringify!($condition));
            $cb.switch_branch_context(stringify!($condition));

            cases.push(($condition.expr(), ret));
            conditions.push($condition.expr());
        )*

        $(
            let catch_all_condition = not::expr(sum::expr(&conditions));
            $cb.push_condition(catch_all_condition.expr());
            let ret = $catch_all;
            $cb.pop_condition();

            println!("$cb.switch_branch_context: catch_all_condition");
            $cb.switch_branch_context("catch_all_condition");

            cases.push((catch_all_condition.expr(), ret));
            conditions.push(catch_all_condition.expr());
        )*

        // All conditions need to be boolean
        for condition in conditions.iter() {
            _require!($cb, condition => bool);
        }
        // Exactly 1 case needs to be enabled
        _require!($cb, sum::expr(&conditions) => 1);

        println!("$cb.exit_branch_context -- _matchx");
        $cb.exit_branch_context();

        cases.apply_conditions()
    }};
}

/// ifx
#[macro_export]
macro_rules! _ifx {
    ($cb:expr, $($condition:expr),* => $when_true:block $(elsex $when_false:block)?)  => {{

        let descr = stringify!($($condition)*);
        let condition = and::expr([$($condition.expr()),*]);

        $cb.enter_branch_context();
        println!("$cb.enter_branch_context");

        $cb.push_condition(condition.expr());
        let ret_true = $when_true;
        $cb.pop_condition();

        println!("$cb.switch_branch_context: {:?}{:?}", "ifx!", descr);
        $cb.switch_branch_context(&format!("{:?}{:?}", "ifx!", descr));

        #[allow(unused_assignments, unused_mut)]
        let mut ret = ret_true.conditional(condition.expr());
        $(
            // In if/else cases, the condition needs to be boolean
            _require!($cb, condition => bool);

            $cb.push_condition(not::expr(condition.expr()));
            let ret_false = $when_false;
            $cb.pop_condition();

            println!("$cb.switch_branch_context: {:?}{:?}", "elsex", descr);
            $cb.switch_branch_context(&format!("{:?}{:?}", "elsex", descr));

            ret = ret_true.select(condition.expr(), &ret_false);
        )*
        println!("$cb.exit_branch_context");
        $cb.exit_branch_context();
        ret
    }};
}

/// matchr
#[macro_export]
macro_rules! matchr {
    ($($condition:expr => $when:expr),* $(, _ => $catch_all:expr)? $(,)?)  => {{
        $(
            if $condition {
                return $when;
            }
        )*
        $(
            return $catch_all;
        )*
        unreachable!();
    }};
}

/// matchw
#[macro_export]
macro_rules! matchw {
    ($($condition:expr => $when:expr),* $(, _ => $catch_all:expr)? $(,)?)  => {{
        $(
            if $condition {
                $when
            }
        )*
        $(
            $catch_all
        )*
    }};
}

/// assign
#[macro_export]
macro_rules! assign {
    ($region:expr, ($column:expr, $offset:expr) => $value:expr) => {{
        let description =
            $crate::concat_with_preamble!(stringify!($column), " => ", stringify!($value));
        let value: F = $value;
        $region.assign_advice(|| description, $column, $offset, || Value::known(value))
    }};
}

/// assign
#[macro_export]
macro_rules! assignf {
    ($region:expr, ($column:expr, $offset:expr) => $value:expr) => {{
        let description =
            $crate::concat_with_preamble!(stringify!($column), " => ", stringify!($value));
        let value: F = $value;
        $region.assign_fixed(|| description, $column, $offset, || Value::known(value))
    }};
}

/// Circuit builder macros
/// Nested macro's can't do repetition (https://github.com/rust-lang/rust/issues/35853)
/// so we expose a couple of permutations here manually.
#[macro_export]
macro_rules! circuit {
    ([$meta:expr, $cb:expr], $content:block) => {{
        #[allow(unused_imports)]
        use $crate::{concat_with_preamble, _require, _matchx, _ifx, _unreachablex};
        #[allow(unused_imports)]
        use gadgets::util::{and, not, or, sum, Expr};
        #[allow(unused_imports)]
        use $crate::circuit_tools::constraint_builder::{Conditionable, Expressable, Selectable};

        #[allow(unused_macros)]
        macro_rules! f {
            ($column:expr, $rot:expr) => {{
                $meta.query_fixed($column.clone(), Rotation($rot as i32))
            }};
            ($column:expr) => {{
                $meta.query_fixed($column.clone(), Rotation::cur())
            }};
        }

        #[allow(unused_macros)]
        macro_rules! a {
            ($column:expr, $rot:expr) => {{
                $meta.query_advice($column.clone(), Rotation($rot as i32))
            }};
            ($column:expr) => {{
                $meta.query_advice($column.clone(), Rotation::cur())
            }};
        }

        #[allow(unused_macros)]
        macro_rules! not {
            ($expr:expr) => {{
                gadgets::util::not::expr($expr.expr())
            }};
        }

        #[allow(unused_macros)]
        macro_rules! invert {
            ($expr:expr) => {{
                Expression::Constant(F::from($expr as u64).invert().unwrap())
            }};
        }

        #[allow(unused_macros)]
        macro_rules! require {
            ($lhs:expr => bool) => {{
                _require!($cb, $lhs => bool);
            }};

            ($lhs:expr => $rhs:expr) => {{
                _require!($cb, $lhs => $rhs);
            }};

            ($name:expr, $lhs:expr => $rhs:expr) => {{
                _require!($cb, $name, $lhs => $rhs);
            }};

            (($a:expr) => @$tag:expr) => {{
                _require!($cb, ($a) => @$tag);
            }};

            (($a:expr, $b:expr) => @$tag:expr) => {{
                _require!($cb, ($a, $b) => @$tag);
            }};

            (($a:expr, $b:expr, $c:expr) => @$tag:expr) => {{
                _require!($cb, ($a, $b, $c) => @$tag);
            }};

            (($a:expr, $b:expr, $c:expr, $d:expr) => @$tag:expr) => {{
                _require!($cb, ($a, $b, $c, $d) => @$tag);
            }};

            ($values:expr => @$tag:expr) => {{
                _require!($cb, $values => @$tag);
            }};

            ($descr:expr, $values:expr => @$tag:expr) => {{
                _require!($cb, $descr, $values => @$tag);
            }};

            (@$tag:expr => ($a:expr, $b:expr, $c:expr)) => {{
                _require!($cb, @$tag => ($a, $b, $c));
            }};

            (@$tag:expr => $values:expr) => {{
                _require!($cb, @$tag => $values);
            }};
        }

        #[allow(unused_macros)]
        macro_rules! ifx {
            ($condition:expr => $when_true:block elsex $when_false:block) => {{
                _ifx!($cb, $condition => $when_true elsex $when_false)
            }};
            ($condition_a:expr, $condition_b:expr => $when_true:block elsex $when_false:block) => {{
                _ifx!($cb, $condition_a, $condition_b => $when_true elsex $when_false)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr => $when_true:block elsex $when_false:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c => $when_true elsex $when_false)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr, $condition_d:expr => $when_true:block elsex $when_false:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c, $condition_d => $when_true elsex $when_false)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr, $condition_d:expr, $condition_e:expr, $condition_f:expr, $condition_g:expr => $when_true:block elsex $when_false:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c, $condition_d, $condition_e, $condition_f, $condition_g => $when_true elsex $when_false)
            }};

            ($condition:expr => $when_true:block) => {{
                _ifx!($cb, $condition => $when_true)
            }};
            ($condition_a:expr, $condition_b:expr => $when_true:block) => {{
                _ifx!($cb, $condition_a, $condition_b => $when_true)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr => $when_true:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c => $when_true)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr, $condition_d:expr => $when_true:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c, $condition_d => $when_true)
            }};
            ($condition_a:expr, $condition_b:expr, $condition_c:expr, $condition_d:expr, $condition_e:expr => $when_true:block) => {{
                _ifx!($cb, $condition_a, $condition_b, $condition_c, $condition_d, $condition_e => $when_true)
            }};
        }

        #[allow(unused_macros)]
        macro_rules! matchx {
            ($condition_a:expr => $when_a:expr,) => {{
                _matchx!($cb, $condition_a => $when_a)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr, $condition_c:expr => $when_c:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b, $condition_c => $when_c)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr, $condition_c:expr => $when_c:expr, $condition_d:expr => $when_d:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b, $condition_c => $when_c, $condition_d => $when_d,)
            }};

            ($condition_a:expr => $when_a:expr, _ => $catch_all:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, _ => $catch_all,)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr, _ => $catch_all:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b, _ => $catch_all,)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr, $condition_c:expr => $when_c:expr, _ => $catch_all:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b, $condition_c => $when_c, _ => $catch_all,)
            }};
            ($condition_a:expr => $when_a:expr, $condition_b:expr => $when_b:expr, $condition_c:expr => $when_c:expr, $condition_d:expr => $when_d:expr, _ => $catch_all:expr,) => {{
                _matchx!($cb, $condition_a => $when_a, $condition_b => $when_b, $condition_c => $when_c, $condition_d => $when_d, _ => $catch_all,)
            }};
        }

        #[allow(unused_macros)]
        macro_rules! unreachablex {
            () => {{
                _unreachablex!($cb)
            }};
            ($arg:expr) => {{
                _unreachablex!($cb, $arg)
            }};
        }

        $content
    }};
}

macro_rules! witness {
    ($name:ident, $t:ty) => {{
        match objects[0].as_any().downcast_ref::<$t>() {
            Some(b) => b,
            None => panic!("&a isn't a B!"),
        }
    }};
}