use eth_types::Field;
use gadgets::util::{Scalar, Expr};
use halo2_proofs::{plonk::Expression, circuit::Value};

use crate::circuit_tools::constraint_builder::ExprVec;



/// Returns the random linear combination of the inputs.
/// Encoding is done as follows: v_0 * R^0 + v_1 * R^1 + ...
pub(crate) mod be_rlc {
    use std::ops::{Add, Mul};

    use crate::util::Expr;
    use eth_types::Field;
    use halo2_proofs::{plonk::Expression, circuit::Value};

    pub(crate) fn expr<F: Field, E: Expr<F>>(
        expressions: &[E], 
        randomness: E
    ) -> (Expression<F>, Expression<F>) {
        if !expressions.is_empty() {
            generic(expressions.iter().map(|e| e.expr()), randomness.expr())
        } else {
            (0.expr(), randomness.expr())
        }
    }

    pub(crate) fn value<'a, F: Field, I>(values: I, randomness: F) -> (F, F)
    where
        I: IntoIterator<Item = &'a u8>,
        <I as IntoIterator>::IntoIter: DoubleEndedIterator,
    {
        let values = values
            .into_iter()
            .map(|v| F::from(*v as u64))
            .collect::<Vec<F>>();
        if !values.is_empty() {
            generic(values, randomness)
        } else {
            (F::ZERO, randomness)
        }
    }

    pub(crate) fn value_wrapped<'a, F: Field, I>(values: I, randomness: Value<F>) -> (Value<F>, Value<F>)
    where
        I: IntoIterator<Item = Value<F>>,
        <I as IntoIterator>::IntoIter: DoubleEndedIterator,
    {
        let values = values.into_iter().collect::<Vec<Value<F>>>();
        if !values.is_empty() {
            generic(values, randomness)
        } else {
            (Value::known(F::ZERO), randomness)
        }
    }

    pub(crate) fn generic<V, I>(values: I, randomness: V) -> (V, V)
    where
        I: IntoIterator<Item = V>,
        <I as IntoIterator>::IntoIter: DoubleEndedIterator,
        V: Clone + Add<Output = V> + Mul<Output = V>,
    {
        let mut values = values.into_iter();
        let init = (values.next().expect("values should not be empty"), randomness.clone());

        values.fold(init, |acc, value| (acc.0 * randomness.clone() + value, acc.1 * randomness.clone()))
    }
}


/// Trait around RLC
pub trait BERLCable<F: Field> {
    /// Returns the RLC of itself
    fn be_rlc(&self, r: &Expression<F>) -> (Expression<F>, Expression<F>);
}

impl<F: Field, E: ExprVec<F> + ?Sized> BERLCable<F> for E {
    fn be_rlc(&self, r: &Expression<F>) -> (Expression<F>, Expression<F>) {
        be_rlc::expr(&self.to_expr_vec(), r.expr())
    }
}

/// Trait around RLC
pub trait BERLCChainable<F> {
    /// Returns the RLC of itself with a starting rlc/multiplier
    fn be_rlc_chain(&self, other:(Expression<F>, Expression<F>)) -> (Expression<F>, Expression<F>);
}

impl<F: Field> BERLCChainable<F> for (Expression<F>, Expression<F>) {
    fn be_rlc_chain(&self, other: (Expression<F>, Expression<F>)) -> (Expression<F>, Expression<F>) {
        (self.0.clone() * other.1 + other.0, self.1 * other.1)
    }
}

/// Trait around RLC
pub trait BERLCableValue<F> {
    /// Returns the RLC of itself
    fn be_rlc_value(&self, r: F) -> (F, F);
}

impl<F: Field> BERLCableValue<F> for Vec<u8> {
    fn be_rlc_value(&self, r: F) -> (F, F) {
        be_rlc::value(self, r)
    }
}

impl<F: Field> BERLCableValue<F> for [u8] {
    fn be_rlc_value(&self, r: F) -> (F, F) {
        be_rlc::value(self, r)
    }
}

impl<F: Field> BERLCableValue<Value<F>> for Vec<Value<F>> {
    fn be_rlc_value(&self, r: Value<F>) ->(Value<F>, Value<F>) {
        be_rlc::value_wrapped(self.to_owned(), r)
    }
}

impl<F: Field> BERLCableValue<Value<F>> for [Value<F>] {
    fn be_rlc_value(&self, r: Value<F>) -> (Value<F>, Value<F>) {
        be_rlc::value_wrapped(self.to_owned(), r)
    }
}

/// Trait around RLC
pub trait BERLCChainableValue<F, I> {
    /// Returns the RLC of itself with a starting rlc/multiplier
    fn be_rlc_chain_value(&self, values: I, r: F) -> (F, F);
}

impl<F: Field> BERLCChainableValue<F, &[u8]> for (F, F) {
    fn be_rlc_chain_value(&self, values: &[u8], r: F) -> (F, F) {
        let other_rlc = be_rlc::value(values, r);
        (self.0 + other_rlc.0, self.1 * other_rlc.1)
    }
}

impl<F: Field> BERLCChainableValue<F, Vec<u8>> for (F, F) {
    fn be_rlc_chain_value(&self, values: Vec<u8>, r: F) -> (F, F) {
        let other_rlc = be_rlc::value(&values, r);
        (self.0 + other_rlc.0, self.1 * other_rlc.1)
    }
}

impl<F: Field> BERLCChainableValue<Value<F>, &[Value<F>]> for (Value<F>, Value<F>) {
    fn be_rlc_chain_value(&self, values: &[Value<F>], r: Value<F>) -> (Value<F>, Value<F>) {
        let other_rlc = be_rlc::value_wrapped(values.to_owned(), r);
        (self.0 * other_rlc.1 + other_rlc.0, self.1 * other_rlc.1)
    }
}

impl<F: Field> BERLCChainableValue<Value<F>, Vec<Value<F>>> for (Value<F>, Value<F>) {
    fn be_rlc_chain_value(&self, values: Vec<Value<F>>, r: Value<F>) -> (Value<F>, Value<F>) {
        let other_rlc = be_rlc::value_wrapped(values, r);
        (self.0 * other_rlc.1 + other_rlc.0, self.1 * other_rlc.1)
    }
}

