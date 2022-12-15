use gadgets::util::Expr;
use halo2_proofs::{
    arithmetic::FieldExt,
    plonk::{Column, ConstraintSystem, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::{
    cs,
    evm_circuit::util::rlc,
    mpt_circuit::helpers::{get_bool_constraint, range_lookups},
    mpt_circuit::FixedTableTag,
    mpt_circuit::{
        columns::{AccumulatorCols, MainCols},
        helpers::{
            get_num_rlp_bytes, get_rlp_meta_bytes, get_rlp_value_bytes, BaseConstraintBuilder,
        },
    },
};

/*
A branch occupies 19 rows:
BRANCH.IS_INIT
BRANCH.IS_CHILD 0
...
BRANCH.IS_CHILD 15
BRANCH.IS_EXTENSION_NODE_S
BRANCH.IS_EXTENSION_NODE_C

Example:

[1 0 1 0 248 241 0 248 241 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 164 92 78 34 81 137 173 236 78 208 145 118 128 60 46 5 176 8 229 165 42 222 110 4 252 228 93 243 26 160 241 85 0 160 95 174 59 239 229 74 221 53 227 115 207 137 94 29 119 126 56 209 55 198 212 179 38 213 219 36 111 62 46 43 176 168 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 60 157 212 182 167 69 206 32 151 2 14 23 149 67 58 187 84 249 195 159 106 68 203 199 199 65 194 33 215 102 71 138 0 160 60 157 212 182 167 69 206 32 151 2 14 23 149 67 58 187 84 249 195 159 106 68 203 199 199 65 194 33 215 102 71 138 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 21 230 18 20 253 84 192 151 178 53 157 0 9 105 229 121 222 71 120 109 159 109 9 218 254 1 50 139 117 216 194 252 0 160 21 230 18 20 253 84 192 151 178 53 157 0 9 105 229 121 222 71 120 109 159 109 9 218 254 1 50 139 117 216 194 252 1]
[0 160 229 29 220 149 183 173 68 40 11 103 39 76 251 20 162 242 21 49 103 245 160 99 143 218 74 196 2 61 51 34 105 123 0 160 229 29 220 149 183 173 68 40 11 103 39 76 251 20 162 242 21 49 103 245 160 99 143 218 74 196 2 61 51 34 105 123 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 0 140 67 252 58 164 68 143 34 163 138 133 54 27 218 38 80 20 142 115 221 100 73 161 165 75 83 53 8 58 236 1 0 160 0 140 67 252 58 164 68 143 34 163 138 133 54 27 218 38 80 20 142 115 221 100 73 161 165 75 83 53 8 58 236 1 1]
[0 160 149 169 206 0 129 86 168 48 42 127 100 73 109 90 171 56 216 28 132 44 167 14 46 189 224 213 37 0 234 165 140 236 0 160 149 169 206 0 129 86 168 48 42 127 100 73 109 90 171 56 216 28 132 44 167 14 46 189 224 213 37 0 234 165 140 236 1]
[0 160 42 63 45 28 165 209 201 220 231 99 153 208 48 174 250 66 196 18 123 250 55 107 64 178 159 49 190 84 159 179 138 235 0 160 42 63 45 28 165 209 201 220 231 99 153 208 48 174 250 66 196 18 123 250 55 107 64 178 159 49 190 84 159 179 138 235 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17]

The constraints in this file check whether the RLC of the branch init row (first branch row)
is computed correctly.

There are three possible cases:
1. Branch (length 21 = 213 - 192) with one byte of RLP meta data
    [213,128,194,32,1,128,194,32,1,128,128,128,128,128,128,128,128,128,128,128,128,128]
    In this case the init row looks like (specifying only for `S`, we put `x` for `C`):
    [1,1,x,x,213,0,0,...]
    The RLC is simply `213`.

2. Branch (length 83) with two bytes of RLP meta data
    [248,81,128,128,...
    In this case the init row looks like (specifying only for `S`, we put `x` for `C`):
    [1,0,x,x,248,81,0,...]
    The RLC is `248 + 81*r`.


3. Branch (length 340) with three bytes of RLP meta data
    [249,1,81,128,16,...
    In this case the init row looks like (specifying only for `S`, we put `x` for `C`):
    [1,0,x,x,249,1,81,...]
    The RLC is `249 + 1*r + 81*r^2`.

We specify the case as (note that `S` branch and
`C` branch can be of different length. `s_rlp1, s_rlp2` is used for `S` and
`s_main.bytes[0], s_main.bytes[1]` is used for `C`):
    rlp1, rlp2: 1, 1 means 1 RLP byte
    rlp1, rlp2: 1, 0 means 2 RLP bytes
    rlp1, rlp2: 0, 1 means 3 RLP bytes

The example branch init above is the second case (two RLP meta bytes).

Note: the constraints for the selectors in branch init row to be boolean are in `branch.rs`
and `extension_node.rs`.
*/

#[derive(Clone, Debug)]
pub(crate) struct BranchInitConfig<F> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> BranchInitConfig<F> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        q_enable: impl Fn(&mut VirtualCells<'_, F>) -> Expression<F> + Copy,
        s_main: MainCols<F>,
        accs: AccumulatorCols<F>,
        r: Expression<F>,
        fixed_table: [Column<Fixed>; 3],
    ) -> Self {
        // Short RLP, meta data contains two bytes: 248, 81
        // [1,0,1,0,248,81,0,248,81,0,3,0,0,0,...
        // The length of RLP stream is: 81.

        // Long RLP, meta data contains three bytes: 249, 2, 17
        // [0,1,0,1,249,2,17,249,2,17,7,0,0,0,...
        // The length of RLP stream is: 2 * 256 + 17.

        // The RLC of the init branch comprises 1, 2, or 3 bytes. This gate ensures the
        // RLC is computed properly in each of the three cases. It also ensures
        // that the values that specify the case are boolean.
        meta.create_gate("Branch init RLC", |meta| {
            let rot = Rotation::cur();
            let q_enable = q_enable(meta);
            let mut cb = BaseConstraintBuilder::default();
            cs!{[cb],
            ifx(q_enable) {
                for (accumulators, is_s) in [
                    (accs.acc_s, true),
                    (accs.acc_c, false)
                ] {
                    let rlp_meta = get_rlp_meta_bytes(meta, s_main.clone(), is_s, rot);
                    let (one_rlp_byte, two_rlp_bytes, three_rlp_bytes) = get_num_rlp_bytes(meta, s_main.clone(), is_s, rot);
                    let rlp = get_rlp_value_bytes(meta, s_main.clone(), is_s, rot);

                    // Check branch accumulator in row 0
                    let acc = meta.query_advice(accumulators.rlc, rot);
                    let mult = meta.query_advice(accumulators.mult, rot);

                    // Boolean checks
                    for selector in rlp_meta {
                        cb.require_boolean("branch init boolean", selector);
                    }

                    // Branch RLC checks
                    // 1 RLP byte
                    cs!{[cb],
                    ifx(one_rlp_byte) {
                        cb.require_equal("Branch accumulator row 0 (1)", rlp[0].expr(), acc.expr());
                        cb.require_equal("Branch mult row 0 (1)", r.expr(), mult.expr());
                    }}
                    // 2 RLP bytes
                    cs!{[cb],
                    ifx(two_rlp_bytes) {
                        cb.require_equal("Branch accumulator row 0 (2)", rlp[0].expr() + rlp[1].expr() * r.expr(), acc.expr());
                        cb.require_equal("Branch mult row 0 (2)", r.expr() * r.expr(), mult.expr());
                    }}
                    // 3 RLP bytes
                    cs!{[cb],
                    ifx(three_rlp_bytes) {
                        cb.require_equal(
                            "Branch accumulator row 0 (3)",
                             rlp[0].expr() + rlp[1].expr() * r.expr() + rlp[2].expr() * r.expr() * r.expr(), acc.expr()
                        );
                        cb.require_equal("Branch mult row 0 (3)", r.expr() * r.expr() * r.expr(), mult.expr());
                    }}
                }
            }}

            cb.gate(1.expr())
        });

        /*
        Range lookups ensure that the values in the used columns are all bytes (between 0 - 255).
        Note: range lookups for extension node rows are in `extension_node_key.rs`.
        */
        range_lookups(
            meta,
            q_enable,
            s_main.bytes.to_vec(),
            FixedTableTag::Range256,
            fixed_table,
        );
        range_lookups(
            meta,
            q_enable,
            [s_main.rlp1, s_main.rlp2].to_vec(),
            FixedTableTag::Range256,
            fixed_table,
        );

        BranchInitConfig {
            _marker: PhantomData,
        }
    }
}