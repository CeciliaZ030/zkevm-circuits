use gadgets::util::{not, sum, Expr};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Region, Value},
    plonk::VirtualCells,
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::{
    circuit,
    circuit_tools::{DataTransition, LRCable},
    evm_circuit::util::rlc,
    mpt_circuit::{
        helpers::{get_num_bytes_list_short, get_num_nibbles},
        param::{ARITY, C_RLP_START, C_START, HASH_WIDTH, RLP_HASH_VALUE, RLP_LIST_LONG, RLP_NIL},
    },
    mpt_circuit::{
        helpers::{BranchNodeInfo, MPTConstraintBuilder},
        param::EXTENSION_ROWS_NUM,
        witness_row::MptWitnessRow,
        MPTContext,
    },
    mpt_circuit::{MPTConfig, ProofValues},
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

[1 0 1 0 248 81 0 248 81 0 14 1 0 6 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 29 143 36 49 6 106 55 88 195 10 34 208 147 134 155 181 100 142 66 21 255 171 228 168 85 11 239 170 233 241 171 242 0 160 29 143 36 49 6 106 55 88 195 10 34 208 147 134 155 181 100 142 66 21 255 171 228 168 85 11 239 170 233 241 171 242 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[0 160 135 117 154 48 1 221 143 224 133 179 90 254 130 41 47 5 101 84 204 111 220 62 215 253 155 107 212 69 138 221 91 174 0 160 135 117 154 48 1 221 143 224 133 179 90 254 130 41 47 5 101 84 204 111 220 62 215 253 155 107 212 69 138 221 91 174 1]
[0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
[226 30 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 160 30 252 7 160 150 158 68 221 229 48 73 181 91 223 120 156 43 93 5 199 95 184 42 20 87 178 65 243 228 156 123 174 0 16]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 160 30 252 7 160 150 158 68 221 229 48 73 181 91 223 120 156 43 93 5 199 95 184 42 20 87 178 65 243 228 156 123 174 0 17]

The last two rows present the extension node. This might be a bit misleading, because the extension
node appears in the trie above the branch (the first 17 rows).

The constraints in `extension_node.rs` check:
 - RLP of the node
 - that the branch hash is in the extension node (the bytes `[30 252 ... 174]` in extension node rows present the hash
of the underlying branch)
 - that the hash of the extension is in the parent node.

Note that an attacker can set `is_extension_node = 1`
for a regular branch (or `is_extension_node = 0` for the extension node),
the final key RLC check fails because key RLC is computed differently
for extension nodes and regular branches - a regular branch occupies only one
key nibble (`modified_node`), while extension node occupies at least one additional
nibble (the actual extension of the extension node).

Some observations:

[228,130,0,149,160,114,253,150,133,18,192,156,19,241,162,51,210,24,1,151,16,
48,7,177,42,60,49,34,230,254,242,79,132,165,90,75,249]
Note that the first element (228 in this case) can go much higher - for
example, if there are 40 nibbles, this would take 20 bytes which
would make the first element 248.

If only one byte in key:
[226,16,160,172,105,12...

Extension node with non-hashed branch:
List contains up to 55 bytes (192 + 55)
[247,160,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
213,128,194,32,1,128,194,32,1,128,128,128,128,128,128,128,128,128,128,128,
128,128]

List contains more than 55 bytes
[248,58,159,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
217,128,196,130,32,0,1,128,196,130,32,0,1,128,128,128,128,128,128,128,128,
128,128,128,128,128]

Note that the extension node can be much shorter than the one above - in case
when there are less nibbles, so we cannot say that 226 appears as
the first byte only when there are hashed nodes in the branch and
there is only one nibble. Branch with two non-hashed nodes
(that's the shortest possible branch): [217,128,196,130,32,0,1,
128,196,130,32,0,1,128,128,128,128,128,128,128,128,128,128,128,128,128]
Note: branch contains at least 26 bytes. 192 + 26 = 218

If proofEl[0] <= 247 (length at most 55, so proofEl[1] doesn't specify the
length of the whole     remaining stream, only of the next
substream) If proofEl[1] <= 128:
    There is only 1 byte for nibbles (keyLen = 1) and this is proofEl[1].
Else:
    Nibbles are stored in more than 1 byte, proofEl[1] specifies the length
of bytes. Else:
proofEl[1] contains the length of the remaining stream.
proofEl[2] specifies the length of the bytes (for storing nibbles).
Note that we can't have only one nibble in this case.
*/

#[derive(Clone, Debug, Default)]
pub(crate) struct ExtensionNodeConfig<F> {
    _marker: PhantomData<F>,
}

/*
Let's say we have branch1 and branch2 below it.

branch1 S row 0 || branch1 C row 0
...
branch1 S row 15 || branch1 C row 15

branch2 S row 0 || branch2 C row 0
...
branch2 S row 15 || branch2 C row 15

Hash of branch2 S is in one of the branch1 rows (in S columns).
Hash of branch2 C is in one of the branch1 rows (in C columns).

In what follows, we write branch without S and C - it is the same for both cases.

Key key1 determines the position of branch2 hash in branch1 (0-15).
To check this, branch2 RLC (over all RLP bytes - all 1+16 rows, 1 is for branch init row)
is checked to have a hash in branch1, at modified_node index
(modified_node depends on key key1).

However, with extension node it's a bit different.

branch1 S row 0 || branch1 C row 0
...
branch1 S row 15 || branch1 C row 15
extension1 S
extension1 C

branch2 S row 0 || branch2 C row 0
...
branch2 S row 15 || branch2 C row 15
extension2 S
extension2 C

There are additional rows immediately after branch 16 rows - one row for
extension node S and one row for extension node C. These rows are empty when
we have a regular branch.

Let's say branch2 is extension node. In this case, extension2 row contains:
  - key bytes that present the extension
  - hash of branch2

---
Example 1:

Key extension of length 2:
[228, 130, 0,          149,        160, 114,                    253,                     150,133,18,192,156,19,241,162,51,210,24,1,151,16,48,7,177,42,60,49,34,230,254,242,79,132,165,90,75,249]
[rlp, rlp, key byte 1, key byte 2, rlp, hash of branch2 byte 0, hash of branch2 byte 1, ...]
[0, 149] presents key extension:
  - 0 because it's even length (in branch it would be 16, see terminator)
  - 149 = 9*16 + 5
Key extension is [9, 5].

Two constraints are needed:
  - the hash of extension node (extension node RLC is needed) needs to be
    checked whether it's in branch1
  - the hash of branch2 is in extension node.

Also, it needs to be checked that key extension corresponds to key1 (not in this chip).

---
Example 2:

Key extension of length 1:
[226, 16,        160,172,105,12...
[rlp, key byte1, ...
[16] presents key extension:
  - 16 = 0 + 16
Key extension is [0].

*/

impl<F: FieldExt> ExtensionNodeConfig<F> {
    pub fn configure(
        meta: &mut VirtualCells<'_, F>,
        cb: &mut MPTConstraintBuilder<F>,
        ctx: MPTContext<F>,
        is_s: bool,
    ) -> Self {
        let position_cols = ctx.position_cols;
        let s_main = ctx.s_main;
        let c_main = ctx.c_main;
        let accs = ctx.accumulators;
        let r = ctx.r.clone();

        let rot_s = if is_s { 0 } else { -1 };
        let rot_last_child = rot_s - 1;
        let rot_branch_init = rot_last_child - (ARITY as i32);
        let rot_branch_child_prev = rot_branch_init - EXTENSION_ROWS_NUM - 1;

        circuit!([meta, cb.base], {
            let not_first_level = a!(position_cols.not_first_level);
            let ext = BranchNodeInfo::new(meta, ctx.clone(), is_s, rot_branch_init);
            let ext_rlc = DataTransition::from(a!(accs.acc_s.rlc, rot_s), a!(accs.acc_c.rlc));

            // There are two cases:
            // - hashed branch has RLP_HASH_VALUE at c_rlp2 and hash in c_advices,
            // - non-hashed branch has 0 at c_rlp2 and all the bytes in c_advices
            // TODO(Brecht): why different layout for hashed values? If for hash detection
            // just do == 32?
            require!(a!(c_main.rlp2) => [0, RLP_HASH_VALUE]);

            // `short` means there is only one nibble in the extension node, `long` means
            // there are at least two. `even` means the number of nibbles is
            // even, `odd` means the number of nibbles is odd. `c16` means that
            // above the branch there are even number of nibbles (the same as
            // saying that `modified_node` of the branch needs to be multiplied
            // by 16 in the computation of the key RLC), `c1` means that above
            // the branch there are odd number of nibbles (the same as saying
            // that `modified_node` of the branch needs to be multiplied by 1 in
            // the computation of the key RLC).
            let type_selectors_c1 = [
                ext.is_short_c1.expr(),
                ext.is_long_even_c1.expr(),
                ext.is_long_odd_c1.expr(),
            ];
            let type_selectors_c16 = [
                ext.is_short_c16.expr(),
                ext.is_long_even_c16.expr(),
                ext.is_long_odd_c16.expr(),
            ];
            let type_selectors = [type_selectors_c1.clone(), type_selectors_c16.clone()].concat();
            let misc_selectors = [ext.is_longer_than_55.expr(), ext.is_ext_non_hashed.expr()];

            ifx! {a!(ctx.branch.is_init) => {
                // Check that the selectors are boolean
                for selector in type_selectors.iter().chain(misc_selectors.iter()) {
                    require!(selector => bool);
                }
                // For extension nodes exactly 1 type selector needs to be enabled.
                require!(sum::expr(&type_selectors) => 1);
                // `is_c16` and `is_c1` selectors are set using the extension node type selector data.
                // (while in case of a regular branch the extension node selectors do not hold this information).
                require!(ext.is_key_even() => sum::expr(&type_selectors_c1));
                require!(ext.is_key_odd() => sum::expr(&type_selectors_c16));
            }}

            // RLP encoding checks: [key, branch]
            // In C we have nibbles, we check below only for S.
            if is_s {
                // Even implies long and implies s_main.bytes[0] = 0
                // This prevents the attacker to set the number of nibbles to be even
                // when it is not even.
                ifx! {ext.is_even() => {
                    require!(a!(s_main.bytes[0]) => 0);
                }}
                ifx! {ext.is_longer_than_55 => {
                    require!(a!(s_main.rlp1) => RLP_LIST_LONG + 1);
                }}
                // Verify that the lenghts are consistent.
                require!(ext.ext_len(meta) => ext.ext_key_num_bytes(meta) + ext.ext_branch_num_bytes(meta));
            }

            // Check whether branch is in the extension node.
            // The intermediate RLC after `s_main` bytes needs to be properly computed.
            // s_rlp1, s_rlp2, s_bytes need to be the same in both extension rows.
            // However, to make space for nibble witnesses, we put nibbles in
            // extension row C s_bytes. So we use s_bytes from S row.
            let rlc = s_main.expr(meta, rot_s).rlc(&r);
            // TODO(Brecht): Do we need to store the RLC here? we can just use `rlc`
            // directly below...
            require!(ext_rlc.prev() => rlc);
            // Check whether the extension node RLC is properly computed.
            // The RLC is used to check whether the extension node is a node at the
            // appropriate position in the parent node.
            let mult = a!(accs.acc_s.mult);
            ifx! {ext.contains_hashed_branch(meta) => {
                let rlc = ext_rlc.prev() + c_main.expr(meta, 0)[1..].to_vec().rlc_chain(&r, mult.expr());
                require!(ext_rlc => rlc);
            } elsex {
                let rlc = ext_rlc.prev() + c_main.expr(meta, 0)[2..].to_vec().rlc_chain(&r, mult.expr());
                require!(ext_rlc => rlc);
                // RLC bytes zero check
                cb.set_length_c(ext.ext_branch_num_bytes(meta));
            }}
            // We check that the branch hash RLC (computed over the first 17 rows)
            // corresponds to the extension node RLC stored in the extension node row.
            // TODO: acc currently doesn't have branch ValueNode info (which 128 if nil)
            let branch_rlc = rlc::expr(
                &[a!(accs.acc(is_s).rlc, rot_last_child), RLP_NIL.expr()],
                &[a!(accs.acc(is_s).mult, rot_last_child)],
            );
            let rlc = c_main.bytes(meta, 0).rlc(&r);
            ifx! {ext.contains_hashed_branch(meta) => {
                // Check that `(branch_rlc, extension_node_hash_rlc`) in in the keccak table.
                require!((1, branch_rlc, ext.num_bytes(meta), rlc) => @"keccak");
            } elsex {
                // Check if the RLC matches.
                require!(branch_rlc => rlc);
            }}

            // Check if the extension node is in its parent.
            // Note: acc_c in both cases.
            let ext_rlc = a!(accs.acc_c.rlc);
            let ext_len = get_num_bytes_list_short(a!(s_main.rlp1, rot_s));
            ifx! {not_first_level => {
                ifx!{not!(ext.is_placeholder()) => {
                    // Only check if there is an account above the leaf.
                    // rot_into_branch_init - 1 because we are in the last branch child
                    // (rot_into_branch_init takes us to branch init).
                    ifx!{ext.is_below_account(meta) => {
                        /* Extension node in first level of storage trie - hash compared to the storage root */
                        // When extension node is in the first level of the storage trie, we need to check whether
                        // `hash(ext_node) = storage_trie_root`. We do this by checking whether
                        // `(ext_node_RLC, storage_trie_root_RLC)` is in the keccak table.
                        // Note: extension node in the first level cannot be shorter than 32 bytes (it is always hashed).
                        let storage_root_rlc = ext.storage_root_in_account_above(meta);
                        require!((1, ext_rlc, ext_len, storage_root_rlc) => @"keccak");
                    } elsex {
                        let mod_node_hash_rlc = a!(accs.mod_node_rlc(is_s), rot_branch_child_prev);
                        ifx!{ext.is_ext_non_hashed() => {
                            /* Extension node in parent branch (non-hashed extension node) */
                            // When an extension node is not hashed, we do not check whether it is in a parent
                            // branch using a lookup (see above), instead we need to check whether the branch child
                            // at `modified_node` position is exactly the same as the extension node.
                            require!(ext_rlc => mod_node_hash_rlc);
                        } elsex {
                            /* Extension node hash in parent branch */
                            // Check whether the extension node hash is in the parent branch.
                            // That means we check whether
                            // `(extension_node_RLC, node_hash_RLC)` is in the keccak table where `node` is a parent
                            // branch child at `modified_node` position.
                            require!((1, ext_rlc, ext_len, mod_node_hash_rlc) => @"keccak");
                        }}
                    }}
                }}
            } elsex {
                /* Branch in first level of account trie - hash compared to root */
                // In the first level of the account trie the extension node hash
                // needs to be match the trie root.
                // Note: the branch counterpart is implemented in `branch_hash_in_parent.rs`.
                require!((1, ext_rlc, ext_len, a!(ctx.inter_root(is_s))) => @"keccak");
            }}

            // Update the number of nibbles processed up till this point.
            if is_s {
                // Calculate the number of bytes
                let key_len = ext.ext_key_len(meta, 0);
                // Calculate the number of nibbles
                let num_nibbles = get_num_nibbles(meta, &mut cb.base, key_len.expr(), ext.is_odd());
                // Make sure the nibble counter is updated correctly
                let nibbles_count_prev = ifx! {not!(ext.is_below_account(meta)), not_first_level.expr() => {
                    ext.nibbles_counter().prev()
                }};
                require!(ext.nibbles_counter() => nibbles_count_prev.expr() + num_nibbles.expr() + 1.expr());
            }
        });

        ExtensionNodeConfig {
            _marker: PhantomData,
        }
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        mpt_config: &MPTConfig<F>,
        pv: &mut ProofValues<F>,
        row: &MptWitnessRow<F>,
        offset: usize,
        is_s: bool,
    ) {
        if pv.is_extension_node {
            if is_s {
                // [228,130,0,149,160,114,253,150,133,18,192,156,19,241,162,51,210,24,1,151,16,
                // 48,7,177,42,60,49,34,230,254,242,79,132,165,90,75,249]

                // One nibble:
                // [226,16,160,172,105,12...
                // Could also be non-hashed branch:
                // [223,16,221,198,132,32,0,0,0,1,198,132,32,0,0,0,1,128,128,128,128,128,128,
                // 128,128,128,128,128,128,128,128,128]

                // [247,160,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                // 213,128,194,32,1,128,194,32,1,128,128,128,128,128,128,128,128,128,128,128,
                // 128,128] [248,58,159,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                // 0,0,0,0,0,0,0,0,0,0,0,0,217,128,196,130,32,0,1,128,196,130,32,0,1,128,128,
                // 128,128,128,128,128,128,128,128,128,128,128]

                // Intermediate RLC value and mult (after key)
                // to know which mult we need to use in c_advices.
                pv.acc_s = F::zero();
                pv.acc_mult_s = F::one();
                let len: usize;
                if row.get_byte(1) <= 32 {
                    // key length is 1
                    len = 2 // [length byte, key]
                } else if row.get_byte(0) < 248 {
                    len = (row.get_byte(1) - 128) as usize + 2;
                } else {
                    len = (row.get_byte(2) - 128) as usize + 3;
                }
                mpt_config.compute_acc_and_mult(
                    &row.bytes,
                    &mut pv.acc_s,
                    &mut pv.acc_mult_s,
                    0,
                    len,
                );

                // Final RLC value.
                pv.acc_c = pv.acc_s;
                pv.acc_mult_c = pv.acc_mult_s;
                let mut start = C_RLP_START + 1;
                let mut len = HASH_WIDTH + 1;
                if row.get_byte(C_RLP_START + 1) == 0 {
                    // non-hashed branch in extension node
                    start = C_START;
                    len = HASH_WIDTH;
                }
                mpt_config.compute_acc_and_mult(
                    &row.bytes,
                    &mut pv.acc_c,
                    &mut pv.acc_mult_c,
                    start,
                    len,
                );

                mpt_config
                    .assign_acc(region, pv.acc_s, pv.acc_mult_s, pv.acc_c, F::zero(), offset)
                    .ok();
            } else {
                // We use intermediate value from previous row (because
                // up to acc_s it's about key and this is the same
                // for both S and C).
                pv.acc_c = pv.acc_s;
                pv.acc_mult_c = pv.acc_mult_s;
                let mut start = C_RLP_START + 1;
                let mut len = HASH_WIDTH + 1;
                if row.get_byte(C_RLP_START + 1) == 0 {
                    // non-hashed branch in extension node
                    start = C_START;
                    len = HASH_WIDTH;
                }
                mpt_config.compute_acc_and_mult(
                    &row.bytes,
                    &mut pv.acc_c,
                    &mut pv.acc_mult_c,
                    start,
                    len,
                );

                mpt_config
                    .assign_acc(region, pv.acc_s, pv.acc_mult_s, pv.acc_c, F::zero(), offset)
                    .ok();
            }
            region
                .assign_advice(
                    || "assign key_rlc".to_string(),
                    mpt_config.accumulators.key.rlc,
                    offset,
                    || Value::known(pv.extension_node_rlc),
                )
                .ok();
        }
    }
}