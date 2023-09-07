//! Mock types and functions to generate GethData used for tests

use eth_types::{address, bytecode, bytecode::Bytecode, word, Address, Bytes, Word};
use ethers_signers::LocalWallet;
use lazy_static::lazy_static;
use rand::{random, SeedableRng};
use rand_chacha::ChaCha20Rng;
mod account;
mod block;
mod sha3;
pub mod test_ctx;
mod transaction;

pub(crate) use account::MockAccount;
pub(crate) use block::MockBlock;
pub use sha3::Sha3CodeGen;
pub use test_ctx::TestContext;
pub use transaction::{AddrOrWallet, MockTransaction, CORRECT_MOCK_TXS};

/// Mock block gas limit
pub const MOCK_BLOCK_GAS_LIMIT: u64 = 10_000_000_000_000_000;

lazy_static! {
    /// Mock 1 ETH
    pub static ref MOCK_1_ETH: Word = eth(1);
    /// Mock coinbase value
    pub static ref MOCK_COINBASE: Address =
        address!("0x00000000000000000000000000000000c014ba5e");
    /// Mock gasprice value
    pub static ref MOCK_GASPRICE: Word = Word::from(1u8);
    /// Mock BASEFEE value
    pub static ref MOCK_BASEFEE: Word = Word::zero();
     /// Mock GASLIMIT value
    pub static ref MOCK_GASLIMIT: Word = Word::from(0x2386f26fc10000u64);
    /// Mock chain ID value
    pub static ref MOCK_CHAIN_ID: Word = Word::from(1338u64);
    /// Mock DIFFICULTY value
    pub static ref MOCK_DIFFICULTY: Word = Word::from(0x200000u64);
    /// Mock accounts loaded with ETH to use for test cases.
    pub static ref MOCK_ACCOUNTS: Vec<Address> = vec![
        address!("0x000000000000000000000000000000000cafe111"),
        address!("0x000000000000000000000000000000000cafe222"),
        address!("0x000000000000000000000000000000000cafe333"),
        address!("0x000000000000000000000000000000000cafe444"),
        address!("0x000000000000000000000000000000000cafe555"),
    ];
    /// Mock EVM codes to use for test cases.
    pub static ref MOCK_CODES: Vec<Bytes> = vec![
        Bytes::from([0x60, 0x10, 0x00]), // PUSH1(0x10), STOP
        Bytes::from([0x60, 0x01, 0x60, 0x02, 0x01, 0x00]), // PUSH1(1), PUSH1(2), ADD, STOP
        Bytes::from([0x60, 0x01, 0x60, 0x02, 0x02, 0x00]), // PUSH1(1), PUSH1(2), MUL, STOP
        Bytes::from([0x60, 0x02, 0x60, 0x01, 0x03, 0x00]), // PUSH1(2), PUSH1(1), SUB, STOP
        Bytes::from([0x60, 0x09, 0x60, 0x03, 0x04, 0x00]), // PUSH1(9), PUSH1(3), DIV, STOP
        Bytes::from([0x30; 256]), // ADDRESS * 256
    ];
    /// Mock wallets used to generate correctly signed and hashed Transactions.
    pub static ref MOCK_WALLETS: Vec<LocalWallet> = {
        let mut rng = ChaCha20Rng::seed_from_u64(2u64);
        vec![
            LocalWallet::new(&mut rng),
            LocalWallet::new(&mut rng),
            LocalWallet::new(&mut rng),
    ]
    };
    /// Mock EVM bytecode for a deployed contract.
    /// PUSH1 0x20
    /// PUSH1 0
    /// PUSH1 0
    /// CALLDATACOPY
    /// PUSH1 0x20
    /// PUSH1 0
    /// RETURN
    ///
    /// bytecode: 0x6020600060003760206000F3
    ///
    /// // constructor
    /// PUSH12 0x6020600060003760206000F3
    /// PUSH1 0
    /// MSTORE
    /// PUSH1 0xC
    /// PUSH1 0x14
    /// RETURN
    ///
    /// bytecode: 0x6B6020600060003760206000F3600052600C6014F3
    pub static ref MOCK_DEPLOYED_CONTRACT_BYTECODE: Word = word!("6B6020600060003760206000F3600052600C6014F3");
}

/// Generate a [`Word`] which corresponds to a certain amount of ETH.
pub fn eth(x: u64) -> Word {
    Word::from(x) * Word::from(10u64.pow(18))
}

/// Express an amount of ETH in GWei.
pub fn gwei(x: u64) -> Word {
    Word::from(x) * Word::from(10u64.pow(9))
}

/// Holds the parameters for generating mock EVM bytecode for a contract call
pub struct MockCallBytecodeParams {
    /// The address to call with the generated bytecode
    pub address: Address,
    /// The data to be passed as arguments to the contract function.
    pub pushdata: Vec<u8>,
    /// The offset in memory where the return data will be stored.
    pub return_data_offset: usize,
    /// The size of the return data.
    pub return_data_size: usize,
    /// The length of the call data.
    pub call_data_length: usize,
    /// The offset in memory where the call data will be stored.
    pub call_data_offset: usize,
    /// The amount of gas to be used for the contract call.
    pub gas: u64,
    /// The instructions to be executed after the contract call.
    pub instructions_after_call: Bytecode,
}

/// Set default parameters for MockCallBytecodeParams
impl Default for MockCallBytecodeParams {
    fn default() -> Self {
        MockCallBytecodeParams {
            address: address!("0x0000000000000000000000000000000000000000"),
            pushdata: Vec::new(),
            return_data_offset: 0x00usize,
            return_data_size: 0x00usize,
            call_data_length: 0x00usize,
            call_data_offset: 0x00usize,
            gas: 0x1_0000u64,
            instructions_after_call: Bytecode::default(),
        }
    }
}

/// Generate random bytes for the specified size.
pub fn rand_bytes(size: usize) -> Vec<u8> {
    (0..size).map(|_| random()).collect::<Vec<u8>>()
}

/// Generate mock EVM bytecode that performs a contract call
pub fn generate_mock_call_bytecode(params: MockCallBytecodeParams) -> Bytecode {
    bytecode! {
        .op_mstore(
            0u64,
            Word::from_big_endian(&params.pushdata)
        )
        .op_call(
            params.gas,
            params.address,
            0u64,
            params.call_data_offset,
            params.call_data_length,
            params.return_data_size,
            params.return_data_offset,
        )
        .append(&params.instructions_after_call)
        STOP
    }
}
