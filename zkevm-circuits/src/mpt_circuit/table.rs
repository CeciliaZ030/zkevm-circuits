
use eth_types::Field;
use halo2_proofs::plonk::{Column, Any};
use strum_macros::EnumIter;
use crate::{
    circuit_tools::cell_manager::TableType, 
    circuit_tools::table::LookupTable,
    table::KeccakTable
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, EnumIter)]
pub enum Table {
    Fixed,
    Tx,
    Rw,
    Bytecode,
    Block,
    Copy,
    Keccak,
    Exp,
}

impl TableType for Table {}

impl<F: Field> LookupTable<F, Table> for KeccakTable {
    
    fn get_type(&self) -> Table {
        Table::Keccak
    }

    fn columns(&self) -> Vec<Column<Any>> {
        vec![
            self.is_enabled.into(),
            self.input_rlc.into(),
            self.input_len.into(),
            self.output_rlc.into(),
        ]
    }

    fn annotations(&self) -> Vec<String> {
        vec![
            String::from("is_enabled"),
            String::from("input_rlc"),
            String::from("input_len"),
            String::from("output_rlc"),
        ]
    }

}