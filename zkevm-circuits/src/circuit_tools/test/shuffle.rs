use std::{marker::PhantomData};

use eth_types::Field;
use gadgets::util::Scalar;
use halo2_proofs::{
    plonk::{Circuit, ConstraintSystem, Advice, Fixed, Column, FirstPhase, Challenge, Error, SecondPhase, Selector, Expression}, 
    circuit::{SimpleFloorPlanner, Layouter, layouter, Value},
    poly::Rotation,
};
use rand::RngCore;

use crate::circuit_tools::{constraint_builder:: ConstraintBuilder, cell_manager::{CellType, CellManager, Cell}};


#[derive(Clone)]
pub struct ShuffleConfig<const W: usize, const H: usize> {
    q_shuffle: Column<Fixed>,
    original: [Column<Advice>; W],
    shuffled: [Column<Advice>; W],
}

#[derive(Clone, Copy, Debug, num_enum::Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ShuffleCells {
    Original,
    Shuffled,
    #[num_enum(default)]
    Storage,
}

impl CellType for ShuffleCells{
    fn byte_type() -> Option<Self> {
        unimplemented!()
    }
    fn storage_for_phase(phase: u8) -> Self {
        match phase {
            0 => ShuffleCells::Storage,
            _ => unreachable!()
        }
    }
}


impl<const W: usize, const H: usize> ShuffleConfig<W, H> {
    pub fn new<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {        
        let q_shuffle = meta.fixed_column();
        let cm = CellManager::new(
            meta,
            vec![
                (ShuffleCells::Original, W, 1, false),
                (ShuffleCells::Shuffled, W, 1, false),
                (ShuffleCells::Storage, 10, 1, false),
            ],
            0,
            H
        );
        let original = array_init::from_iter(
            cm.get_typed_columns(ShuffleCells::Original)
            .iter()
            .map(|c| c.column.clone())
        ).unwrap(); 
        let shuffled = array_init::from_iter(
            cm.get_typed_columns(ShuffleCells::Shuffled)
            .iter()
            .map(|c| c.column.clone())
        ).unwrap(); 

        let mut cb: ConstraintBuilder<F, ShuffleCells> =  ConstraintBuilder::new(10 , Some(cm), None);

        meta.create_gate("Shuffle", |meta| {
            // Rest of the gate we handle in macro
            circuit!([meta, cb], {
                ifx!(f!(q_shuffle) => {
                    let original = cb.query_cells_dyn(ShuffleCells::Original, W * H);
                    let shuffled = cb.query_cells_dyn(ShuffleCells::Shuffled, W * H);
    
                    for (i, o) in original.iter().enumerate() {
                        let mut dest_set = Vec::new();
                        for (j, s) in shuffled.iter().enumerate() {
                            if i == j { continue; }
                            dest_set.push(s.expr());
                        }
                        require!(o.expr() => dest_set);
                    }
                });

            });
            cb.build_constraints()
                .into_iter()
                .map(|(name, constraint)| 
                    cb.store_expression(name, constraint, ShuffleCells::Storage)
                ).collect::<Vec<Expression<F>>>()
        });
        
        ShuffleConfig { 
            q_shuffle,
            original, 
            shuffled
        }
    }
}

#[derive(Clone, Default)]
struct ShuffleCircuit<F: Field, const W: usize, const H: usize> {
    original: Value<[[F; H]; W]>,
    shuffled: Value<[[F; H]; W]>,
}

impl<F: Field, const W: usize, const H: usize> Circuit<F> for ShuffleCircuit<F, W, H> {
    type Config = ShuffleConfig<5, 6>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // dummy column for phase1 challange
        meta.advice_column_in(FirstPhase);
        ShuffleConfig::new(meta)        
    }

    fn synthesize(
        &self, 
        config: Self::Config, 
        mut layouter: impl Layouter<F>
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.assign_region(|| "Shuffle", |mut region| {
            for offset in (0..H) {
                
            }
            todo!()
        })
    }
}

#[test]
fn test() {

    use halo2_proofs::{ dev::MockProver, halo2curves::bn256::Fr};

    let circuit = ShuffleCircuit::<Fr, 3, 4>::default();
    let prover = MockProver::<Fr>::run(6, &circuit, vec![]).unwrap();
    // prover.assert_satisfied_par();
}

fn rand_2d_array<F: Field, R: RngCore, const W: usize, const H: usize>(rng: &mut R) -> [[F; H]; W] {
    [(); W].map(|_| [(); H].map(|_| F::random(&mut *rng)))
}

fn shuffled<F: Field, R: RngCore, const W: usize, const H: usize>(
    original: [[F; H]; W],
    rng: &mut R,
) -> [[F; H]; W] {
    let mut shuffled = original;

    for row in (1..H).rev() {
        for column in shuffled.iter_mut() {
            let rand_row = (rng.next_u32() as usize) % row;
            column.swap(row, rand_row);
        }
    }

    shuffled
}