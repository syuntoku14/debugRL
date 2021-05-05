if [ $1 = "oracle" ]; then
    for algo in OVI OCVI OMVI; do
        python examples/tabular_gridcraft.py --solver $algo --epochs 10 --steps_per_epoch 10 --evaluation_interval 1 --log_interval 1 --exp_name OracleGridCraft 
    done
    python examples/plot.py results/OracleGridCraft -x Steps
elif [ $1 = "exact" ]; then
    for algo in EFVI EFCVI EFMVI; do
        python examples/tabular_gridcraft.py --solver $algo --epochs 10 --steps_per_epoch 10 --proj_iters 20 --evaluation_interval 1 --log_interval 1 --exp_name ExactFittedGridCraft
    done
    python examples/plot.py results/ExactFittedGridCraft -x Steps
elif [ $1 = "sampling" ]; then
    for algo in SVI SCVI SMVI; do
        python examples/tabular_gridcraft.py --solver $algo --epochs 10 --steps_per_epoch 1500 --evaluation_interval 10 --log_interval 1 --exp_name SamplingGridCraft
    done
    python examples/plot.py results/SamplingGridCraft -x Steps
elif [ $1 = "samplingfitted" ]; then
    for algo in SFVI SFCVI SFMVI; do
        python examples/tabular_gridcraft.py --solver $algo --epochs 10 --steps_per_epoch 1500 --evaluation_interval 10 --log_interval 1 --exp_name SamplingFittedGridCraft
    done
    python examples/plot.py results/SamplingFittedGridCraft -x Steps
fi