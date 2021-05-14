run_vi () {
    python experiments/runner/tabular_gridcraft.py --epochs 10 --steps_per_epoch 100 --evaluation_interval 1 --log_interval 1 --exp_name DeadlyTriad-VI --solver OVI --task_name DS:$1-NS:$2 --lr 0.1 --diag_scale $1 --num_samples $2 --use_oracle_visitation False --no_approx False
}
run_cvi () {
    python experiments/runner/tabular_gridcraft.py --epochs 10 --steps_per_epoch 1000 --evaluation_interval 1 --log_interval 1 --exp_name DeadlyTriad-CVI --solver OCVI --task_name DS:$1-NS:$2 --lr 0.1 --diag_scale $1 --num_samples $2 --use_oracle_visitation False --no_approx False --kl_coef 0.01 --er_coef 0.01
}


declare -A pids

for ds in 10 1e6; do
    for ns in 10 1e6; do
        run_vi $ds $ns &
        pids[vi$ds$ns]=$!
        run_cvi $ds $ns &
        pids[cvi$ds$ns]=$!
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    echo $pid
done


python experiments/plot.py results/DeadlyTriad-VI -x Steps
python experiments/plot.py results/DeadlyTriad-CVI -x Steps
cp results/DeadlyTriad-VI/ReturnPolicy.png experiments/DeadlyTriad/VI-Performance.png
cp results/DeadlyTriad-CVI/ReturnPolicy.png experiments/DeadlyTriad/CVI-Performance.png