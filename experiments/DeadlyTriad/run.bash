run_vi () {
    python experiments/runner/tabular_gridcraft.py --epochs 10 --steps_per_epoch 100 --evaluation_interval 1 --log_interval 1 --exp_name DeadlyTriad --solver OVI --task_name DS:$1-NS:$2 --lr 0.1 --diag_scale $1 --num_samples $2 --use_oracle_visitation False --no_approx False
}


declare -A pids

for ds in 10 1e6; do
    for ns in 10 1e6; do
        run_vi $ds $ns &
        pids[vi$ds$ns]=$!
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    echo $pid
done


python experiments/plot.py results/DeadlyTriad -x Steps
cp results/DeadlyTriad/ReturnPolicy.png experiments/DeadlyTriad/Performance.png