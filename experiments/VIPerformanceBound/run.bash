run_vi () {
    python experiments/runner/tabular_gridcraft.py --epochs 10 --steps_per_epoch 10 --evaluation_interval 1 --log_interval 1 --noise_scale 0.5 --exp_name PerformanceBound --solver OVI --task_name VI --lr 1.0
}

run_cvi () {
    python experiments/runner/tabular_gridcraft.py --epochs 10 --steps_per_epoch 10 --evaluation_interval 1 --log_interval 1 --noise_scale 0.5 --exp_name PerformanceBound --solver OCVI --task_name CVI-er:$1-kl:$2 --er_coef $1 --kl_coef $2 --lr 1.0
}

declare -A pids

run_vi &
pids[vi]=$!
for er in 0.0 1.0; do
    for kl in 0.0 1.0; do
        run_cvi $er $kl &
        pids[cvi$er$kl]=$!
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    echo $pid
done


python experiments/plot.py results/PerformanceBound -x Steps
cp results/PerformanceBound/ReturnPolicy.png experiments/VIPerformanceBound/Performance.png