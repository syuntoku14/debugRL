epochs=10
steps_per_epoch=10000
run_cpi() {
    python experiments/runner/tabular_gridcraft.py --epochs $epochs --steps_per_epoch $steps_per_epoch --evaluation_interval 1 --log_interval 1 --noise_scale 0.5 --exp_name MonotonicPolicyImprovement --solver OCPI --task_name CPI
}
run_spi() {
    python experiments/runner/tabular_gridcraft.py --epochs $epochs --steps_per_epoch $steps_per_epoch --evaluation_interval 1 --log_interval 1 --noise_scale 0.5 --exp_name MonotonicPolicyImprovement --solver OCPI --task_name SPI --use_spi True
}
run_pi() {
    python experiments/runner/tabular_gridcraft.py --epochs $epochs --steps_per_epoch $steps_per_epoch --evaluation_interval 1 --log_interval 1 --noise_scale 0.5 --exp_name MonotonicPolicyImprovement --solver OCPI --task_name CPI-mix_rate:1e-3 --constant_mix_rate 1e-3
}

declare -A pids

run_cpi &
pids[cpi]=$!
run_spi &
pids[spi]=$!
run_pi &
pids[pi]=$!



# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    echo $pid
done


python experiments/plot.py results/MonotonicPolicyImprovement -x Steps
cp results/MonotonicPolicyImprovement/ReturnPolicy.png experiments/MonotonicPolicyImprovement/Performance.png
python experiments/MonotonicPolicyImprovement/check_mi.py results/MonotonicPolicyImprovement