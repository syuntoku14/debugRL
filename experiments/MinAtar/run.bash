run_minatar () {
    python experiments/runner/run_minatar.py --exp_name $2 --solver $1 --env $2 --seed $3 --save --device cuda
}

declare -A pids

for solver in DQN M-DQN; do
    for env in breakout; do # seaquest asterix freeway space_invaders; do
        for seed in {0..4}; do
            run_minatar $solver $env $seed  # &
        done
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    echo $pid
done
