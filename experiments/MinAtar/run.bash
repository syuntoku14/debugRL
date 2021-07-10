for solver in DQN M-DQN; do
    for env in breakout seaquest asterix freeway space_invaders; do
        for seed in {0..4}; do
            tsp python experiments/runner/run_minatar.py --exp_name $env --solver $solver --env $env --seed $seed --save --device cuda
        done
    done
done
