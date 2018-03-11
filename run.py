from a2c import Model
from policies import PolicyFullyConnected
from utilities import global_seed
from runner import Runner
from utilities import init_env

def main():
    global_seed()

    map_name = "4x4"
    env = init_env(map_name)

    model = Model(
        policy=PolicyFullyConnected,
        observation_space = env.observation_space,
        action_space = env.action_space,
        learning_rate = 1e-3,
        nsteps=3,
        decay=0.99
    )

    runner = Runner(
        env = env,
        model = model,
        num_steps= 3,
        discount_rate = 0.99,
        save_summary_steps=1000,
        performance_num_episodes=100
    )

    time_steps = 3000

    for _ in range(time_steps):
        runner.run()


"""
1000   0.01
2000   0.0
3000   0.0
4000   0.15
5000   0.9
6000   0.94
7000   0.98
8000   1.0
9000   1.0
"""
if __name__ == '__main__':
    main()