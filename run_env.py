from Environment.catch import *
from a2c import Model
from policies import PolicyFullyConnected
from utilities import print_score, global_seed, discount

def main():
    global_seed()
    learning_rate = 1e-3
    decay = 0.99
    discount_rate = 0.99
    nsteps = 3
    env = Catch(5)
    policy = PolicyFullyConnected
    total_steps = int(2e4)

    observation_space = env.observation_space
    action_space = env.action_space

    model = Model(policy=policy, observation_space=observation_space, action_space=action_space,
                  nsteps=nsteps, learning_rate=learning_rate, decay=decay)

    observation = env.reset()

    for _ in range(1, total_steps):
        batch_observations, batch_rewards, batch_actions, batch_values, batch_dones = [], [], [], [], []
        for n in range(nsteps):
            action, value = model.predict(observation)

            batch_observations.append(np.copy(observation))
            batch_actions.append(action)
            batch_values.append(value)

            next_observation, reward, done = env.step(action)
            print_score(reward, limit=10000)

            if done:
                next_observation = env.reset()

            batch_rewards.append(reward)
            batch_dones.append(done)
            observation = next_observation

        next_value = model.predict_value(observation)[0]

        if batch_dones[-1] == 0:
            discounted_rewards = discount(batch_rewards + [next_value], batch_dones + [False], discount_rate)[:-1]
        else:
            discounted_rewards = discount(batch_rewards, batch_dones, discount_rate)

        model.train(batch_observations, discounted_rewards, batch_actions, batch_values)

if __name__ == '__main__':
    main()