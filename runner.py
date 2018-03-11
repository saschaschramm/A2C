import numpy as np

class Runner:

    def __init__(self, env, model, num_steps, discount_rate, save_summary_steps, performance_num_episodes):
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.discount_rate = discount_rate
        self.save_summary_steps = save_summary_steps
        self.performance_num_episodes = performance_num_episodes
        self.observation = env.reset()
        self.step_count = 0

        self.episodic_rewards = []
        self.episodic_performance = []
        self.average_episodic_performance = 0
        self.episode = 0

    def discount(self, rewards, dones, discount_rate):
        discounted = []
        total_return = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                total_return = reward
            else:
                total_return = reward + discount_rate * total_return
            discounted.append(total_return)
        return np.asarray(discounted[::-1])

    def evaluate(self, reward, done):
        self.step_count += 1
        self.episodic_rewards.append(reward)
        if done:
            self.episodic_performance.append(sum(self.episodic_rewards))

            self.episodic_rewards = []
            self.episode += 1

        # summary
        if (self.step_count % self.save_summary_steps) == 0:
            print("{}   {}".format(self.step_count, self.average_episodic_performance))
            #self.file_writer.add_summary(0, self.step_count, self.average_episodic_performance)
            #self.model.save(0)

        # average episodic performance
        if len(self.episodic_performance) == self.performance_num_episodes:
            self.average_episodic_performance = sum(self.episodic_performance) / self.performance_num_episodes
            self.episodic_performance.pop(0)

    def run(self):
        batch_observations, batch_rewards, batch_actions, batch_values, batch_dones = [], [], [], [], []
        for _ in range(self.num_steps):
            action, value = self.model.predict(self.observation)

            batch_observations.append(self.observation)
            batch_actions.append(action)
            batch_values.append(value)

            self.observation, reward, done = self.env.step(action)

            self.evaluate(reward, done)
            batch_rewards.append(reward)
            batch_dones.append(done)

        next_value = self.model.predict_value(self.observation)[0]

        if batch_dones[-1] == 0:
            discounted_rewards = self.discount(batch_rewards + [next_value], batch_dones + [False], self.discount_rate)[:-1]
        else:
            discounted_rewards = self.discount(batch_rewards, batch_dones, self.discount_rate)

        self.model.train(batch_observations, discounted_rewards, batch_actions, batch_values)