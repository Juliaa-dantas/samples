import gym
import collections
from tensorboardX import SummaryWriter

# collection module is used instead of python's general purpose built functions like lists or dics,....
# tensorboard is a tool for providing the measurement and visualisation needed during the ML workflow

env_name = 'FrozenLake8x8-v0'
gamma = 0.9
test_episode = 20

class agent:
    def __init__(self):
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
# defaultdict means that if a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry is created.
# counter: It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values.

# introduction a function for gathering random experience from the environment to update reward & transition tables
# exploration mode
    def play_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward  # key : value====> ex: (0, 2, 8): 0.0
            self.transits[(self.state, action)][new_state] += 1  # ex: (0, 0): Counter({0: 14521, 8: 7179}), (0, 1): Counter({8: 51, 0: 44, 1: 44})
            if done == True:
                self.state = self.env.reset()
            else:
                self.state = new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if (best_value is None) or (best_value < action_value):
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        tot_reward = 0
        state = env.reset()
        while True:
            action = self.select_action(state)  # choosing the best action
            new_state, reward, done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] +=1
            tot_reward += reward
            if done == True:
                break
            state = new_state
        return tot_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tg_state, count in target_counts.items():
                    key = (state, action, tg_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tg_state)
                    val = reward + gamma * self.values[(tg_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value

# ---------------------training loop----------------------
if __name__ == "__main__":
    test_env = gym.make(env_name)
    agent = agent()
    writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = 0
    while True:
        iter_no += 1
        agent.play_random_steps(100)
        agent.value_iteration()
# -------------------- Test -------------
        reward = 0
        for _ in range(test_episode):
            reward += agent.play_episode(test_env)
        reward /= test_episode
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f ----> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.8:
            print("Solved in %d iterations" % iter_no)
            break
        writer.close()
