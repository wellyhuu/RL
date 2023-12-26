import torch
from torch.nn import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import env

env = gym.make('env_name-v0', size = 10)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ... (your existing code)

def plot(rewards, epsilon_decay) :
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(epsilon_decay, color=color, label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Total Reward and Epsilon Decay - Q-Learning')
    plt.grid(axis='x', color='0.80')
    plt.show()

def Q_value_initialize(state, action, type=0):
    if type == 1:
        return np.ones((state, action))
    elif type == 0:
        return np.zeros((state, action))
    elif type == -1:
        return np.random.random((state, action))
'''
def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample() 

    return action
'''
def normalize(list): # you can use this to normalize your plot values
    xmin = min(list) 
    xmax = max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list 

def Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = 10 * 10, env.action_space.n
    Q_net = QNetwork(n_states, n_actions)
    target_Q_net = QNetwork(n_states, n_actions)
    optimizer = optim.Adam(Q_net.parameters(), lr=alpha)

    timestep_reward = []
    epsilon_values = []
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    total_reward = 0

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s = env.reset()
        s = s['agent'][0] * 10 + s['agent'][1]
        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            env.test()
            a = epsilon_greedy(Q_net, epsilon_threshold, s, n_actions)
            s_, reward, terminated, _, died = env.step(a)
            s_ = s_['agent'][0] * 10 + s_['agent'][1]

            total_reward += reward
            a_next = epsilon_greedy(Q_net, epsilon_threshold, s_, n_actions, explore=False)
            target = reward + gamma * target_Q_net(torch.tensor([s_]).float()).max().item()

            if terminated or died == 1:
                loss = nn.MSELoss()(Q_net(torch.tensor([s]).float())[a], torch.tensor([reward]).float())
            else:
                loss = nn.MSELoss()(Q_net(torch.tensor([s]).float())[a], torch.tensor([target]).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s, a = s_, a_next

            if terminated or died == 1:
                s = env.reset()
                s = s['agent'][0] * 10 + s['agent'][1]

        timestep_reward.append(total_reward)

        print(f"Episode: {episode + 1}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, epsilon_values)
    return timestep_reward, Q_net

def epsilon_greedy(Q_net, epsilon, s, n_actions, explore=True):
    if explore and np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return Q_net(torch.tensor([s]).float()).argmax().item()

# ... (your existing code)

def test_agent(Q, n_tests=1, delay=0.3, max_steps_test=100):
    env = gym.make('env_name-v0', render_mode="human", size=10)  # Change size to 15 for your MazeEnv

    for testing in range(n_tests): 
        print(f"Test #{testing}")
        s = env.reset()
        t = 0
        while t < max_steps_test:
            agent_position = s['agent']
            state_index = agent_position[0] * 10 + agent_position[1]
            t += 1
            time.sleep(delay)
            env.test()
            a = np.argmax(Q[state_index]).item()
            print(f"Chose action {a} for state {agent_position}")
            s, reward, terminated, _, died = env.step(a)
            
            print()

            if terminated or died == 1:
                print("Finished!", reward)
                time.sleep(delay)
                break


if __name__ == "__main__":
    alpha = 0.3  # learning rate
    gamma = 0.95  # discount factor
    episodes = 500
    max_steps = 1500

    epsilon = 0.01  # epsilon greedy exploration-exploitation (higher more random)
    EPS_START = 1
    EPS_END = 0.001
    timestep_reward, Q = Qlearning(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2
    )

    # Test policy (no learning)
    print(f"Q values:\n{Q}\nTesting now:")
    number_of_tests = 5
    test_agent(Q, number_of_tests)