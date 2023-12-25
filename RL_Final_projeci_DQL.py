import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import env

class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.layer1 = layers.Dense(64, activation='relu')
        self.layer2 = layers.Dense(num_actions)

    def call(self, state):
        x = self.layer1(state)
        return self.layer2(x)

env = gym.make('env_name-v0', size=10)

def epsilon_greedy(model, epsilon, state):
    if np.random.rand() > epsilon:
        q_values = model.predict(state)
        action = np.argmax(q_values[0]).item()
    else:
        action = env.action_space.sample()

    return action

def plot(rewards, epsilon_decay):
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(epsilon_decay, color=color, label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Total Reward and Epsilon Decay - Deep Q-Learning')
    plt.grid(axis='x', color='0.80')
    plt.show()

def test_agent(model, n_tests=1, delay=0.3, max_steps_test=100):
    env = gym.make('env_name-v0', render_mode="human", size=10)

    for testing in range(n_tests):
        print(f"Test #{testing}")
        s = env.reset()
        s = np.array([s['agent'][0] * 10 + s['agent'][1]], dtype=np.float32)
        t = 0
        while t < max_steps_test:
            agent_position = s['agent']
            state_index = agent_position[0] * 10 + agent_position[1]
            t += 1
            time.sleep(delay)
            env.test()
            a = np.argmax(model(s)).item()
            print(f"Chose action {a} for state {agent_position}")
            s, reward, terminated, _, died = env.step(a)

            print()

            if terminated or died == 1:
                print("Finished!", reward)
                time.sleep(delay)
                break

def Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests):
    n_states, n_actions = 10 * 10, env.action_space.n

    model = QNetwork(num_actions=n_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    loss_fn = tf.keras.losses.MeanSquaredError()

    timestep_reward = []
    epsilon_values = []
    epsilon_decay_rate = -np.log(EPS_END / EPS_START) / episodes
    total_reward = 0

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s = env.reset()
        s = np.array([s['agent'][0] * 10 + s['agent'][1]], dtype=np.float32)

        epsilon_threshold = EPS_START * np.exp(-epsilon_decay_rate * episode)
        epsilon_values.append(epsilon_threshold)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            env.test()
            with tf.GradientTape() as tape:
                q_values = model(tf.expand_dims(s, 0))  # Expand dimensions to create a batch
                a = epsilon_greedy(model, epsilon_threshold, s)
                s_, reward, terminated, _, died = env.step(a)
                s_ = np.array([s_['agent'][0] * 10 + s_['agent'][1]], dtype=np.float32)

                total_reward += reward
                a_next_q = model(tf.expand_dims(s_, 0))
                target_q = reward + gamma * tf.reduce_max(a_next_q)
                current_q = q_values  # Remove [0, a]
                loss = loss_fn(current_q, target_q)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            s = s_

            if terminated or died == 1:
                s = env.reset()
                s = np.array([s['agent'][0] * 10 + s['agent'][1]], dtype=np.float32)

        timestep_reward.append(total_reward)

        print(f"Episode: {episode + 1}, steps: {t}, Total reward: {total_reward}")

    plot(timestep_reward, epsilon_values)
    return timestep_reward, model

if __name__ == "__main__":
    alpha = 0.001
    gamma = 0.95
    episodes = 500
    max_steps = 1500

    epsilon = 0.01
    EPS_START = 1
    EPS_END = 0.001
    timestep_reward, model = Qlearning(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, n_tests=2
    )

    print(f"Testing now:")
    number_of_tests = 5
    test_agent(model, number_of_tests)
