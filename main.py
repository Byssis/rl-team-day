import gymnasium as gym
import numpy as np
import pylab
from src.agent import DQNAgent

EPISODES = 1000

def train():
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  print(f"state_size: {state_size}  action_size: {action_size}")
  agent = DQNAgent(state_size, action_size)
  scores, episodes = [], []
  for e in range(EPISODES):
    done = False
    score = 0
    state, _ = env.reset()
    state = np.reshape(state, (1, state_size))

    while not done:
      if agent.render and e % 20 == 0:
        env.render()

      action = agent.get_action(state)
      next_state, reward, done, info, _ = env.step(action)
      next_state = np.reshape(next_state, (1, state_size))
      agent.append_sample(state, action, reward, next_state, done)
      score += reward
      state = next_state
      agent.train_model()

    if e % agent.target_update_frequency == 0:
      agent.update_target_model()
    
    scores.append(score)
    episodes.append(e)

    print(f"episode: {e}  score: {score}  memory length: {len(agent.memory)}")

    if np.mean(scores[-min(50, len(scores)):]) > 195:
      agent.check_solve = True
      break
  agent.store_model("save_model/cartpole_dqn.h5")
  plot_data(episodes, scores)

def test():
  env = gym.make('CartPole-v2', render_mode='human')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  agent.load_model("save_model/cartpole_dqn.h5")
  agent.render = True
  for e in range(EPISODES):
    done = False
    score = 0
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done:
      env.render()
      action = agent.get_test_action(state)
      next_state, reward, done, info, _ = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])
      score += reward
      state = next_state
      if done:
        print(f"episode: {e}  score: {score}")

def plot_data(episodes, scores):
  pylab.figure(0)
  pylab.plot(episodes, scores, 'b')
  pylab.xlabel("Episodes")
  pylab.ylabel("Score")
  pylab.savefig("results/scores.png")

if __name__ == "__main__":
  train()
  test()


