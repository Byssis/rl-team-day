import os
import argparse
import gymnasium as gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from src.agent import DQNAgent

def train(args):
  number_of_episodes = args.episodes
  print(f"Training for {number_of_episodes} episodes")
  
  env = gym.make(args.env)
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  dir_path = args.save_dir + "/" + args.env
  os.makedirs(dir_path, exist_ok=True)

  agent = DQNAgent(
    state_size, 
    action_size, 
    discount_factor=args.gamma, 
    learning_rate=args.learning_rate, 
    epsilon=args.epsilon, 
    batch_size=args.batch_size, 
    memory_size=args.memory_size, 
    train_start=args.train_start, 
  )
  if args.load_model:
    agent.load_model(args.load_model)

  test_states_number = args.test_state_number

  test_states = np.zeros((test_states_number, state_size))
  max_q = np.zeros((number_of_episodes, test_states_number))
  max_q_mean = np.zeros((number_of_episodes, 1))

  start_time = datetime.now()
  done = True
  for i in range(test_states_number):
    if done:
      done = False
      state, _ = env.reset()
      # state = np.reshape(state, (1, state_size))
      test_states[i] = state
    else:
      action = random.randrange(action_size)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      # next_state = np.reshape(next_state, (1, state_size))
      test_states[i] = state
      state = next_state

  if args.render:
    env = gym.make(args.env, render_mode='human') 

  scores, episodes, means = [], [], []
  for e in range(number_of_episodes):
    done = False
    score = 0
    state, _ = env.reset() 
    #Compute Q values for plotting
    tmp = agent.policy.predict(test_states, verbose=0)
    max_q[e][:] = np.max(tmp, axis=1)
    max_q_mean[e] = np.mean(max_q[e][:])
    
    while not done:
      action = agent.get_action(state)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      #Save sample <s, a, r, s'> to the replay memory
      agent.append_sample(state, action, reward, next_state, terminated)
      #Training step
      agent.optimize()
      score += reward
      state = next_state 

    if e % args.target_update_frequency == 0:
      agent.update_target_model()

    if e % args.save_every == 0 and e > 0:
      file_name = dir_path + f"/check_point_model_dqn_{e}.h5"
      print(f"Saving model to {file_name}")
      agent.save_model(file_name)

    #Plot the play time for every episode
    scores.append(score)
    episodes.append(e)
    mean = np.mean(scores[-min(30, len(scores)):])
    means.append(mean)
    print(f"episode: {e}  score: {score}  memory length: {len(agent.memory)} mean: {mean} q_mean: {max_q_mean[e]}, epsilon: {agent.epsilon}")
    # plot_scores(scores, means, max_q_mean[:e + 1])
    # stop training
    if args.goal and  mean >= args.goal:
      print("solved after", e-100, "episodes")
      break
      

  end_time = datetime.now()
  print('Duration: {}'.format(end_time - start_time))
  agent.save_model(dir_path + "/model_dqn_final.h5")
  # save args
  if dir_path:
    with open(dir_path + "/args.txt", "w") as f:
      f.write(str(args))
  plot_scores(scores, means, max_q_mean[:e + 1], True, dir_path)

def test(args):
  episodes = args.episodes
  env = gym.make(args.env, render_mode='human')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  if args.load_model:
    agent.load_model(args.load_model)
  else:
    raise Exception("No model to test")
  
  for e in range(episodes):
    done = False
    score = 0
    state, _ = env.reset()
    while not done:
      env.render()
      action = agent.get_action(state)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      score += reward
      state = next_state
    if done:
      print(f"episode: {e}  score: {score}")
  

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_scores(scores, means, q_value, show_result=False, result_dir=None):
  plt.figure(1)
  if show_result:
    plt.title('Result')
  else:
    plt.clf()
    plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel('Score')
  plt.plot(scores)
  if result_dir:
    plt.savefig(result_dir + "/scores.png")
  # Take 100 episode averages and plot them too
  plt.plot(means)
  plt.figure(2)
  plt.clf()
  plt.title('Q Value')
  plt.xlabel('Episode')
  plt.ylabel('Q Value')
  plt.plot(q_value)
  if result_dir:
    plt.savefig(result_dir + "/qvalues.png")
  plt.pause(0.001)  # pause a bit so that plots are updated
  if is_ipython:
    if not show_result:
      display.display(plt.gcf())
      display.clear_output(wait=True)
    else:
      display.display(plt.gcf())



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='RL Training')
  parser.add_argument('--env', default='CartPole-v1', type=str, help='gym environment')
  parser.add_argument('--render', default=False, type=bool, help='render the environment')
  parser.add_argument('--train', default=True, type=bool, help='train the agent')
  parser.add_argument('--test', default=False, type=bool, help='test the agent')
  parser.add_argument('--load_model', default=None, type=str, help='load the saved model from a specified file')
  parser.add_argument('--save_every', default=50, type=int, help='save the model every specified episodes')
  parser.add_argument('--save_dir', default="results", type=str, help='directory to save the results')
  parser.add_argument('--episodes', default=10000, type=int, help='number of episodes to run')
  parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
  parser.add_argument('--epsilon', default=0.02, type=float, help='initial epsilon value for epsilon-greedy exploration')
  parser.add_argument('--batch_size', default=32, type=int, help='batch size for experience replay')
  parser.add_argument('--train_start', default=1000, type=int, help='start training after specified number of episodes')
  parser.add_argument('--memory_size', default=10000, type=int, help='size of the replay memory')
  parser.add_argument('--target_update_frequency', default=1, type=int, help='frequency of updating target network')
  parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')
  parser.add_argument('--test_state_number', default=10000, type=int, help='number of states to test')
  parser.add_argument('--goal', default=195, type=int, help='goal score to be achieve')

  args = parser.parse_args()
  print(args)
  if args.train:
    train(args)
  if args.test:
    test(args)










