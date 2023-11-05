import random
import numpy as np
from collections import deque
from keras.layers import Dense
# from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

class DQNAgent:
  def __init__(self, state_size, action_size):
    self._initialize_parameters()
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=self.memory_size)
    self.model = self._build_model()
    self.target_model = self._build_model()
    self.update_target_model()

  def _initialize_parameters(self):
    self.check_solve = False
    self.render = False
    self.discount_factor = 0.99
    self.learning_rate = 5e-4
    self.epsilon = 0.02
    self.batch_size = 64
    self.memory_size = 10000
    self.train_start = 1000
    self.target_update_frequency = 1
    self.test_state_no = 10000

  def _build_model(self):
    model = Sequential()
    model.add(Dense(16, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    print(model.summary())
    return model

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def get_action(self, state):
    if np.random.random() > self.epsilon:
      return np.argmax(self.model.predict(state, verbose=0))
    return np.random.randint(self.action_size)
  
  def get_test_action(self, state):
    return np.argmax(self.model.predict(state, verbose=0))
  
  def append_sample(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def store_model(self, path):
    self.model.save_weights(path)
  
  def load_model(self, path):
    self.model.load_weights(path)

  def train_model(self):
    if len(self.memory) < self.train_start:
      return
    mini_batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
    update_input = np.zeros((len(mini_batch), self.state_size))
    update_target = np.zeros((len(mini_batch), self.state_size))
    action, reward, done = [], [], []

    for i, (state, act, rew, next_st, dn) in enumerate(mini_batch):
      update_input[i] = state
      action.append(act)
      reward.append(rew)
      update_target[i] = next_st
      done.append(dn)

    target = self.model.predict(update_input, verbose=0)
    target_val = self.target_model.predict(update_target, verbose=0)

    for i in range(len(mini_batch)):
      if done[i]:
        target[i][action[i]] = reward[i]
      else:
        target[i][action[i]] = reward[i] + self.discount_factor * np.max(target_val[i])

    self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

