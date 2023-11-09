import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
# for mac m1/m2
from tensorflow.keras.optimizers.legacy import Adam


class DQNAgent:
  def __init__(self, state_size, action_size, discount_factor=0.4, learning_rate=5e-2, epsilon=0.4, batch_size=32, memory_size=10000, train_start=1000):
    #Get size of state and action
    self.state_size = state_size
    self.action_size = action_size
    # Discount factor for rewards
    self.discount_factor = discount_factor
    # Learning rate for the inner loop network
    self.learning_rate = learning_rate

    # Epsilon is the probability of selecting a random action
    self.epsilon = epsilon
    
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.train_start = train_start

    # Create memory buffer using deque
    self.memory = deque(maxlen=self.memory_size)

    self.policy = self.build_model()
    self.target_model = self.build_model()
    self.policy.summary()

    self.update_target_model()


  def build_model(self):
    # Neural Net for Deep-Q learning Model
    # Input is state, output is Q-value of each action
    # Should match the model complexity of the problem
    # TODO: Implement the model
    model = Sequential([
      Dense(16, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'), # Input layer
      Dense(self.action_size, activation='linear', kernel_initializer='he_uniform') # Output layer
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    return model

  def update_target_model(self):
    self.target_model.set_weights(self.policy.get_weights())

  def save_model(self, path):
    self.policy.save_weights(path)

  def load_model(self, path):
    self.policy.load_weights(path)
    self.update_target_model()

  def get_action(self, state):
    # TODO: Implement the exploration policy
    state = np.reshape(state, [1, self.state_size])
    return np.argmax(self.policy.predict(state, verbose=0))
  
  def get_acction_test(self, state):
    return np.argmax(self.policy.predict(state, verbose=0))

  # Save sample <s,a,r,s'> to the replay memory
  def append_sample(self, state, action, reward, next_state, done):
    state = np.reshape(state, [1, self.state_size]) 
    next_state = np.reshape(next_state, [1, self.state_size])
    self.memory.append((state, action, reward, next_state, done)) #Add sample to the end of the list

  def optimize(self):
    if len(self.memory) < self.train_start: #Do not train if not enough memory
      return
    batch_size = min(self.batch_size, len(self.memory))
    # Sample <s,a,r,s'> from replay memory
    mini_batch = random.sample(self.memory, batch_size)

    update_input = np.zeros((batch_size, self.state_size)) 
    update_target = np.zeros((batch_size, self.state_size)) 
    action, reward, done = [], [], [] 

    for i in range(self.batch_size):
      update_input[i] = mini_batch[i][0] # Allocate s(i) to the network input array from iteration i in the batch
      action.append(mini_batch[i][1]) # Store a(i)
      reward.append(mini_batch[i][2]) # Store r(i)
      update_target[i] = mini_batch[i][3] # Allocate s'(i) for the target network array from iteration i in the batch
      done.append(mini_batch[i][4])  #Store done(i)

    target = self.policy.predict(update_input, verbose=0) # Generate target values for training the inner loop network using the network model
    target_val = self.target_model.predict(update_target, verbose=0) # Generate the target values for training the outer loop target network

    for i in range(self.batch_size): #For every batch
      if done[i]:
        target[i][action[i]] = reward[i]
      else:
        target[i][action[i]] = reward[i] + self.discount_factor * np.max(target_val[i])

    # Train the inner loop network
    self.policy.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
       