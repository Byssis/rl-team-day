from dataclasses import dataclass
from numpy import ndarray
from typing import List

@dataclass
class Transition:
  state: ndarray
  action: int
  next_state: ndarray
  reward: float

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory: List[Transition] = list()
    self.position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
        self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)