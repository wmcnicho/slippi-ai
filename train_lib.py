import datetime
from typing import Iterator, Tuple
import embed
import os
import secrets

import numpy as np
import tensorflow as tf
import tree

import data
from learner import Learner
import utils
from typing import Union

def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  expt_dir = f'experiments/{get_experiment_tag()}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir

# necessary because our dataset has some mismatching types, which ultimately
# come from libmelee occasionally giving differently-typed data
# Won't be necessary if we re-generate the dataset.
embed_game = embed.make_game_embedding()

def sanitize_game(game: data.CompressedGame) -> data.CompressedGame:
  """Casts inputs to the right dtype and discard unused inputs."""
  gamestates = embed_game.map(lambda e, a: a.astype(e.dtype), game.states)
  return game._replace(states=gamestates)

def sanitize_batch(batch: data.Batch) -> data.Batch:
  return batch._replace(game=sanitize_game(batch.game))

class TrainManager:

  def __init__(
      self,
      learner: Learner,
      data_source: Iterator[Tuple[data.Batch, float]],
      step_kwargs={},
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.total_frames = 0
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self) -> dict:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
      batch = sanitize_batch(batch)
    with self.step_profiler:
      stats, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    self.total_frames += np.sum(batch.game.counts + 1)
    stats.update(
        epoch=epoch,
        total_frames=self.total_frames,
    )
    return stats

def log_stats(ex, stats, step=None, sep='.'):
  def log(path, value):
    if isinstance(value, tf.Tensor):
      value = value.numpy()
    if isinstance(value, np.ndarray):
      value = value.mean()
    key = sep.join(map(str, path))
    ex.log_scalar(key, value, step=step)
  tree.map_structure_with_path(log, stats)

class LearningRateCycler:
  """
  Implementaion of 'triangular' cyclical learning rate policy defined in
  https://arxiv.org/pdf/1506.01186.pdf
  """
  DEFAULT_CONFIG = dict(
    base_lr=0.00005,
    max_lr=0.0005,
    step_size=2000,
    disable=False,
  )

  def __init__(self, base_lr:float, max_lr:float, step_size:float, disable:bool):
    self.base_lr = base_lr
    self.max_lr = max_lr
    self.step_size = step_size
    self.disable = disable
      
  def get_rate(self, total_iter_count) -> Union[float, None]:
    """ 
      Returns the updated learning rate given the total number of iterations and
      position in the triangular cycle.
      if disabled will return None
    """
    if self.disable:
      return None
    cycle = np.floor(1+total_iter_count/(2*self.step_size))
    x = np.abs(total_iter_count/self.step_size - 2*cycle + 1)
    lr = self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))
    return lr

  def get_rate_form_epoch_ratio(self, epoch:float) -> Union[float, None]:
    """
      Given the ratio of completed epoch, returns the updated learning rate in a triangular cycle.
      if disabled will return None
    """
    if self.disable:
      return None
    x = np.abs(1 - (epoch % 6) / 3) # This 6 could be parametarized
    lr = x * self.base_lr + (1 - x) * self.max_lr
    return lr
