"""Adapter for DM to Gym environments.

This is forked from google3/third_party/py/bsuite/utils/gym_wrapper.py
"""

from typing import Any, Dict, Optional, Tuple, Union

import dm_env
from dm_env import specs
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# OpenAI gym step format = obs, reward, is_finished, is_truncated, other_info
_GymTimestep = Tuple[
    np.ndarray | dict[str, Any], float, bool, bool, Dict[str, Any]
]


class DMToGymWrapper(gym.Env):
  """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, env: dm_env.Environment):
    self._env = env  # type: dm_env.Environment
    self._last_observation = None  # type: Optional[np.ndarray]
    self.viewer = None

  def reset(
      self, seed: Any = None
  ) -> tuple[np.ndarray | dict[str, Any], dict[str, Any]]:
    """Resets the environment.

    Args:
      seed: Random seed.

    Returns:
      Observation post reset.
    """
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
    timestep = self._env.reset()
    self._last_observation = timestep.observation
    return timestep.observation, {}

  def step(self, action: int) -> _GymTimestep:
    timestep = self._env.step(action)
    self._last_observation = timestep.observation
    reward = timestep.reward or 0.0
    return timestep.observation, reward, timestep.last(), False, {}

  def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
    if self._last_observation is None:
      raise ValueError('Environment not ready to render. Call reset() first.')

    if mode == 'rgb_array':
      return self._last_observation

    if mode == 'human':
      if self.viewer is None:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=g-import-not-at-top
        from gym.envs.classic_control import rendering

        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self._last_observation)
      return self.viewer.isopen

  @property
  def action_space(self) -> spaces.Space:
    return spec2space(self._env.action_spec())

  @property
  def observation_space(self) -> spaces.Space:
    return spec2space(self._env.observation_spec())

  @property
  def reward_range(self) -> Tuple[float, float]:
    reward_spec = self._env.reward_spec()
    if isinstance(reward_spec, specs.BoundedArray):
      return reward_spec.minimum, reward_spec.maximum
    return -float('inf'), float('inf')

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)


def spec2space(spec: specs, name: Optional[str] = None) -> gym.Space:
  """Converts a dm_env spec to a Gymnasium space.

  Args:
    spec: dm_env spec.
    name: Optional name as key for the returned gym.Space.Dict in case a dict is
      provided.

  Returns:
    A gym.spaces object

  Raises:
    ValueError: If the provided spec is not one of the supported types.
  """
  if isinstance(spec, specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, specs.BoundedArray):
    return spaces.Box(
        low=float(spec.minimum),
        high=float(spec.maximum),
        shape=spec.shape,
        dtype=spec.dtype,
    )
  elif isinstance(spec, specs.Array):
    return spaces.Box(
        low=-float('inf'), high=float('inf'), shape=spec.shape, dtype=spec.dtype
    )
  elif isinstance(spec, dict):
    return spaces.Dict(
        {key: spec2space(value, name) for key, value in spec.items()}
    )
  else:
    raise ValueError('Unexpected gym spec: {}'.format(spec))


def space2spec(
    space: gym.Space, name: Optional[str] = None
) -> specs.Array | dict[str, Any]:
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.

  Raises:
    ValueError: If the provided space is not one of the supported types.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name,
    )

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name,
    )

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec,
        name=name,
    )

  elif isinstance(space, spaces.Tuple):
    return tuple(space2spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {key: space2spec(value, name) for key, value in space.spaces.items()}

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


class GymToDMWrapper(dm_env.Environment):
  """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

  def __init__(self, gym_env: gym.Env):
    self.gym_env = gym_env
    # Convert gym action and observation spaces to dm_env specs.
    self._observation_spec = space2spec(
        self.gym_env.observation_space, name='observations'
    )
    self._action_spec = space2spec(self.gym_env.action_space, name='actions')
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    observation = self.gym_env.reset()
    return dm_env.restart(observation)

  def step(self, action: int) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()

    # Convert the gym step result to a dm_env TimeStep.
    observation, reward, done, info = self.gym_env.step(action)
    self._reset_next_step = done

    if done:
      is_truncated = info.get('TimeLimit.truncated', False)
      if is_truncated:
        return dm_env.truncation(reward, observation)
      else:
        return dm_env.termination(reward, observation)
    else:
      return dm_env.transition(reward, observation)

  def close(self) -> None:
    self.gym_env.close()

  def observation_spec(self) -> specs:
    """Returns the observation spec."""
    return self._observation_spec

  def action_spec(self) -> specs:
    """Returns the action spec."""
    return self._action_spec

  def reward_spec(self) -> dm_env.specs:
    """Returns the reward spec."""
    reward_range = self.gym_env.reward_range
    return specs.BoundedArray(
        minimum=reward_range[0],
        maximum=reward_range[1],
        dtype=np.float32,
        shape=(),
    )
