import gymnasium as gym


class SERLObsWrapper(gym.ObservationWrapper):
  """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Dict({
        "state": gym.spaces.flatten_space(self.env.observation_space["state"]),
        **(self.env.observation_space["images"]),
    })

  def observation(self, obs):
    obs = {
        "state": gym.spaces.flatten(
            self.env.observation_space["state"], obs["state"]
        ),
        **(obs["images"]),
    }
    return obs
