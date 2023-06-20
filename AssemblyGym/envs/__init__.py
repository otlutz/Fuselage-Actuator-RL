import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FuselageActuators-v22',
    entry_point='AssemblyGym.envs.FuselageActuators.FuselageActuators_env_v22:FuselageActuatorsEnv',
    max_episode_steps=100,
    reward_threshold=100,
)