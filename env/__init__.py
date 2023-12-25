from gym.envs.registration import register

register(
    id = 'env_name-v0',
    entry_point = 'env.Mazeenv:MazeEnv'
)