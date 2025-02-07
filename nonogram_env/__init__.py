from gymnasium.envs.registration import register

register(
    id="nonogram_env/GridWorld-v0",
    entry_point="nonogram_env.envs:GridWorldEnv",
)
