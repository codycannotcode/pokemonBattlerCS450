import os

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen8EnvSinglePlayer, SimpleHeuristicsPlayer, MaxBasePowerPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_8_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )
    
    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

# Creates an A2C model, trains it, and then logs the results into the a tensorboard directory.
# after the model is finished training, it is saved into the file a2c_pokemon_model.zip
# env is a gymnasium environment, should input the SimpleRlPlayer here
# during or after training, you can view the model's progress with tensorboard
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

# You can probably change the model type very easily with this code
def a2c_train(env, total_timesteps):
    log_dir = './a2c_pokemon_tensorboard'

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_env = SimpleRLPlayer(opponent=SimpleHeuristicsPlayer(battle_format='gen8randombattle'))
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # evaluate final time after training finishes 
    eval_callback.on_training_end()

    model.save('a2c_pokemon_model')

# quick copy + paste, maybe make modular later (michael did already i think)
def ppo_train(env, total_timesteps):
    log_dir = './ppo_pokemon_tensorboard'

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_env = SimpleRLPlayer(opponent=MaxBasePowerPlayer(battle_format='gen8randombattle'))
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=log_dir,
        eval_freq=25_000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # evaluate final time after training finishes 
    eval_callback.on_training_end()

    model.save('ppo_pokemon_model')

# evaluates the model by seeing how many times it can win against the SimpleHeuristicBot
# battles is the number of battles to evaluate with
def a2c_evaluation(env: SimpleRLPlayer, model: A2C, battles):
    finished_battles = 0

    env.reset_battles()
    obs, _ = env.reset()
    
    while finished_battles < battles:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if done:
            finished_battles += 1
            print(f"EPISODE {finished_battles}")
            obs, _ = env.reset()
            if finished_battles >= battles:
                break

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (env.n_won_battles, battles)
    )


NB_TRAINING_STEPS = 25_000
TEST_EPISODES = 100
GEN_8_DATA = GenData.from_gen(8)

if __name__ == "__main__":
    simpleHeuristicsPlayer = SimpleHeuristicsPlayer(battle_format='gen8randombattle')
    maxBasePowerPlayer = MaxBasePowerPlayer(battle_format='gen8randombattle')

    env = SimpleRLPlayer(opponent=maxBasePowerPlayer)

    # train the bot
    # ppo_train(env, NB_TRAINING_STEPS)

    # evaluate the bot
    model = PPO.load('ppo_pokemon_model')
    a2c_evaluation(env, model, TEST_EPISODES)

