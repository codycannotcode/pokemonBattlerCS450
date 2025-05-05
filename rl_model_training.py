import os

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
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
                move.base_power / 250
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_8_DATA.type_chart
                ) / 4

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # IMPORTANT NOTE!!!!
        # making assumption that battle.team and battle.opponent return a fixed order for the sake of time
        # definitely fix this later

        opponent_revealed_mons = len(battle.opponent_team) / 6

        team_hp = [1.0] * 6
        team_base_stats = [[0] * 6 for _ in range(6)]
        team_levels = [1.0] * 6

        for i, mon in enumerate(battle.team.values()):
            team_hp[i] = mon.current_hp_fraction
            team_base_stats[i] = list(mon.base_stats.values())
            team_levels[i] = mon.level / 100
        team_base_stats = np.array(team_base_stats) / 255.0
        team_base_stats = team_base_stats.flatten()
        
        opponent_hp = [1.0] * 6
        opponent_base_stats = [[0] * 6 for _ in range(6)]
        opponent_levels = [1.0] * 6

        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_hp[i] = mon.current_hp_fraction
            opponent_base_stats[i] = list(mon.base_stats.values())
            opponent_levels[i] = mon.level / 100
        opponent_base_stats = np.array(opponent_base_stats) / 255.0
        opponent_base_stats = opponent_base_stats.flatten()

        # combine all information into a single vector
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent, opponent_revealed_mons],
                team_hp,
                team_base_stats,
                team_levels,
                opponent_hp,
                opponent_base_stats,
                opponent_levels,
            ]
        )

        assert final_vector.shape == self.observation_space.shape
        return final_vector

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30, status_value=0.5
        )
    
    def describe_embedding(self):
        low = np.concatenate([
            [0] * 4, # move base power
            [0] * 4, # damage multiplier
            [0, 0, 0],
            [0] * 6, # team hp
            [0] * 36, # team base stats
            [0] * 6, # team levels
            [0] * 6, # opponent hp
            [0] * 36, # opponent base stats
            [0] * 6, # opponent levels
        ])
        high = np.concatenate([
            [1] * 4, # move base power
            [1] * 4, # damage multiplier
            [1, 1, 1],
            [1] * 6, # team hp
            [1] * 36, # team base stats
            [1] * 6, # team levels
            [1] * 6, # opponent hp
            [1] * 36, # opponent base stats
            [1] * 6, # opponent levels
        ])
        embedding = Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
        return embedding

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
        eval_freq=5_000,
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

    # policy_kwargs = dict(
    #     net_arch = [128, 128]
    # )
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     policy_kwargs=policy_kwargs
    # )
    model = PPO.load('stats_levels', env=env)

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
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='stats_levels',
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

    # evaluate final time after training finishes 
    eval_callback.on_training_end()

    model.save('stats_levels')

# evaluates the model by seeing how many times it can win against the SimpleHeuristicBot
# battles is the number of battles to evaluate with
def a2c_evaluation(env: SimpleRLPlayer, model: A2C, battles):
    finished_battles = 0

    env.reset_battles()
    obs, _ = env.reset()
    
    done = False
    while finished_battles < battles:
        if done or env.current_battle.finished:
            finished_battles += 1
            print(f"EPISODE {finished_battles}")
            obs, _ = env.reset()
            if finished_battles >= battles:
                break
            done = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (env.n_won_battles, battles)
    )


NB_TRAINING_STEPS = 250_000
TEST_EPISODES = 100
GEN_8_DATA = GenData.from_gen(8)

if __name__ == "__main__":
    simpleHeuristicsPlayer = SimpleHeuristicsPlayer(battle_format='gen8randombattle')
    maxBasePowerPlayer = MaxBasePowerPlayer(battle_format='gen8randombattle')

    env = SimpleRLPlayer(opponent='cocobutterpuffs')
    # train the bot
    # ppo_train(env, NB_TRAINING_STEPS)

    # evaluate the bot
    model = PPO.load('stats_levels')
    a2c_evaluation(env, model, 1)

