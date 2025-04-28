import asyncio

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import RandomPlayer, Player, SimpleHeuristicsPlayer, MaxBasePowerPlayer

# class MaxDamagePlayer(Player):
#     def choose_move(self, battle):
#         # Chooses a move with the highest base power when possible
#         if battle.available_moves:
#             # Iterating over available moves to find the one with the highest base power
#             best_move = max(battle.available_moves, key=lambda move: move.base_power)

#             if battle.can_tera:
#                 return self.create_order(best_move, terastallize=True)
#             # Creating an order for the selected move
#             return self.create_order(best_move)
#         else:
#             # If no attacking move is available, perform a random switch
#             # This involves choosing a random move, which could be a switch or another available action
#             return self.choose_random_move(battle)

async def main():
    p1 = SimpleHeuristicsPlayer(battle_format='gen8randombattle')
    p2 = MaxBasePowerPlayer(battle_format='gen8randombattle')

    # max_damage_player = MaxDamagePlayer()

    await p1.battle_against(p2, n_battles=50)

    print(
        f'Player {p1.username} won {p1.n_won_battles} out of {p1.n_finished_battles} played'
    )

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())