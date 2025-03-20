import asyncio
from dqn_env import DQNEnv
from test_player import SimpleRLPlayer
from team_support import TeamSupport
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import os
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers.legacy import Adam
from gymnasium.utils.env_checker import check_env
from poke_env import RandomPlayer, Player
from poke_env import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration
from dqn_player import DQNPlayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
SIZE_FIELD = 12
SIZE_STATUS_PKMN = 23
SIZE_MOVE = 25
SIZE_PKMN = 23 + 25*4


async def battle(env1, env2):
    await env1.battle_against(env2, n_battles=1)
    env1.close()
    env2.close()


if __name__=="__main__":
    team_supp = TeamSupport()
    player = AccountConfiguration("dqnp", None)
    player_1 = AccountConfiguration("dqnp_1", None)
    random_player_1 = RandomPlayer(battle_format="gen4ou", team=team_supp)
    sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou", team=team_supp)
    #random_player_2 = RandomPlayer(battle_format="gen4ou", team=team_supp,log_level=20

    dqn_player = DQNPlayer(battle_format="gen4ou", team=team_supp)
    asyncio.run(battle(dqn_player, sh_player))
    #dqnp_env_1.set_agent(dqn_1)
    #asyncio.run(battle(dqnp_env, sh_player))
    #asyncio.run(battle(dqn, dqnp_env, dqn_1, dqnp_env_1))

    