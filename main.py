import asyncio
from dqn_env import DQNEnv
from test_player import SimpleRLPlayer
from team_support import TeamSupport
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import os
from gymnasium.utils.env_checker import check_env
from poke_env import RandomPlayer, Player
from poke_env import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration
from dqn_player import DQNPlayer
import numpy as np
import util
from rl.callbacks import Callback
import time

async def battle(player_1, env_1, player_2, env_2):
    await player_1.battle_against(player_2, n_battles=1)
    #env_1.close()
    #env_2.close()

async def battle_human(player):
    await player.send_challenges("murotek", 1)

def train(steps, i):
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(session)

    team_supp = TeamSupport()
    conf_player = AccountConfiguration("dqn_traning", None)
    conf_sparring = AccountConfiguration(f"dqn_sparring_{i}", None)
    dqn_env_sparring = DQNEnv(battle_format="gen4ou",
                              account_configuration=conf_sparring,
                              opponent=conf_player.username)
    file_model_sparring="./checkpoints/versione_1"
    file_model_training="./checkpoints/versione_1"

    bot_sparring = DQNPlayer(
        dqn_env_sparring,
        file_model_sparring,
        battle_format="gen4ou",
        team=team_supp
    )
    dqn_env_player =DQNEnv(battle_format="gen4ou",
        account_configuration=conf_player,
        team=team_supp,
        opponent=bot_sparring,
        start_challenging=True
        
    )
    #dqn_env_player.set_opponent(bot_sparring)
    n_action = dqn_env_player.action_space.n

    model_training = util.create_model(n_action)
    model_training.load_weights(file_model_training)

    agent_training = util.create_agent(model_training, n_action)

    agent_training.fit(dqn_env_player, steps)

    model_training.save_weights(file_model_training, overwrite=True)

    print(f"Winning match {dqn_env_player.n_won_battles}")
    print(f"Losing match {dqn_env_player.n_lost_battles}")
    dqn_env_player.close()
    

if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    #train(100000,0)

    team_supp = TeamSupport("teams_all")
    team_supp_all = TeamSupport("teams_all")
    conf_player = AccountConfiguration("player_1", None)
    conf_sparring = AccountConfiguration("player_2", None)

    dqn_env_sparring = DQNEnv(battle_format="gen4ou",
                              account_configuration=conf_sparring,
                              opponent=conf_player.username)
    dqn_env_player = DQNEnv(battle_format="gen4ou",
                              account_configuration=conf_player,
                              opponent=None)
    #sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou",
    #                                   team=team_supp, 
    #                                   )

    bot_player_1 = DQNPlayer(dqn_env_player, "./checkpoints/versione_1", battle_format="gen4ou", team=team_supp)
    asyncio.run(battle_human(bot_player_1))
    #bot_player_2 = DQNPlayer(dqn_env_sparring, "./checkpoints/versione_1", battle_format="gen4ou", team=team_supp_all)
    #asyncio.run(battle(bot_player_1, dqn_env_player,bot_player_2, dqn_env_sparring ))
    #dqn_env_player.close()
    #dqn_env_sparring.close()
    #dqn_env_player =DQNEnv(battle_format="gen4ou",#sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou",
    #                                   team=team_supp, 
    #                                   )

    #                       account_configuration=conf_player,
    #                       team=team_supp,
    #                       opponent=bot_sparring,
    #                       start_challenging=True
    #                       )
    
    #dqn_env_player.set_opponent(bot_sparring)
    #n_action = dqn_env_player.action_space.n

    #update_model = util.create_model(n_action)
    #update_model.load_weights("./checkpoints/versione_1")
    #dqn_agent = util.create_agent()
    #dqn_agent.fit(dqn_env_player,nb_steps=10000, verbose=1)
    #print(f"Winning match {dqn_env_player.n_won_battles}")
    #print(f"Winning match {dqn_env_player.n_lost_battles}")
    #dqn_env_player.close()
    #dqn_env_sparring.close()
