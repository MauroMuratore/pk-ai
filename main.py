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

async def battle(env1, env2):
    await env1.battle_against(env2, n_battles=1)
    env1.close()
    env2.close()

def train(epochs, steps):
    team_supp = TeamSupport()
    conf_player = AccountConfiguration("dqn_traning", None)
    conf_sparring = AccountConfiguration("dqn_sparring", None)
    dqn_env_sparring = DQNEnv(battle_format="gen4ou",
                              account_configuration=conf_sparring,
                              opponent=conf_sparring.username)
    file_model_sparring="./checkpoints/versione_1"
    file_model_training="./checkpoints/versione_1"
    for i in range(epochs):
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
        n_action = dqn_env_player.action_space.n

        model_training = util.create_model(n_action)
        model_training.load_weights(file_model_training)

        agent_training = util.create_agent(model_training, n_action)


        #reset_callback = Callback()
        #reset_callback.on_step_end = lambda step, logs: dqn_env_player.reset() if logs.get("done") else None
 
        agent_training.fit(dqn_env_player, steps)

        model_training.save_weights(file_model_training, overwrite=True)

        print(f"Winning match {dqn_env_player.n_won_battles}")
        print(f"Winning match {dqn_env_player.n_lost_battles}")
        bot_sparring.close()
        dqn_env_player.close()
        


if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    train(3, 60)
    #team_supp = TeamSupport()
    #conf_player = AccountConfiguration("dqn_traning", None)
    #conf_sparring = AccountConfiguration("dqn_sparring", None)

    #dqn_env_sparring = DQNEnv(battle_format="gen4ou",
    #                          account_configuration=conf_sparring,
    #                          opponent=conf_sparring.username)
    #sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou",
    #                                   team=team_supp, 
    #                                   )

    #bot_player = DQNPlayer(dqn_env_player, "./checkpoints/versione_1", battle_format="gen4ou", team=team_supp)
    #bot_sparring = DQNPlayer(dqn_env_sparring, "./checkpoints/versione_1", battle_format="gen4ou", team=team_supp)
    #asyncio.run(battle(bot_player, bot_sparring))
    #dqn_env_player =DQNEnv(battle_format="gen4ou",
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