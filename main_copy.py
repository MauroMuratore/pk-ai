import asyncio
import multiprocessing
from dqn_env import DQNEnv
from test_player import SimpleRLPlayer
from team_support import TeamSupport
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.models import Model
import tensorflow as tf
import os
import util
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.optimizers.legacy import Adam
from gymnasium.utils.env_checker import check_env
from poke_env import RandomPlayer, Player
from poke_env import SimpleHeuristicsPlayer, MaxBasePowerPlayer
from poke_env import AccountConfiguration
from dqn_player import DQNPlayer
import time



async def battle(env1, env2):
    await env1.battle_against(env2, n_battles=1)
    
def battle_bot():
    
    team_supp = TeamSupport("teams")
    conf_player = AccountConfiguration("DQNPlayer", None)
    conf_player2 = AccountConfiguration("Opponent", None)
    
    dqnplayer_env = DQNEnv(battle_format="gen4ou", 
                      start_challenging=True,
                      team=team_supp,
                      account_configuration=conf_player,
                      opponent=conf_player2.username)
    player2 = SimpleHeuristicsPlayer(battle_format="gen4ou", 
                                       team=team_supp,
                                       account_configuration=conf_player2)
    dqnplayer = DQNPlayer(dqnplayer_env, 
                        ck_point_model="./checkpoints/versione_2", 
                        battle_format="gen4ou", 
                        team=team_supp)
    asyncio.run(battle(dqnplayer,player2))
    
def train(steps,weights_path='./checkpoints/versione_', epoch=0, version=0 ,save=False):
    
    team_supp = TeamSupport("teams")
    conf_player = AccountConfiguration(f"{'dqn_traning_'}{epoch}", None)
    conf_opponent_sh = AccountConfiguration(f"{'opponent_sh_'}{epoch}", None)
    conf_opponent_max = AccountConfiguration(f"{'opponent_max_'}{epoch}", None)
    conf_opponent_dqn = AccountConfiguration(f"{'opponent_dqn_'}{epoch}", None)
    
    weights_to_load = f"{weights_path}{version}"
    #if os.path.isfile(weights_to_load):
    #    version = version - 1
    #    weights_loadable = True
    #else:
    #    version = 0
    #    weights_loadable = False
    
    sh_opponent = SimpleHeuristicsPlayer(battle_format="gen4ou", 
                                       team=team_supp,
                                       account_configuration=conf_opponent_sh)
    max_power_opponent = MaxBasePowerPlayer(battle_format="gen4ou", 
                                          team=team_supp,
                                          account_configuration=conf_opponent_max)
    
    #dqn_opponent_env = DQNEnv(battle_format="gen4ou", 
    #                  start_challenging=True,
    #                  team=team_supp,
    #                  account_configuration=conf_opponent_dqn,
    #                  opponent=conf_player.username)
    #
    #dqn_opponent = DQNPlayer(dqn_opponent_env, 
    #                            './checkpoints/versione_10', 
    #                            battle_format="gen4ou",
    #                            team= team_supp)
    
    #TO-DO: crea opponents list
        
    train_env = DQNEnv(battle_format="gen4ou", 
                      start_challenging=True,
                      team=team_supp,
                      account_configuration=conf_player,
                      opponent=[sh_opponent, max_power_opponent]) #dqn_opponent
    
    n_action = train_env.action_space.n
    model = util.create_model(n_action)
    if version > 0:    
        model.load_weights(weights_to_load)
    agent = util.create_agent(model, n_action, random_step=1000)
    agent.fit(train_env,nb_steps=steps)
    

    if save:
        weights_to_save = f"{weights_path}{version+1}"
        model.save_weights(weights_to_save, overwrite=True)
    train_env.close()
    

    
    


    
async def eval(player, player2, battles=1):
    await player.battle_against(player2, n_battles=battles)

        
        


def main():
    
    team_supp = TeamSupport("teams")
    conf_player = AccountConfiguration("dqn_test", None)
    conf_player2 = AccountConfiguration("sh_player", None)
    
    sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou", 
                                       team=team_supp,
                                       account_configuration=conf_player2)
    max_power_player = MaxBasePowerPlayer(battle_format="gen4ou", team=team_supp)
    test_env = DQNEnv(battle_format="gen4ou", 
                      start_challenging=True,
                      team=team_supp,
                      account_configuration=conf_player,
                      opponent=sh_player)
    
    #Checking env
    #check_env(test_env)
    #test_env.close()

    
    #Trainig
    weights_path = './checkpoints/versione_'
    steps = 200000
    n_epochs = 5
    start_version = 0
    save = True
    
    for i in range(n_epochs):
        version = start_version + i
        train(steps, weights_path, i, version, save)
    
    #Test
    #battle_bot()
    
    
    #EValuation
    #eval_env = DQNEnv(battle_format="gen4ou", 
    #                  start_challenging=True,
    #                  team=team_supp,
    #                  account_configuration=conf_player2,
    #                  opponent=sh_player)
    #eval_player = DQNPlayer(eval_env, model= model)
    #asyncio.run(eval(eval_player, sh_player, 1))
    #print(eval_player.battles)
        
    #n_action2 = test_env2.action_space.n
    #model2 = util.create_model(n_action2)    
    #agent2 = util.create_agent(model2,n_action2)
    
    
    
    #process = multiprocessing.Process(target=fit, args=[agent,test_env])
    #process2 = multiprocessing.Process(target=fit, args=[agent2,test_env2])
    #process.start()
    #process2.start()
    #process.join()
    #process2.join()

    

if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    main()

    
    
    
    
    
    
    
    
    
    
    #player = AccountConfiguration("dqnp", None)
    #player_1 = AccountConfiguration("dqnp_1", None)
    #random_player_1 = RandomPlayer(battle_format="gen4ou", team=team_supp)
    #sh_player = SimpleHeuristicsPlayer(battle_format="gen4ou", team=team_supp)
    ##random_player_2 = RandomPlayer(battle_format="gen4ou", team=team_supp,log_level=20
#
    #dqn_player = DQNPlayer(battle_format="gen4ou", team=team_supp)
    #asyncio.run(battle(dqn_player, sh_player))
    ##dqnp_env_1.set_agent(dqn_1)
    ##asyncio.run(battle(dqnp_env, sh_player))
    ##asyncio.run(battle(dqn, dqnp_env, dqn_1, dqnp_env_1))

    