from poke_env.player.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
import tensorflow as tf
import numpy as np
import util

class DQNPlayer(Player):

    def __init__(self, dqn_env, ck_point_model=None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.model = util.create_model(dqn_env.action_space.n)
        self.model.load_weights(ck_point_model)
        self.dqn_env = dqn_env
        self.graph = tf.compat.v1.get_default_graph()
        self.session = tf.compat.v1.keras.backend.get_session()
        self.session.run(tf.compat.v1.global_variables_initializer())
    
    def choose_move(self, battle: AbstractBattle):
        state = self.embed_battle(battle)
        with self.graph.as_default(), self.session.as_default():
            pred = self.model.predict(state.reshape(1,-1))        
        battle_order = self.dqn_env.action_to_move(np.argmax(pred), battle)
        return battle_order

    def close(self):
        self.dqn_env.close()
    
    def embed_battle(self, battle: AbstractBattle):
        return self.dqn_env.embed_battle(battle)
