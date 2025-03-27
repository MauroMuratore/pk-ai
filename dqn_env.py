import numpy as np # type: ignore
import pk_calc
from gymnasium.spaces import Space, Box
from poke_env.environment.status import Status
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player import (
    Gen4EnvSinglePlayer
)
class DQNEnv(Gen4EnvSinglePlayer):
    
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action = self.dqn_agent.foward(state)
        battle_order = self.action_to_move(action)
        return battle_order
    
    def reward_computing_helper(self, battle, *, fainted_value = 0, hp_value = 0, number_of_pokemons = 6, starting_value = 0, status_value = 0, victory_value = 1, penality_turn=1):
        reward = super().reward_computing_helper(battle,  
                                                 fainted_value=fainted_value , 
                                                 hp_value=hp_value, 
                                                 number_of_pokemons=number_of_pokemons, 
                                                 starting_value=starting_value , 
                                                 status_value=status_value , 
                                                 victory_value=victory_value )
        for i in range(10):
            if battle.turn > 25 + 10*i:
                reward -= penality_turn
            else:
                break
        return reward

    def calc_reward(self,last_battle ,current_battle) -> float:
        return self.reward_computing_helper( #NB!!!!
            current_battle,
            victory_value = 10,
            fainted_value = 1,
            hp_value = 0.01,
            #penality_turn=1
        )
    
    def embed_battle(self, battle: AbstractBattle):
        # FIELD - length 13
        # retrieve turn neuron 1
        turn = max(battle.turn/50, 1)
        # retrive weather neuron 4
        b_weather = list(battle.weather.keys())
        weather = pk_calc.calc_weather(b_weather)
        # retrive side_condition neuron 4
        b_side_condition = list(battle.side_conditions.keys())
        side_condition = pk_calc.calc_side_condition(b_side_condition)
        # retrive opponent_side_condition neuron 4
        b_opp_side_condition = list(battle.opponent_side_conditions.keys())
        opp_side_condition = pk_calc.calc_side_condition(b_opp_side_condition)

        # POKEMON ACTIVE - length 146
        b_pkmn_active = battle.active_pokemon
        pkmn_active = pk_calc.calc_pokemon(b_pkmn_active) 
        boost_active = pk_calc.calc_boost(b_pkmn_active.boosts)

        # OPP POKEMON ACTIVE - length 146
        b_opp_pkmn_active = battle.opponent_active_pokemon
        opp_pkmn_active = pk_calc.calc_pokemon(b_opp_pkmn_active)
        boost_opp_active = pk_calc.calc_boost(b_opp_pkmn_active.boosts)

        # TEAM
        b_team = battle.team
        l_team = []
        for _, b_t in b_team.items():
            if b_t.active:
                continue
            pkmn = pk_calc.calc_pokemon(b_t)
            l_team.append(pkmn)
        for i in range(5-len(l_team)):
            l_team.append(np.zeros(len(pkmn_active)))
        team = np.concatenate(l_team)
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        
        # OPP TEAM
        b_opp_team = battle.opponent_team
        l_o_team = []
        len_opp_team = len(b_opp_team)
        for _, b_t, in b_opp_team.items():
            if b_t.active:
                continue
            pkmn = pk_calc.calc_pokemon(b_t)
            l_o_team.append(pkmn)
        for i in range(6 -len_opp_team):
            l_o_team.append(np.zeros(len(pkmn_active)))

        opp_team = np.concatenate(l_o_team)
        opp_fainted_mon_team = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        if False:
            print(f"weather {len(weather)}")
            print(f"side {len(side_condition) + len(opp_side_condition)}")
            print(f"pkmn active {len(pkmn_active)}")
            print(f"opp pkmn active {len(opp_pkmn_active)}")
            print(f"team {len(team)}")
            print(f"opp team {len(opp_team)}")
        v_return = np.concatenate([
            turn,
            weather,
            side_condition,
            opp_side_condition,
            pkmn_active,
            boost_active,
            opp_pkmn_active,
            boost_opp_active,
            team,
            opp_team,
            fainted_mon_team,
            opp_fainted_mon_team
        ], axis=None)
        self.len_space = len(v_return)
        #print(self.len_space)
        return v_return
    
    #NB: dovremmo cambiare questi numeri per dare un significato migliore delle azioni prese
    # ES: fainted_mon_team sarà al minimo 0 (0 pkmn esausti) e max 1 (6 pkmn esausti)
    def describe_embedding(self) -> Space:
        low = []
        high =[]
        for i in range(1777): 
            low.append(-0.1)
            high.append(1.1)
        low = np.array(low)
        high = np.array(high)
        return Box(
            low=low,
            high=high,
            dtype=np.float64
        )