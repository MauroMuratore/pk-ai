import numpy as np # type: ignore
import pk_calc
from gymnasium.spaces import Space, Box
from poke_env.environment.status import Status
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player import (
    Gen4EnvSinglePlayer,
)
class DQNEnv(Gen4EnvSinglePlayer):
    
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action = self.dqn_agent.foward(state)
        battle_order = self.action_to_move(action)
        return battle_order
    
    def set_agent(self, dqn_agent):
        self.dqn_agent=dqn_agent

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle,
            victory_value = 20,
            fainted_value = 2,
            hp_value = 0.01,
        )
    
    def embed_battle(self, battle: AbstractBattle):
        # FIELD
        # retrive weather neuron 4
        b_weather = list(battle.weather.keys())
        weather = pk_calc.calc_weather(b_weather)
        # retrive side_condition neuron 4
        b_side_condition = list(battle.side_conditions.keys())
        side_condition = pk_calc.calc_side_condition(b_side_condition)
        # retrive opponent_side_condition neuron 4
        b_opp_side_condition = list(battle.opponent_side_conditions.keys())
        opp_side_condition = pk_calc.calc_side_condition(b_opp_side_condition)

        # POKEMON ACTIVE
        b_pkmn_active = battle.active_pokemon
        pkmn_active = pk_calc.calc_pokemon(b_pkmn_active)

        # OPP POKEMON ACTIVE
        b_opp_pkmn_active = battle.opponent_active_pokemon
        opp_pkmn_active = pk_calc.calc_pokemon(b_opp_pkmn_active)

        # TEAM
        b_team = battle.team
        l_team = []
        for l, b_t in b_team.items():
            if b_t.name == b_pkmn_active.name:
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
            if b_t.name == b_opp_pkmn_active.name:
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
            weather,
            side_condition,
            opp_side_condition,
            pkmn_active,
            opp_pkmn_active,
            team,
            opp_team,
            fainted_mon_team,
            opp_fainted_mon_team
        ], axis=None)
        self.len_space = len(v_return)
        return v_return
    
    def describe_embedding(self) -> Space:
        low = []
        high =[]
        for i in range(1613):
            low.append(-10.0)
            high.append(10.1)
        low = np.array(low)
        high = np.array(high)
        return Box(
            low=low,
            high=high,
            dtype=np.float64
        )