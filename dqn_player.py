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
class DQNPlayer(Gen4EnvSinglePlayer):
    
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
        #print(f"--------------------------{self.len_space}-----------------------")
        print(f"{self.len_space}")
        return v_return.T
    
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
    
    
    def _calc_pokemon(self, pokemon) -> np.array:
        # type neuron 9
        pkmn_type = self._calc_type(pokemon.type_1, pokemon.type_2)
        # stats neuron 6
        pkmn_tot_stats = self._calc_tot_stats(pokemon.base_stats)
        # hp neuron 1
        pkmn_hp = pokemon.current_hp / pokemon.max_hp
        # status neuron 6
        pkmn_status = self._calc_status(pokemon.status)
        # items 
        #pkmn_item = self._calc_item(pokemon.item)
        # ability
        #pkmn_ability =
        # moves
        b_moves = pokemon.moves
        l_moves = []
        for _,move in b_moves.items():
            m = self._calc_move(move)
            l_moves.append(m)
        pkmn_moves = np.array(l_moves)
        missing_move = 4 - len(b_moves)
        for i in range(missing_move):
            a = np.zeros(25)
            pkmn_moves = np.concatenate((pkmn_moves, a), axis=None)

        return np.concatenate([
            pkmn_type,
            pkmn_tot_stats,
            pkmn_hp,
            pkmn_status,
            pkmn_moves
        ], axis=None)

    
    def _calc_type(self, type_1, type_2 = None) -> np.array:
        type_return = self._index_type(type_1)
        if type_2 != None:
            type_return = type_return + self._index_type(type_2)
        return type_return
    
    def _index_type(self, type_pkmn) -> np.array:
        type_return = np.zeros(9)
        if type_pkmn == PokemonType.FIRE:
            type_return[0] = 1
        elif type_pkmn == PokemonType.WATER:
            type_return[1] = 1
        elif type_pkmn == PokemonType.GRASS:
            type_return[2] = 1
        elif type_pkmn == PokemonType.DARK:
            type_return[3] =1
        elif type_pkmn == PokemonType.PSYCHIC:
            type_return[4] =1
        elif type_pkmn == PokemonType.FIGHTING:
            type_return[5] =1
        elif type_pkmn == PokemonType.GROUND:
            type_return[6] =1
        elif type_pkmn == PokemonType.FLYING:
            type_return[7] =1
        elif type_pkmn == PokemonType.ELECTRIC:
            type_return[8] =1
        return type_return
        
    def _calc_tot_stats(self, stats_pkmn) -> np.array:
        stats_return = np.zeros(6)
        for stat, value in stats_pkmn.items():
            l_stat = stat.lower()
            if l_stat =="hp":
                stats_return[0] = min(value/250.0,1)
            elif l_stat == "atk":
                stats_return[1] = min(value/250.0, 1)
            elif l_stat == "def":
                stats_return[2] = min(value/250.0, 1)
            elif l_stat == "spa":
                stats_return[3] = min(value/250.0, 1)
            elif l_stat == "spd":
                stats_return[4] = min(value/250.0, 1)
            elif l_stat == "spe":
                stats_return[5] = min(value/250.0, 1)
        return stats_return

    def _calc_status(self, status) -> np.array:
        status_return = np.zeros(6)
        if status == Status.BRN:
            status_return[0] = 1
        elif status == Status.FRZ:
            status_return[1] = 1
        elif status == Status.PAR:
            status_return[2] = 1
        elif status == Status.PSN:
            status_return[3] = 1
        elif status == Status.SLP:
            status_return[4] = 1
        elif status == Status.TOX:
            status_return[5] = 1
        return status_return
    
    def _calc_move(self, move)->np.array:
        move_return = np.zeros(10)
        # accuracy neuron 1
        move_return[0] = move.accuracy
        # base power neuron 1
        move_return[1] = move.base_power / 200.0
        # split neuron 3
        if move.category == MoveCategory.PHYSICAL:
            move_return[2] = 1
        elif move.category == MoveCategory.SPECIAL:
            move_return[3] = 1
        elif move.category == MoveCategory.STATUS:
            move_return[4] = 1
        
        # heal neuron 1
        move_return[5] = move.heal
        # drain neuron 1
        move_return[6] = move.drain
        # recoil neuron 1
        move_return[7] = move.recoil
        # protect neuron 1
        if move.is_protect_move:
            move_return[8] = 1
        
        # pp neuron 1
        move_return[9] = float(move.current_pp / move.max_pp)

        # boost self
        # move_return[10:15] = self._calc_boost(move.self_boost)
        # boost enemy
        #move_return[15:20] = self._calc_boost(move.boost)

        # status neuron 6
        status = np.zeros(6)
        if move.status != None :
            status = self._calc_status(move.status)
        
        # type neuron 9 
        m_type = self._calc_type(move.type)

        return np.concatenate([
            move_return,
            status,
            m_type
        ])

    def _calc_boost(self, boost)->np.array:
        boost_return = np.zeros(5)
        for boo, value in boost.items():
            if boo == "atk":
                boost_return[0] = value
            elif boo == "def":
                boost_return[1] = value
            elif boo == "spa":
                boost_return[2] = value
            elif boo == "spd":
                boost_return[3] = value
            elif boo == "spe":
                boost_return[4] = value

    def _calc_item(self, item):
        item_return = np.zeros()