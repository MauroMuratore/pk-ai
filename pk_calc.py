import numpy as np
from poke_env.environment.weather import Weather
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.environment.move_category import MoveCategory


def calc_weather(weather) -> np.array:
    w_return = np.zeros(4)
    if len(weather) == 0:
        return w_return
    elif weather == Weather.RAINDANCE:        
        w_return[0] = 1
    elif weather == Weather.SUNNYDAY:
        w_return[1] = 1
    elif weather == Weather.SANDSTORM:
        w_return[2] = 1
    elif weather == Weather.HAIL:
        w_return[3] = 1
    return w_return


def calc_side_condition(side_cond) -> np.array:
    sc_return = np.zeros(4)
    for sc in side_cond:
        if sc == SideCondition.LIGHT_SCREEN:
            sc_return[0] = 1
        elif sc == SideCondition.REFLECT:
            sc_return[1] = 1
        elif sc == SideCondition.STEALTH_ROCK:
            sc_return[2] = 1
        elif sc == SideCondition.SPIKES:
            sc_return[3] += 1.0/3.0
    return sc_return


def calc_pokemon(pokemon) -> np.array:
    # type neuron 9
    pkmn_type = calc_type(pokemon.type_1, pokemon.type_2)
    # stats neuron 6
    pkmn_tot_stats = calc_tot_stats(pokemon.base_stats)
    # hp neuron 1
    if pokemon.max_hp > 0:
        pkmn_hp = pokemon.current_hp / pokemon.max_hp
    else:
        pkmn_hp = pokemon.current_hp
    # level neuron 1
    pkmn_lvl = pokemon.level / 100
    # status neuron 6
    pkmn_status = calc_status(pokemon.status)
    # items 
    #pkmn_item = self._calc_item(pokemon.item)
    # ability
    #pkmn_ability =
    # moves
    b_moves = pokemon.moves
    l_moves = []
    for _,move in b_moves.items():
        m = calc_move(move)
        l_moves.append(m)
    pkmn_moves = np.array(l_moves)
    len_move = 25
    missing_moves = 4 - len(b_moves)
    for i in range(missing_moves):
        a = np.zeros(len_move)
        pkmn_moves = np.concatenate((pkmn_moves, a), axis=None)

    return np.concatenate([
        pkmn_type,
        pkmn_tot_stats,
        pkmn_hp,
        pkmn_lvl,
        pkmn_status,
        pkmn_moves
    ], axis=None)


def calc_type(type_1, type_2 = None) -> np.array:
    type_return = index_type(type_1)
    if type_2 != None:
        type_return = type_return + index_type(type_2)
    return type_return

def index_type(type_pkmn) -> np.array:
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


def calc_tot_stats(stats_pkmn) -> np.array:
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


def calc_status(status) -> np.array:
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

def calc_move(move) -> np.array:
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

    # status neuron 6
    status = np.zeros(6)
    if move.status != None :
        status = calc_status(move.status)

    # type 9
    m_type = calc_type(move.type)

    return np.concatenate([
        move_return,
        status,
        m_type
    ])