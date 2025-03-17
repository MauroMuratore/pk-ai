import asyncio
from team_support import TeamSupport
from poke_env import AccountConfiguration
from poke_env import RandomPlayer
from poke_env.data import GenData

async def battle(p1,p2):
    await p1.battle_against(p2, n_battles=1)


if __name__=="__main__":
    team_supp = TeamSupport()
    teams = team_supp.yield_teams()
    random_player_1 = RandomPlayer(battle_format="gen4ou", team=teams[0],log_level=20)
    random_player_2 = RandomPlayer(battle_format="gen4ou", team=teams[1],log_level=20)
    
    asyncio.run(
        battle(random_player_1, random_player_2)
    )
