import os
import random 
from poke_env.teambuilder import Teambuilder

class TeamSupport(Teambuilder):


    def __init__(self, dir):
        random.seed(42)
        team_files = [ os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        self.teams = []
        for t_file in team_files:
            with open(t_file, "r") as file_team:
                str_team = file_team.read()
                parsed_team = self.parse_showdown_team(str_team)
                packed_team = self.join_team(parsed_team)
                self.teams.append(packed_team)

    def yield_team(self):
        return random.choice(self.teams)