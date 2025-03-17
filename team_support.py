import os
import random 

class TeamSupport:

    def __init__(self):
        team_files = [ os.path.join("teams",f) for f in os.listdir("teams") if os.path.isfile(os.path.join("teams", f))]
        self.teams = []
        for t_file in team_files:
            with open(t_file, "r") as t:
                self.teams.append(t.read())

    def yield_teams(self):
        return [self.teams[0], self.teams[1]]