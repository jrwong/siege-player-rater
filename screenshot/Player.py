class Player:
    # This class returns a player's status in the killfeed
    # def __init__(self, name, rounds, kills, deaths, team_kills, opening_kills, opening_deaths, clutches,
    #              plants, defuses, trades, headshot):
    #     self.trades = trades
    #     self.defuses = defuses
    #     self.plants = plants
    #     self.clutches = clutches
    #     self.opening_deaths = opening_deaths
    #     self.opening_kills = opening_kills
    #     self.team_kills = team_kills
    #     self.deaths = deaths
    #     self.kills = kills
    #     self.headshot = headshot
    #     self.rounds = rounds
    #     self.name = name

    def __init__(self, name):
        self.trades = 0
        self.defuses = 0
        self.plants = 0
        self.clutches = 0
        self.opening_deaths = 0
        self.opening_kills = 0
        self.team_kills = 0
        self.deaths = 0
        self.kills = 0
        self.headshot = 0
        self.rounds = 1
        self.name = name

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Player):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        return "name: % s kills:% s deaths:% s opening_kills:% s opening_deaths:% s trades:% s " % (self.name, self.kills, self.deaths, self.opening_kills, self.opening_deaths, self.trades)