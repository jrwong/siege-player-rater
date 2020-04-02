class KillfeedEvent:
    # This class represents a kill/death and the associated information
    def __init__(self, kill, death, scoreboard_readout):
        self.kill = kill
        self.death = death
        self.scoreboard_readout = scoreboard_readout