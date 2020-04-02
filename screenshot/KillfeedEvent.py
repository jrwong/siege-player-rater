class KillfeedEvent:

    # This class represents a kill/death and the associated information
    def __init__(self, kill, death, scoreboard_readout):
        self.kill = kill
        self.death = death
        self.scoreboard_readout = scoreboard_readout

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, KillfeedEvent):
            return self.kill == other.kill and self.death == other.death \
                   and self.scoreboard_readout.orange_score == other.scoreboard_readout.orange_score \
                   and self.scoreboard_readout.blue_score == other.scoreboard_readout.blue_score
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))