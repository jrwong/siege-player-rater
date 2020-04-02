class PlayerInKillfeed:
    # This class returns a player's status in the killfeed
    def __init__(self, desc, name, x, y):
        self.desc = desc
        self.name = name
        self.x = x
        self.y = y

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, PlayerInKillfeed):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))