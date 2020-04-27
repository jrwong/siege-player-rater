from Player import Player

# go through each killfeed event in running_killfeed and assign values to players
def compile_player_scores(running_killfeed, players):
    for kf in running_killfeed:
        killer = find_player(kf.kill.name, players)
        killer.kills += 1
        death = find_player(kf.death.name, players)
        death.deaths += 1

    for player in players:
        print([players[player]])

# find name in list of players and return the player object, otherwise create player and add to players\
# players is a dictionary with name as the key and the player object as the value
def find_player(name, players):
    if name not in players:
        players[name] = Player(name)

    return players[name]
