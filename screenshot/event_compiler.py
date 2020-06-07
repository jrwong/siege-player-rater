from Player import Player


# go through each killfeed event in running_killfeed and assign values to players
def compile_player_scores(running_killfeed, players):
    length = len(running_killfeed)
    i = 0
    while i < length:
        kf = running_killfeed[i]
        prev_kf = running_killfeed[i-1]

        # should only record on non-teamkills
        if kf.death.desc != kf.kill.desc:
            # opening kill/death
            if i == 0:
                opening_killer = find_player(kf.kill.name, players)
                opening_killer.opening_kills += 1
                opening_death = find_player(kf.death.name, players)
                opening_death.opening_deaths += 1

            killer = find_player(kf.kill.name, players)
            killer.kills += 1

        death = find_player(kf.death.name, players)
        death.deaths += 1

        # trades: if the person in the prev_kf died from the person who was killed in kf, record trade score for
        # death in prev_kf
        if kf.death.name == prev_kf.kill.name:
            traded = find_player(prev_kf.death.name, players)
            traded.trades += 1

        i += 1

    for player in players:
        print([players[player]])

# find name in list of players and return the player object, otherwise create player and add to players\
# players is a dictionary with name as the key and the player object as the value
def find_player(name, players):
    if name not in players:
        players[name] = Player(name)

    return players[name]
