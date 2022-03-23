"""
TODO: Reset all the constants and rewards of this file at your convenience
TODO: Every time you modify constants.py you will need to reinstall the CommonsGame environment!!!
Remember, with the terminal on the folder with setup.py, write:   pip install -e .
Do not forget the dot
"""
training_now = True

# Environment logic altering!
TIMEOUT_FRAMES = 25
TOO_MANY_APPLES = 12
COMMON_POOL_HAS_LIMIT = True
AGENTS_CAN_GET_SICK = False
AGENTS_HAVE_DIFFERENT_EFFICIENCY = True
SUSTAINABILITY_MATTERS = training_now  # If False, apples ALWAYS regenerate
REGENERATION_PROBABILITY = 0.05  # Only matters if SUSTAINABILITY does not matter
respawnProbs = [0.01, 0.05, 0.1]

# Positive rewards
DONATION_REWARD = 1.0
TOOK_DONATION_REWARD = 1.0
APPLE_GATHERING_REWARD = 1.0
DID_NOTHING_BECAUSE_MANY_APPLES_REWARD = 0.0  # related with sustainability probably

# Negative rewards
TOO_MANY_APPLES_PUNISHMENT = -1.0  # related with sustainability probably
SHOOTING_PUNISHMENT = -0.0
HUNGER = -1.0
LOST_APPLE = -0.0




bigMap = [
    list('======================================'),
    list('======================================'),
    list('                                      '),
    list('             @      @@@@@       @     '),
    list('         @   @@         @@@    @  @   '),
    list('      @ @@@  @@@    @    @ @@ @@@@    '),
    list('  @  @@@ @    @  @ @@@  @  @   @ @    '),
    list(' @@@  @ @    @  @@@ @  @@@        @   '),
    list('  @ @  @@@  @@@  @ @    @ @@   @@ @@  '),
    list('   @ @  @@@    @ @  @@@    @@@  @     '),
    list('    @@@  @      @@@  @    @@@@        '),
    list('     @       @  @ @@@    @  @         '),
    list(' @  @@@  @  @  @@@ @    @@@@          '),
    list('     @ @   @@@  @ @      @ @@   @     '),
    list('      @@@   @ @  @@@      @@   @@@    '),
    list('  @    @     @@@  @             @     '),
    list('              @                       '),
    list('                                      ')
]

smallMap = [
    list('==========='),
    list('==========='),
    list(' @    @    '),
    list('   @@  @ @ '),
    list('  @@@@ @@@ '),
    list('   @@   @  '),
    list('          @')]

tinyMap = [
    list('===='),
    list('===='),
    list(' @@ '),
    list(' @  '),
    list('    ')]



