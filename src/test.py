from enum import Enum

class State(Enum):
    START = "start"

test = State.START.value

print(test)