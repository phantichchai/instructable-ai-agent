from enum import Enum

class ActionMapping(Enum):
    MOVE_FORWARD = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_BACKWARD = 3
    JUMP = 4
    SPRINT = 5
    NORMAL_ATTACK = 6
    ELEMENTAL_SKILL = 7
    ELEMENTAL_BURST = 8
    AIM_MODE = 9
    INTERACT = 10
    OPEN_MAP = 11
    OPEN_INVENTORY = 12
    SWITCH_CHARACTER_1 = 13
    SWITCH_CHARACTER_2 = 14
    SWITCH_CHARACTER_3 = 15
    SWITCH_CHARACTER_4 = 16
    SPECIAL_QUEST_ABILITIES = 17
    OPEN_MAIN_MENU = 18
    OPEN_CHARACTER_MENU = 19
    OPEN_QUESTS_MENU = 20
    CHARGE_ATTACK = 21

class KeyBinding(Enum):
    MOVE_FORWARD = 'w'
    MOVE_LEFT = 'a'
    MOVE_RIGHT = 'd'
    MOVE_BACKWARD = 's'
    JUMP = 'space'
    SPRINT = 'shift'
    NORMAL_ATTACK = 'left'
    ELEMENTAL_SKILL = 'e'
    ELEMENTAL_BURST = 'q'
    AIM_MODE = 'r'
    INTERACT = 'f'
    OPEN_MAP = 'm'
    OPEN_INVENTORY = 'b'
    SWITCH_CHARACTER_1 = '1'
    SWITCH_CHARACTER_2 = '2'
    SWITCH_CHARACTER_3 = '3'
    SWITCH_CHARACTER_4 = '4'
    SPECIAL_QUEST_ABILITIES = 't'
    OPEN_MAIN_MENU = 'esc'
    OPEN_CHARACTER_MENU = 'c'
    OPEN_QUESTS_MENU = 'j'
    CHARGE_ATTACK = 'left'

HOLDABLE_ACTIONS = {
    ActionMapping.MOVE_FORWARD,
    ActionMapping.MOVE_LEFT,
    ActionMapping.MOVE_RIGHT,
    ActionMapping.MOVE_BACKWARD,
    ActionMapping.SPRINT
}