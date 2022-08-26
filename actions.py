import nxbt
from nxbt import Sticks
import random

UPB = """
L_STICK@-000+100 B 0.05s
"""
DOWNB = """
L_STICK@-000-100 B 0.05s
"""
LEFTB = """
L_STICK@-100+000 B 0.05s
"""

RIGHTB = """
L_STICK@+100+000 B 0.05s
"""

UPSMASH = """
L_STICK@-000+100 B A 0.05s
"""
DOWNSMASH ="""
L_STICK@-000-100 B A 0.05s
"""
LEFTSMASH = """
L_STICK@-100+000 B A 0.05s
"""

RIGHTSMASH = """
L_STICK@+100+000 B A 0.05s
"""

GRAB = """
Y .01s
L_STICK@-000-100 .01s
"""
lag = [0, 0, 0, 0, 3, 42, 42, 42, 15, 54, 85, 42, 22, 85, 9, 43, 49, 49, 38, 37, 42, 42, 1]
reward = [0, 0, 0, 0, 0, 40, 20, 10, 40, 10, 40, 0, 0, 10, 10, 5, 5, 30, 40, 20, 20]
def doAction(nx, controller_index, action = random.randrange(0, 21)):
    if action==0: #move left  lag is 0
        nx.tilt_stick(controller_index, Sticks.LEFT_STICK, -100, 0)
    elif action==1: #move right 0
        nx.tilt_stick(controller_index, Sticks.LEFT_STICK, 100, 0)
    elif action==2: #move up 0 
        nx.tilt_stick(controller_index, Sticks.LEFT_STICK, 0, 100)
    elif action==3: #move down 0
        nx.tilt_stick(controller_index, Sticks.LEFT_STICK, 0, -100)
    elif action==4: #press X 3
        nx.press_buttons(controller_index, [nxbt.Buttons.X])
    elif action==5: #right tilt 42
        nx.tilt_stick(controller_index, Sticks.RIGHT_STICK, 100, 0)
    elif action==6: #up tilt 42
        nx.tilt_stick(controller_index, Sticks.RIGHT_STICK, 0, 100)
    elif action==7: #down tilt 42
        nx.tilt_stick(controller_index, Sticks.RIGHT_STICK, 0, -100)
    elif action==8: #press A 15
        nx.press_buttons(controller_index, [nxbt.Buttons.A])
    elif action==9: #press B 54
        nx.press_buttons(controller_index, [nxbt.Buttons.B])
    elif action==10: #press Y 85
        nx.press_buttons(controller_index, [nxbt.Buttons.Y])
    elif action==11: #left tilt 42
        nx.tilt_stick(controller_index, Sticks.RIGHT_STICK, -100, 0)
    elif action==12: #press L 22
        nx.press_buttons(controller_index, [nxbt.Buttons.L])
    elif action==13: #press R 85
        nx.press_buttons(controller_index, [nxbt.Buttons.R])
    elif action==14: #press UPB 9
        nx.macro(controller_index, UPB)
    elif action==15: #press DOWNB 43
        nx.macro(controller_index, DOWNB)
    elif action==16: #press LEFTB 49
        nx.macro(controller_index, LEFTB)
    elif action==17: #press RIGHTB 49
        nx.macro(controller_index, RIGHTB)
    elif action==18: #press UPSMASH 38
        nx.macro(controller_index, UPSMASH)
    elif action==19: #press DOWNSMASH 37
        nx.macro(controller_index, DOWNSMASH)
    elif action==20: #press LEFTSMASH 42
        nx.macro(controller_index, LEFTSMASH)
    elif action==21: #press RIGHTSMASH 42
        nx.macro(controller_index, RIGHTSMASH)
    return lag[action], reward[action]
    