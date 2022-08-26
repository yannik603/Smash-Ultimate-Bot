
# Smash Ultimate AI
A [PyTorch](https://pytorch.org/) model that uses reinforcement learning to train an External Smash Ultimate AI.
## Setup and Requirements
A Capture card to set up with the switch dock
A Bluetooth card to use the Bluetooth Emulator
[PyTorch](https://pytorch.org/)
[Tensorflow](https://www.tensorflow.org/install) optional for Tensorboard
[NXBT](https://github.com/Brikwerk/nxbt) Emulates a Bluetooth controller - only works on Linux systems

>Run preview.py to make sure the capture card device number is configured correctly and to check if nxbt can connect to the switch
>`cap = cv2.VideoCapture(0) #change to 0 or 1 depending on your device`

Run qlearning.py and follow the instructions

## How it works

![Dash then attack](https://github.com/yannik603/Smash-Ultimate-Bot/blob/main/ReadmePics/DashThenAttack.gif)

At first, the image is passed and resized
The model then takes an action and waits for the end lag of the movement to end.

![Frame Data](https://github.com/yannik603/Smash-Ultimate-Bot/blob/main/ReadmePics/FrameData.gif)

after the move is finished it checks for a change in percentage to check if either the opponent or the model took dmg

![Change in HP](https://github.com/yannik603/Smash-Ultimate-Bot/blob/main/ReadmePics/DamageTaken.gif)

The reward the move gets depends on how much dmg, its end lag, and if the move is a combo starter. The image at the same time is split up into a grid that runs through a second model that detects where the characters are in order to get the distance.

![Game&Watch](https://github.com/yannik603/Smash-Ultimate-Bot/blob/main/ReadmePics/Game&Watch.png)

 If the model runs toward the enemy and it doesn't get hit it gets a reward, but if it attacks and it doesn't hit anything while being far away it also gets a negative reward.
 ## To-do list
 

 - [ ] Better Reward function that rewards consecutive combos
 - [ ] A universal model to detect the characters on all stages
 - [ ] Use an API to get the frame data
 - [ ] Detect the exact percentage that was taken
