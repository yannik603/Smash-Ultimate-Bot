import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from env import env
import numpy as np
import os
import time, datetime
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
print("Setting MetricLogger")
class MetricLogger:
    def __init__(self, save_dir):
        self.writer = SummaryWriter()
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_distances = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        
        self.actions = []
        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q, action, episode, distance):
        self.curr_ep_reward += reward
        self.ep_distances.append(distance)
        self.curr_ep_length += 1
        self.actions.append(action)
        self.writer.add_histogram("Actions", action, episode)
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0


    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        mean_distance = np.round(np.mean(self.ep_distances[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        #add to tensorboard
        self.writer.add_scalar("Mean Reward", mean_ep_reward, episode)     
        self.writer.add_scalar("Mean Length", mean_ep_length, episode)
        self.writer.add_scalar("Mean Loss", mean_ep_loss, episode)
        self.writer.add_scalar("Mean Q Value", mean_ep_q, episode)
        self.writer.add_scalar("Time Delta", time_since_last_record, episode)
        self.writer.add_scalar("Epsilon", epsilon, episode)
        self.writer.add_scalar("Step", step, episode)
        self.writer.add_scalar("Distance", mean_distance, episode)
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_distances = []
        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        
        self.actions = []
print("Setting Player Net")
class PlayerNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        #make a model that takes in an input of 3x40x64 and outputs the output dim and a batch size of 1
        self.online = nn.Sequential(
        #make a model that takes in an input of 3x48x48 and outputs one of 13 classes
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4096, 2024),
                nn.ReLU(),
                nn.Linear(2024, 500),
                nn.ReLU(),
                nn.Linear(500, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 21))

        #load = none
        #online a model using the alexnet model
        #self.online = self.makeModel(input_dim, output_dim)
        #load = os.path.join(script_dir, "models")
        load = 'model1'
        # if load!=None:
        #     self.online = torch.load('model1')
        self.online.load_state_dict(torch.load('states1'))
        #set the model to train mode
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen.
        # for p in self.target.parameters():
        #     p.requires_grad = False
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
    def makeModel(self,input_dim, output_dim):
        load = modeldir = os.path.join(script_dir, "models")
        c, h, w = input_dim
        # self.online = nn.Sequential(
        # #make a model that takes in an input of 3x40x64 and outputs the output dim and a batch size of 1
        # nn.Conv2d(c, 32, kernel_size=8, stride=4),
        # nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=1),
        # nn.ReLU(),
        # nn.Flatten(),
        # nn.Linear(256, 128),
        # nn.ReLU(),
        # nn.Linear(128, output_dim))
        if load!=None:
            self.online = torch.load(load)
        
        return self.online
print("Setting Player")
class Player:
    def __init__(self, state_dim, action_dim, save_dir):
        ##super().__init__(state_dim, action_dim, save_dir)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # players's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = PlayerNet(self.state_dim, self.action_dim).double()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
            print("Using CUDA")

        self.exploration_rate = .9
        self.exploration_rate_decay = 0.999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = .9
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1000  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1000  # no. of experiences between Q_target & Q_online syn
    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))
        torch.cuda.empty_cache()
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
        np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.double()) * self.gamma * next_Q).double()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
        
    
    def save(self):
        save_path = (
        self.save_dir / f"net\\net{int(self.curr_step // self.save_every)}.chkpt"
    )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
    )
        print(f"Net saved to {save_path} at step {self.curr_step}")
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
print("Setting directories")
script_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

modeldir = os.path.join(script_dir, "model1")
statedir = os.path.join(script_dir, "states1")
modeldir2 = os.path.join(script_dir, "model2")
statedir2 = os.path.join(script_dir, "states2")

player = Player(state_dim=(3, 48, 48), action_dim=21, save_dir=save_dir)
player2 = Player(state_dim=(3, 48, 48), action_dim=21, save_dir=save_dir)

logger = MetricLogger(save_dir)
logger2 = MetricLogger(save_dir)
print("Stating Environment")
envi = env()
#envi2 = env(True, 1)
episodes = 1000000
saveEvery = 100
for e in range(episodes):

    state = envi.reset()
    #state2 = envi2.reset()
    # make the model play itself
    while True:
 
        # Run agent on the state
        action = player.act(state)

        # Agent performs action
        next_state, reward, done, info, distance = envi.step(action)

        # Remember
        player.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = player.learn()

        # Logging
        logger.log_step(reward, loss, q, action, e, distance)

        # Update state
        state = next_state

        # Check if end of game
        # do player 2 not supported until nxbt supports 2 bluetooth adapters to train with against each other
        # action2 = player2.act(state)
        # next_state2, reward2, done2, info2 = envi2.step(action2)
        # player2.cache(state, next_state2, action2, reward2, done2)
        # q2, loss2 = player2.learn()
        # logger2.log_step(reward2, loss2, q2, action2, e)
        # state = next_state2
        
        
        if done:
            #envi.reset()
            #envi2.reset()
            break;

    logger.log_episode()
   # logger2.log_episode()
    
    if e % 20 == 0:
        logger.record(episode=e, epsilon=player.exploration_rate, step=player.curr_step)
    if e % saveEvery == 0:
        #logger.save()
        torch.save(player.net, modeldir)
        #torch.save(player2.net, modeldir2)
        #save the model dictionary
        torch.save(player.net.state_dict(), statedir)
        #torch.save(player2.net.state_dict(), statedir2)