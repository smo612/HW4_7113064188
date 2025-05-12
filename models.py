import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ✅ Dueling Q-Network
class DuelingQNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.fc_adv = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        value = self.fc_value(x)
        adv = self.fc_adv(x)
        return value + (adv - adv.mean(dim=1, keepdim=True))

# Lightning Module
class DQNLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=0.99, batch_size=64, tau=0.005, buffer_size=10000, double_dqn=True):
        super().__init__()
        self.save_hyperparameters()
        self.q_net = DuelingQNet()
        self.target_net = DuelingQNet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.steps = 0
        self.double_dqn = double_dqn

    def forward(self, x):
        return self.q_net(x)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = torch.argmax(self.q_net(next_states), dim=1)
                target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                target_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.hparams.gamma * target_q * (1 - dones)

        loss = F.mse_loss(q_values, target)

        # Target network soft update
        if self.steps % 10 == 0:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.hparams.tau * param.data + (1.0 - self.hparams.tau) * target_param.data)

        self.log("train_loss", loss)
        self.steps += 1
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer], [scheduler]

    def sample_batch(self):
        if len(self.buffer) < self.hparams.batch_size:
            return None
        return self.buffer.sample(self.hparams.batch_size)
