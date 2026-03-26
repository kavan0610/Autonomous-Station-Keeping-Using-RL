import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ManualNormalizer:
    """
    Manually scales non-dimensional CRTBP components so the neural network 
    sees inputs generally bounded between -1.0 and 1.0.
    """
    def __init__(self, au_km):
        # Estimated maximums in non-dimensional units
        self.pos_scale = 1.5           # Absolute positions are usually near 1.0
        self.vel_scale = 0.05          # Absolute velocities are small
        self.pos_err_scale = 100000.0 / au_km  # Max allowed error before termination
        self.vel_err_scale = 0.0001    # Velocity errors are tiny
        self.lookahead_scale = 0.015   # Max distance across the halo orbit in ND

    def normalize(self, state):
        norm_state = np.zeros_like(state, dtype=np.float32)
        
        # Scale specific segments of the 21D observation array
        norm_state[0:3] = state[0:3] / self.pos_scale
        norm_state[3:6] = state[3:6] / self.vel_scale
        norm_state[6:9] = state[6:9] / self.pos_err_scale
        norm_state[9:12] = state[9:12] / self.vel_err_scale
        norm_state[12:15] = state[12:15] / self.lookahead_scale
        norm_state[15:18] = state[15:18] / self.lookahead_scale
        norm_state[18:21] = state[18:21] / self.lookahead_scale
        
        # Final safety clip to prevent explosive gradients
        return np.clip(norm_state, -5.0, 5.0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, init_action_std):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        
        self.action_std = torch.full((action_dim,), init_action_std).to(device)
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Output is strictly [-1, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        
    def act(self, state, deterministic=True):
        action_mean = self.actor(state)
        
        if deterministic:
            return action_mean.detach(), None, None
            
        dist = Normal(action_mean, self.action_std)
        action = dist.sample()
        
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    
class PPO_GAE:
    def __init__(self, state_dim, action_dim, env_au_km, lr=3e-4, gamma=0.99, lam=0.95, 
                 K_epochs=10, eps_clip=0.2, batch_size=64, init_action_std=0.6):
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size 
        
        self.action_std = init_action_std 
        
        self.normalizer = ManualNormalizer(env_au_km)
        
        self.policy = ActorCritic(state_dim, action_dim, init_action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim, init_action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, evaluate=True):
        norm_state = self.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state_tensor, deterministic=evaluate)
        
        if evaluate:
            return action.squeeze(0).cpu().numpy()
            
        
        return action.squeeze(0).cpu().numpy()
        
    def load(self, checkpoint_path):
        """Loads weights into both the active and old policy networks."""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))