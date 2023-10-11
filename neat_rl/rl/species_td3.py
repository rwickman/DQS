import copy, os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math, random

from torch.nn.functional import log_softmax
from neat_rl.networks.species_actor import SpeciesActor
from neat_rl.networks.species_critic import SpeciesCritic
from neat_rl.networks.discriminator import Discriminator 

from neat_rl.rl.species_replay_buffer import SpeciesReplayBuffer

class SpeciesTD3:
    def __init__(self, args, state_dim, action_dim, max_action, behavior_dim):
        self.args = args
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.behavior_dim = behavior_dim
        print("state_dim", state_dim)
        print("action_dim", action_dim)
        print("max_action", max_action)

        self.args.policy_noise = args.policy_noise * max_action
        self.args.noise_clip = args.noise_clip * max_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_networks()

        self.critic_loss_fn = nn.MSELoss()
        self.disc_loss_fn = nn.CrossEntropyLoss()

        self.replay_buffer = SpeciesReplayBuffer(self.state_dim, action_dim, behavior_dim, self.args.replay_capacity, self.args.resample_species)

        self.total_iter = 0
        self.rl_save_file = os.path.join(self.args.save_dir, "td3.pt")


    def init_networks(self):
        self.actor = SpeciesActor(self.state_dim, self.action_dim, self.args.critic_hidden_size, self.args.n_hidden, self.max_action, self.args.num_species).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.critic = SpeciesCritic(self.state_dim, self.action_dim, self.args.critic_hidden_size, self.args.n_hidden, self.args.num_species).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        if self.args.use_behavior_disc:
            disc_inp_dim = self.behavior_dim
        else:
            disc_inp_dim = self.state_dim
        
        self.discriminator = Discriminator(
            self.args.num_species,
            disc_inp_dim,
            self.args.critic_hidden_size,
            self.args.n_hidden).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.disc_lr)


    def select_action(self, state, species_id):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state, species_id).cpu().data.numpy().flatten()

    def sample_action(self, state, species_id):
        species_id = torch.tensor([species_id]).to(self.device)
        action = (
            self.select_action(state, species_id)
                + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        return action
    
    def get_diversity(self, behavior, species_id):
        return log_softmax(self.discriminator(torch.FloatTensor(behavior).to(self.device)), dim=-1)[species_id].item() - math.log(1/self.args.num_species)

    def train(self):
        for _ in range(self.args.replay_ratio):
            self.total_iter += 1
            state, action, next_state, reward, species_id, behavior, terminated = self.replay_buffer.sample(self.args.batch_size)
            if self.args.resample_species:
                org_species = species_id
                if random.random() <= 0.5:
                    species_id = torch.randint(0, self.args.num_species, torch.Size([self.args.batch_size])).to(self.device)
            
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.args.policy_noise
                ).clamp(-self.args.noise_clip, self.args.noise_clip)

                next_action = (
                    self.actor_target(next_state, species_id) + noise
                ).clamp(-self.max_action, self.max_action)

                if self.args.use_behavior_disc:
                    disc_logits = self.discriminator(behavior)
                else:
                    disc_logits = self.discriminator(next_state)
                diversity_bonus = log_softmax(disc_logits, dim=-1).gather(-1, species_id.unsqueeze(1))  - math.log(1/self.args.num_species)
                qd_reward = reward + self.args.disc_lam * diversity_bonus
                    

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action, species_id)
                target_Q = torch.min(target_Q1, target_Q2)
                
                target_Q = qd_reward + terminated * self.args.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action, species_id)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_norm)
            self.critic_optimizer.step()

            if self.args.use_behavior_disc:
                logits = self.discriminator(behavior)
            else:
                logits = self.discriminator(state)

            if self.args.resample_species:
                disc_loss = self.disc_loss_fn(logits, org_species)
            else:
                disc_loss = self.disc_loss_fn(logits, species_id)
                
            self.discriminator_optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.max_norm)
            disc_loss.backward()
            self.discriminator_optimizer.step()
            
            # Delayed policy updates
            if self.total_iter % self.args.policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state, species_id), species_id).mean()
                #actor_loss = self.critic_loss_fn(self.actor(state, species_id), action_org)
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_norm)
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                if self.total_iter % 2048 == 0:
                    print("TOTAL ITER: ", self.total_iter)
                    print("behavior target_Q", current_Q1[:10].view(-1), target_Q[:10].view(-1))

                    print("species_id", species_id[:10])
                    print("reward", reward[:10].view(-1))
                    print("actor_loss", actor_loss, "critic_loss", critic_loss, "disc_loss", disc_loss)
                    print("behavior", behavior[:10])
                    #print("skew_weights", skew_weights[:10], probs[:10], skew_weights.max(), probs.max(), skew_weights.sum(), probs.sum())
                    print("AVG REWARD", reward.mean(), "AVG DIVERSITY", diversity_bonus.mean())
                    print("diversity_bonus", diversity_bonus.view(-1)[:10], "\n")
                
            if self.args.reset_policy and self.total_iter % self.args.reset_policy_steps == 0:
                print("\n \n RESTING POLICY \n \n")
                self.init_networks()
                
    
    def save(self):
        model_dict = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "total_iter": self.total_iter,
        }
        torch.save(model_dict, self.rl_save_file)
        import time
        buffer_start_time = time.time()
        self.replay_buffer.save(self.args.save_dir)
        print("REPLAY BUFFER SAVE TIME: ", time.time() - buffer_start_time)


    def load(self):
        model_dict = torch.load(self.rl_save_file)

        self.actor.load_state_dict(model_dict["actor"])
        self.actor_target.load_state_dict(model_dict["actor_target"])
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic.load_state_dict(model_dict["critic"])
        self.critic_target.load_state_dict(model_dict["critic_target"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.discriminator.load_state_dict(model_dict["discriminator"])
        self.discriminator_optimizer.load_state_dict(model_dict["discriminator_optimizer"])
        if "total_iter" in model_dict:
            self.total_iter = model_dict["total_iter"]
        
        self.replay_buffer.load(self.args.save_dir)