import gym
import torch
from neat_rl.rl.td3 import TD3
import QDgym

class Environment:
    def __init__(self, args, num_episodes=500000):
        self.args = args
        self.num_episodes=num_episodes
        self.env = gym.make(self.args.env)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])
        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        self.td3 = TD3(self.args, state_dim, action_dim, max_action)


        if self.args.load:
            self.td3.load()
        
        if self.args.render:
            self.env = gym.make(self.args.env, render_mode='human')
        else:
            self.env = gym.make(self.args.env)
        
        self.total_timesteps = 0


    def run(self, render=False):



        state = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        cur_step = 0

        while not done:
            cur_step += 1
            if self.td3.replay_buffer.size < self.args.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.td3.sample_action(state)

            next_state, reward, done, _ = self.env.step(action)

            self.td3.replay_buffer.add(state, action, next_state, reward, done)
            if self.total_timesteps % self.args.update_freq == 0 and self.td3.replay_buffer.size >= self.args.learning_starts:
                self.td3.train()

            self.total_timesteps += 1 

            if render:
                self.env.render()     
            total_reward += reward
            state = next_state

        return total_reward, cur_step

    def train(self):
        total_timesteps = 0
        for i in range(self.args.num_episodes):
            total_reward, timesteps = self.run()
            #total_reward, timesteps = self.run(True)
            #self.total_timesteps += timesteps

            #total_reward = self.run(False, True)
            print(f"TOTAL REWARD {total_reward} FOR EPISODE {i} TOTAL TIMESTEPS {self.total_timesteps}")
            # if i >= 10:
            #     for _ in range(16):
            #         self.td3.train(self.td3.replay_buffer)
                
        #     if i % 32 == 0: 
        #         self.td3.save()
        #         print("SAVED MODEL")
        
        # self.td3.save()