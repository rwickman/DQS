
import numpy as np
import torch
import os

from neat_rl.helpers.util import get_device

class ReplayBuffer:
	"""Replay buffer for a species that allocates some memory for expert experiences."""
	def __init__(self, state_dim, action_dim, behavior_dim, expert_size, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		# Size of the replay buffer reserved to the expert memory
		self.expert_size = expert_size
		self.is_expert = True
		# Create pointers to the reserved space for 
		self.expert_start_ptr = -1
		self.expert_end_ptr = -1
		

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.behavior = np.zeros((max_size, behavior_dim))
		self.not_done = np.zeros((max_size, 1))

		self.device = get_device()

	def add(self, state, action, next_state, reward, behavior, done):
		if self.ptr == self.expert_end_ptr:
			print("=" * 10, "\n", "self.size, self.ptr, expert_start_ptr, expert_end_ptr", self.size, self.ptr, self.expert_start_ptr, self.expert_end_ptr, "\n")

		if not self.is_expert and self.expert_size > 0 and self.ptr == self.expert_start_ptr:
			self.ptr = (self.expert_end_ptr + 1) % self.max_size

		if self.ptr == self.expert_end_ptr:
			print("=" * 10, "\n", "self.size, self.ptr, expert_start_ptr, expert_end_ptr", self.size, self.ptr, self.expert_start_ptr, self.expert_end_ptr, "\n")
	
		assert self.ptr != self.expert_end_ptr

		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.behavior[self.ptr] = behavior
		self.not_done[self.ptr] = 1. - done
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

			
	def get_save_dict(self):
		d = {
			"state": self.state,
			"action": self.action,
			"next_state": self.next_state,
			"reward": self.reward,
			"behavior": self.behavior,
			"not_done": self.not_done,
			"ptr": self.ptr,
			"expert_start_ptr": self.expert_start_ptr,
			"expert_end_ptr": self.expert_end_ptr,
			"size": self.size,
			"is_expert": self.is_expert
		}


		return d

	def sample(self, batch_size):
		ind = np.random.choice(self.size, size=batch_size, replace=False)

		return [
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.behavior[ind],
			self.not_done[ind]
		]

	def sample_states(self, batch_size):
		"""Sample only states."""
		ind = np.random.choice(self.size, size=batch_size, replace=False)
		sampled_states = self.state[ind]
		return torch.FloatTensor(sampled_states).to(self.device)

	def update_expert(self, did_improve: bool):
		if not self.is_expert and did_improve:
			self.is_expert = did_improve
			self.expert_start_ptr = -1
			self.expert_end_ptr = -1

	def toggle_reset(self):
		if self.is_expert and self.expert_size > 0:
			if self.size == self.max_size:
				self.expert_start_ptr = (self.ptr - self.expert_size - 1) % self.size
				self.expert_end_ptr = (self.ptr - 1) % self.size
			else:
				self.expert_start_ptr = max(0, self.ptr - self.expert_size - 1)
				self.expert_end_ptr = self.ptr - 1

			# Sanity-check
			assert self.expert_start_ptr >= 0 and self.expert_start_ptr < self.size
			assert self.expert_end_ptr >= 0 and self.expert_end_ptr < self.size

			
		self.is_expert = False

	def load(self, data):
		"""Load the replay buffer for a particular species."""
		self.state = data["state"]
		self.action = data["action"]
		self.reward = data["reward"]
		self.behavior = data["behavior"]
		self.next_state = data["next_state"]
		self.not_done = data["not_done"]
		self.ptr = int(data["ptr"])
		self.expert_start_ptr = int(data["expert_start_ptr"])
		self.expert_end_ptr = int(data["expert_end_ptr"])
		self.size = int(data["size"])
		if "is_expert" in data:
			self.is_expert = bool(data["is_expert"])
		else:
			self.is_expert = False


class SpeciesReplayBuffer:
	def __init__(self, state_dim, action_dim, behavior_dim, num_species, expert_capacity_pct, max_size=int(1e6)):
		self.num_species = num_species

		# Create a buffer for each species
		expert_size = int(expert_capacity_pct * max_size//num_species)
		self._buffers = [
			ReplayBuffer(state_dim, action_dim, behavior_dim, expert_size, max_size=max_size//num_species)
			for _ in range(num_species)
		]
		self.device = get_device()

	def add(self, state, action, next_state, reward, species_id, behavior, done):
		self._buffers[species_id].add(state, action, next_state, reward, behavior, done)

	def sample(self, batch_size):
		# Number of examples per species to sample
		ex_per_species = batch_size // self.num_species
		
		# states, actions, next_states, rewards, behaviors, not_dones, species_ids
		batch = [[], [], [], [], [], [], []]

		# Sample all the species
		for i in range(self.num_species):
			species_batch = self._buffers[i].sample(ex_per_species)
			
			for j in range(len(species_batch)):
				batch[j].append(species_batch[j])
			
			# Add the species
			batch[-1] += [i] * ex_per_species
		
		# Join them into a single batch
		for i in range(len(batch) - 1):
			batch[i] = torch.FloatTensor(
				np.concatenate(batch[i])).to(self.device)
		
		batch[-1] = torch.LongTensor(batch[-1]).to(self.device)
		
		return batch

	def sample_states(self, batch_size, species_id):
		return self._buffers[species_id].sample_states(batch_size)

	def save(self, save_dir):
		out_file = os.path.join(save_dir, "replay_buffer.npz")
		save_dicts = {}
		for i in range(self.num_species):
			save_dicts[str(i)] = self._buffers[i].get_save_dict()

		np.savez(out_file, **save_dicts)

	def load(self, save_dir):
		out_file = os.path.join(save_dir, "replay_buffer.npz")
		if os.path.exists(out_file):
			data = np.load(out_file, allow_pickle=True)
			for i in range(len(data)):
				species_data = data[str(i)].item()
				self._buffers[i].load(species_data)
	
	def update_expert(self, species_id, did_improve):
		self._buffers[species_id].update_expert(did_improve)
	
	def toggle_reset(self, species_id):
		self._buffers[species_id].toggle_reset()

	@property
	def size(self):
		return min(self._buffers[i].size for i in range(self.num_species))