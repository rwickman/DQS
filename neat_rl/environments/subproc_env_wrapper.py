import os
import gym
import torch
import random
import time

from torch.multiprocessing import Queue, Process, Pool, set_start_method, Manager, get_context

from neat_rl.environments.env_pop_diversity import EnvironmentGADiversity
from neat_rl.helpers.util import add_to_archive
from neat_rl.rl.species_td3ga import SpeciesTD3GA
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.actor import Actor

def _worker(org_net, org_idx, species_id, exp_queue, env_name, should_sample):
    # org, org_idx, species_id, exp_queue, env_name, should_sample = args
    env = gym.make(env_name)
    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        if should_sample:
            action = env.action_space.sample()
        else:
            action = org_net.select_action(state)

        next_state, reward, done, info = env.step(action)    
        behavior = env.desc
        exp_queue.put((state, action, next_state, reward, species_id, behavior, done))

        total_reward += reward
        state = next_state.copy()
    
    return org_idx, total_reward, env.desc


class SubprocEnvWrapper(EnvironmentGADiversity):
    """Used to manage multiple environments simultaneously."""

    def __init__(self, args, archive, archive_species_ids, kdt):
        super().__init__(args, archive, archive_species_ids, kdt)


    def _get_exp(self, exp_queue, should_sample):
        """Get an experience from an environment."""
        # Get an experience from one of the environments
        exp = exp_queue.get()
        self.td3ga.replay_buffer.add(*exp)
        self.total_timesteps += 1
        
        # Train the species models 
        if not should_sample and self.total_timesteps % self.args.update_freq == 0:
            start_time = time.time()
            self.td3ga.train()
            self.total_train_time += time.time() - start_time

    def _population_to_device(self, device):
        for org in self.population.orgs:
            org.net.to(device)
            org.net.device = device

    def train(self):
        n_cpus = os.cpu_count()
        start_time = time.time()
        max_fitness = None
        min_fitness = None
        total_fitness = 0
        random.shuffle(self.population.orgs)
        self.total_train_time = 0

        # Create the queue that will hold the list of experiences

        ctx = get_context("spawn")
        # Create the pool of workers
        with ctx.Manager() as manager:
            exp_queue = manager.Queue()
            with ctx.Pool(n_cpus) as pool:

                results = []
                should_sample = self.td3ga.replay_buffer.size < self.args.learning_starts
                
                # Create a task for each organism 
                self._population_to_device("cpu")
                for org_idx, org in enumerate(self.population.orgs):
                    species_id = self.population.org_id_to_species[org.id]

                    worker_args = (org.net, org_idx, species_id, exp_queue, self.args.env, should_sample)

                    results.append(pool.apply_async(_worker, worker_args))
                # Wait until all the environments are done running
                while len(results) > 0:
                    if not exp_queue.empty():
                        self._get_exp(exp_queue, should_sample)
                    
                    # Check if any of the environments are done running
                    # Check if any of the environments are done running
                    unfinished_results = []
                    for result in results:
                        if result.ready():
                            org_idx, total_reward, behavior = result.get()
                            total_fitness += total_reward

                            # Update the organism statistics and archive if at training phase
                            if not should_sample:
                                org = self.population.orgs[org_idx]
                                org.behavior = behavior
                                org.update_fitness(total_reward)
                                add_to_archive(
                                    org,
                                    self.archive,
                                    self.archive_species_ids,
                                    self.kdt,
                                    self.population.org_id_to_species[org.id])
                            
                            # Update the fitness range
                            if max_fitness is None or total_reward > max_fitness:
                                max_fitness = total_reward
                            
                            if min_fitness is None or total_reward < min_fitness:
                                min_fitness = total_reward
                        else:
                            unfinished_results.append(result)
                    
                    results = unfinished_results

                # Get the remaining experiences
                while not exp_queue.empty():
                    self._get_exp(exp_queue, should_sample)

        avg_fitness = total_fitness / len(self.population.orgs)
        fitness_range = max_fitness - min_fitness
        print("TOTAL TRAIN TIME: ", self.total_train_time)
        print("ENV RUN TIME: ", time.time() - start_time)
        self._population_to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.total_eval += len(self.population.orgs)
        return max_fitness, avg_fitness, fitness_range, total_fitness