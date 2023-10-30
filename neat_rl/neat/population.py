import random

from neat_rl.neat.organism import Organism
from neat_rl.neat.species import Species
from neat_rl.neat.reproduction import GradientReproduction
from neat_rl.neat.stagnation import Stagnation

from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim import Adam
import copy

class GradientPopulation:
    def __init__(self, args, td3ga):
        self.args = args
        self.breeder = GradientReproduction(args)
        self.stagnation = Stagnation(self.args)
        self.td3ga = td3ga
        self.org_id_to_species = {}
        self.species_list = []
        self.cur_id = 1
        self.generation = 0
        self.orgs = []
    
    def setup(self, net):
        self.base_org = Organism(self.args, net)
        self.orgs = self.spawn(self.base_org, self.args.pop_size)
        self.speciate()
        self.org_id_to_species = {}
        total_orgs = 0
        for cur_species in self.species_list:
            total_orgs += len(cur_species.orgs)
            for org in cur_species.orgs:

                assert org.id not in self.org_id_to_species 
                self.org_id_to_species[org.id] = cur_species.species_id
        print("total_orgs", total_orgs, len(self.orgs))
        assert total_orgs == len(self.orgs)

    def _create_species(self):
        species = Species(self.args, len(self.species_list))
        self.species_list.append(species)
        self.stagnation.add_species(species)
        return species

    def speciate(self):
        """Create the initial species and add organisms to them."""        
        # Create the species
        for _ in range(self.args.num_species):
            self._create_species()

        # Create organism for each species
        for i in range(self.args.pop_size):
            cur_species_id = i % self.args.num_species
            self.species_list[cur_species_id].add(self.orgs[i])


    def spawn(self, base_org, pop_size):
        orgs = []
        for i in range(pop_size):
            copy_org = base_org.copy(self.cur_id)
            self.cur_id += 1
            orgs.append(copy_org)
        
        return orgs

    def breed(self, cur_species):
        if random.random() <= self.args.pg_rate:
            # Pick a random parent from the species
            parent_1 = random.choice(cur_species.orgs)
            parent_2 = parent_1

            # Copy the parent network, optimizer, and LR scheduler
            child_net = parent_1.net.copy(transfer_weights=True)
            child_optimizer = Adam(child_net.parameters(), lr=self.args.org_lr)
            child_optimizer.load_state_dict(copy.deepcopy(parent_1.optimizer.state_dict()))
            child_lr_scheduler = MultiplicativeLR(child_optimizer, lr_lambda=lambda e: 0.9)
            child_lr_scheduler.load_state_dict(copy.deepcopy(parent_1.lr_scheduler.state_dict()))

            # Update the child
            self.td3ga.pg_update(child_net, child_optimizer, child_lr_scheduler, cur_species.species_id)
        else:
            if len(cur_species.orgs) > 1:
                parent_1, parent_2 = random.sample(cur_species.orgs, 2)
            else:
                parent_1 = parent_2 = cur_species.orgs[0]


            child_net = self.breeder.reproduce(
                parent_1.net, parent_2.net)

        # Create the new organism
        new_org = Organism(
            self.args, child_net, gen=max(parent_1.generation, parent_2.generation) + 1, id=self.cur_id)
        
        # Set the parents to record generation history
        new_org.parents = parent_1.parents.copy()
        new_org.add_parent(parent_1.id)

        new_org.optimizer = child_optimizer
        new_org.lr_scheduler = child_lr_scheduler

        # Increment the current organism ID
        self.cur_id += 1
        return new_org
    
    def prune_species(self, cur_species):
        species_len = len(cur_species.orgs)

        # Randomize order so that sorting uniform avg fitness is random
        random.shuffle(cur_species.orgs)
        
        # Sort so best organisms are first
        cur_species.orgs.sort(key=lambda x: x.avg_fitness, reverse=True)
        
        # Calculate how many organsims should remain "alive"
        num_live = int(max(self.args.survival_rate * species_len, 1))

        # Remove all but the top
        cur_species.orgs = cur_species.orgs[:num_live]

        num_spawn = species_len - num_live
        return num_spawn

    def _reset_species(self, species):
        self.stagnation.reset(species)
        species.orgs = self.spawn(self.base_org, len(species.orgs))
        species.age = 0
        self.td3ga.replay_buffer.toggle_reset(species.species_id)

    def evolve(self):
        """Remove worst organisms and spawn organsism from breeding best organisms."""
        # Reset the organisms
        self.orgs = []

        # Create next iteration of organisms
        for cur_species in self.species_list:
            did_improve = cur_species.update()
            # Check if the current is an expert
            self.td3ga.replay_buffer.update_expert(cur_species.species_id, did_improve)

            if self.stagnation.update(cur_species):
                # Indicates this species has stananted, so resetting the species
                self._reset_species(cur_species)
            else:
                num_spawn = self.prune_species(cur_species)

                for org in cur_species.orgs:
                    org.age += 1
                
                # Save new orgs in list to prevent breeding with new_org
                new_orgs = []
                for _ in range(num_spawn):
                    new_orgs.append(self.breed(cur_species))
                
                cur_species.orgs.extend(new_orgs)

            # Add species' organsims to list of all organisms
            self.orgs.extend(cur_species.orgs)

        self.org_id_to_species = {}
        for cur_species in self.species_list:
            for org in cur_species.orgs:
                self.org_id_to_species[org.id] = cur_species.species_id
        
        self.generation += 1

        assert len(self.orgs) == self.args.pop_size


    def get_best(self):
        best_fitness = best_org = None
        for org in self.orgs:
            if best_fitness is None or org.best_fitness > best_fitness:
                best_org = org
                best_fitness = org.best_fitness
        
        return best_org
                