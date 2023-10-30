from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
import random

class Organism:
    def __init__(self, args, net, gen=0, id=0):
        self.args = args
        self.net = net # The network controlling the behavior of the organism
        self.generation = gen # Tells what generation this organism is from
        self.fitness = 0
        self.best_fitness = -100000
        self.id = id
        self._fitness_avg = 0
        self._num_updates = 0
        self.age = 0
        self.behavior = None
        self.bonus_avg = 0
        self.bonus_best = 0
        self.optimizer = Adam(
            net.parameters(),
            lr=self.args.org_lr)
        self.lr_scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lambda e: 0.9)
        self.parents = []

    def copy(self, org_id=0):
        copy_net = self.net.copy(transfer_weights=False)
        copy_org = Organism(self.args, copy_net, self.generation, org_id)
        copy_org.age = self.age
        copy_org.generation = self.generation
        return copy_org

    def update_fitness(self, fitness, diversity_bonus=None):
        self._num_updates += 1
        
        # Calculation the moving average
        self._fitness_avg += (fitness - self._fitness_avg) / self._num_updates

        self.best_fitness = max(fitness, self.best_fitness)
        self.fitness = fitness
        if diversity_bonus:
            self.bonus_avg += (diversity_bonus - self.bonus_avg) / self._num_updates
            self.bonus_best = max(self.bonus_best, diversity_bonus)

    def add_parent(self, parent_id):
        """Add the parent the spawned this organism."""
        self.parents.append(parent_id)

    @property
    def avg_fitness(self):
        return self._fitness_avg 

    def __call__(self, x):
        return self.net(x)
    
    def snapshot(self):
        """Get a snapshot of the metrics of this organisms."""
        return {
            "id": self.id,
            "age": self.age,
            "avg_fitness": self.avg_fitness,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "num_updates": self._num_updates,
            "behavior": list(self.behavior) if self.behavior is not None else None,
            "bonus_avg": self.bonus_avg,
            "bonus_best": self.bonus_best,
            "parents": self.parents.copy(),
        }