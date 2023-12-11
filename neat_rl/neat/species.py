

class Species:
    def __init__(self, args, id=0):
        self.args = args
        self.orgs = []
        self.species_id = id
        self.age = 0
        self.last_updated = 0

        # Max over the fitness averaged over all organisms 
        self.best_avg_fitness = -1e5

        # The max average fitness over all organisms 
        self.best_max_fitness = -1e5
        
        self.best_avg_diversity = -1e5

        # Max best fitness across all organisms
        self.max_total_fitness = -1e5
    
    def add(self, org):
        self.orgs.append(org)
    
    def first(self):
        return self.orgs[0]

    def update(self):
        self.age += 1
        cur_avg_fitness = sum([o.avg_fitness for o in self.orgs]) / len(self.orgs)
        self.best_avg_fitness = max(cur_avg_fitness, self.best_avg_fitness)
        cur_best_max_fitness = self.best_max_fitness
        self.best_max_fitness = max([o.avg_fitness for o in self.orgs] + [self.best_max_fitness])
        self.max_total_fitness = max([o.best_fitness for o in self.orgs] + [self.max_total_fitness])
        self.best_avg_diversity = max([o.bonus_avg for o in self.orgs] + [self.best_avg_diversity])

        return self.best_max_fitness > cur_best_max_fitness

    @property
    def avg_fitness(self):
        return sum([o.avg_fitness for o in self.orgs]) / len(self.orgs)
    
    @property
    def max_fitness(self):
        return max([o.avg_fitness for o in self.orgs])

    @property
    def best_org(self):
        """Get the best organism from the species."""
        return sorted(self.orgs, key=lambda o: o.avg_fitness, reverse=True)[0]
    
    def snapshot(self):
        """Get a snapshot of this species, excluding the networks themselves."""
        # Record the species level information
        snapshot_dict = {
            "species_id": self.species_id,
            "age": self.age,
            "last_updated": self.last_updated,
            "best_avg_fitness": self.best_avg_fitness,
            "best_max_fitness": self.best_max_fitness,
            "best_avg_diversity": self.best_avg_diversity,
            "max_total_fitness": self.max_total_fitness,
            "cur_avg_fitness": self.avg_fitness,
            "cur_max_fitness": self.max_fitness,
        }

        # Record the organism level information
        snapshot_dict["orgs"] = {}
        for o in self.orgs:
            snapshot_dict["orgs"][o.id] = o.snapshot()
        
        return snapshot_dict
