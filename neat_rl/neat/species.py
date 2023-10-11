

class Species:
    def __init__(self, args, id=0):
        self.args = args
        self.orgs = []
        self.age = 0
        self.last_updated = 0
        self.species_id = id
        self.adj_fitness = 0
        self.best_avg_fitness = -1e5
        self.best_max_fitness = -1e5
    
    def add(self, org):
        self.orgs.append(org)
    
    def first(self):
        return self.orgs[0]

    def update(self):
        self.age += 1
        cur_avg_fitness = sum([o.avg_fitness for o in self.orgs]) / len(self.orgs)
        self.best_avg_fitness = max(cur_avg_fitness, self.best_avg_fitness)
        self.best_max_fitness = max([o.avg_fitness for o in self.orgs] + [self.best_max_fitness])
        
    @property
    def avg_fitness(self):
        return sum([o.avg_fitness for o in self.orgs]) / len(self.orgs)
    
    @property
    def max_fitness(self):
        return max([o.avg_fitness for o in self.orgs])
        