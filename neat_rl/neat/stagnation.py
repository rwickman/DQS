from dataclasses import dataclass

@dataclass
class SpeciesMetrics:
    last_update: int = 0
    best_avg_fitness: float = None

class Stagnation:
    """Detect stagnation in a species."""
    def __init__(self, args):
        self.args = args
        self.species_metrics: dict = {}

    def add_species(self, species):
        self.species_metrics[species.species_id] = SpeciesMetrics()

    def update(self, species):
        """Update the stagnation count for a species.

        Returns:
            a bool that indicates if a species is stagnant or not.
        """
        species_id = species.species_id
        if self.args.stagnation_metric == "max":
            fitness_metric = species.max_fitness
        else:
            fitness_metric = species.avg_fitness


        if (self.species_metrics[species_id].best_avg_fitness == None
            or self.species_metrics[species_id].best_avg_fitness < fitness_metric):
            self.species_metrics[species_id].best_avg_fitness = fitness_metric
            self.species_metrics[species_id].last_update = 0
        else:
            self.species_metrics[species_id].last_update += 1

        return self.species_metrics[species_id].last_update >= self.args.max_stagnation

    def reset(self, species):
        self.species_metrics[species.species_id] = SpeciesMetrics()
        