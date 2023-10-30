from dataclasses import dataclass

@dataclass
class SpeciesMetrics:
    last_update: int = 0
    best_metric: float = None


class Stagnation:
    """Detect stagnation in a species."""
    def __init__(self, args):
        self.args = args
        self.species_metrics: dict = {}
        self.stagnation_history = {}
        self.best_metric_history = {}

    def add_species(self, species):
        self.species_metrics[species.species_id] = SpeciesMetrics()
        self.stagnation_history[species.species_id] = []
        self.best_metric_history[species.species_id] = []


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

        if (self.species_metrics[species_id].best_metric == None
            or self.species_metrics[species_id].best_metric < fitness_metric):
            # Save the best organisms and their metric
            best_org = species.best_org
            self.best_metric_history[species_id].append({
                "best_metric": fitness_metric,
                "species_snapshot": species.snapshot()
            })

            self.species_metrics[species_id].best_metric = fitness_metric
            self.species_metrics[species_id].last_update = 0
        else:
            self.species_metrics[species_id].last_update += 1

        return self.species_metrics[species_id].last_update >= self.args.max_stagnation

    def reset(self, species):
        # Record the species metrics before it resets
        self.stagnation_history[species.species_id].append({
            "best_metric": self.species_metrics[species.species_id].best_metric,
            "species_snapshot": species.snapshot(),
        })

        # Reset the metrics as all organisms in the species will be replaced
        self.species_metrics[species.species_id] = SpeciesMetrics()
        