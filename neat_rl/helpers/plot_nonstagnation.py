"""Get statistics from a stagnated species."""
import argparse, json, os
import numpy as np

from neat_rl.helpers.saving import load_stagnation
from neat_rl.neat.stagnation import Stagnation

# Iterate over best_metric history for a species
# Find the number of generations it took to get the best species metric
def get_best_org(best_metric_history):
    """Get the best average fitness from an organism
        and the number of generations to took to get there."""

    # Iterate over all the best metrics to find when it is equal to the max
    max_org_avg = -1e9
    max_history = None
    for history in best_metric_history:
        if history["species_snapshot"]["best_max_fitness"] > max_org_avg:
            max_org_avg = history["species_snapshot"]["best_max_fitness"]
            max_history = history

    # Get how many generations it took to get this maximum
    orgs = max_history["species_snapshot"]["orgs"]
    
    for org_id, org in orgs.items():
        if org["avg_fitness"] == max_org_avg:
            num_generations = org["generation"]

    return max_org_avg, num_generations




def print_stats(save_dir, num_species):
    """Compute the Pearson correlation coefficient for different metrics."""
    results_dirs = [
        os.path.join(save_dir, d) for d in os.listdir(save_dir)
    ]
    max_fitness = []
    avg_elite_age = []
    max_elite_age = []
    generation = []
    avg_fitness = []
    species_age = []
    num_resets = []

    # Number of generation it took for an organsim to get the maximum average fitness
    num_gen_to_get_best = []
    max_org_avg = []

    for d in results_dirs:
        # Load the population
        pop_file = os.path.join(d, "pop.json")
        with open(pop_file) as f:
            pop_dict = json.load(f)

        # Load the stagnation class
        stagnation = Stagnation(None)
        load_stagnation(pop_dict["stagnation"], stagnation)
        
        # Get the max fitness species and average age for each species
        species_list = pop_dict["species_list"]
        for species in species_list:
            species_id = species["id"]

            # Get the age of the elites from each species
            non_elite_species = 0
            age_vals = []
            avg_generation = []
            max_age = -1e9
            avg_fitness_vals = []
            for org_id in species["org_ids"]:
                org = [o for o in pop_dict["orgs"] if o["id"] == org_id][0]

                if org["age"] > 0:
                    age_vals.append(org["age"])
                    max_age = max(max_age, org["age"])
                else:
                    non_elite_species += 1
                avg_generation.append(org["generation"])
                avg_fitness_vals.append(org["avg_fitness"])

            # Sanity-check
            # assert non_elite_species == 4

            # Only add if the species didn't recently reset
            if len(age_vals) > 0 and species["age"] > 5:
                # Compute the average elite age
                avg_age = sum(age_vals) / len(age_vals)
                avg_elite_age.append(avg_age)                
                generation.append(sum(avg_generation) / len(avg_generation))
                avg_fitness.append(sum(avg_fitness_vals) / len(avg_fitness_vals))

                # Add the max age
                max_elite_age.append(max_age)
                
                # Sum the fitness scores
                max_fitness.append(species["max_total_fitness"])

                species_age.append(species["age"])
                stag_history = stagnation.stagnation_history[species_id]
                num_resets.append(len(stag_history))
                
                
                # Get the number of generations it took to reach the best average fitness
                max_org_avg_val, gen = get_best_org(stagnation.best_metric_history[species_id])
                num_gen_to_get_best.append(gen)
                max_org_avg.append(max_org_avg_val)
                
    
    # Average the values across all the results
    # avg_fitness = [f / len(results_dirs) for f in max_fitness]
    # avg_elite_age = [age / len(results_dirs) for age in avg_elite_age]
    #np.corrcoef()
    print(f"NUMBER OF SAMPLES: {len(avg_fitness)}")
    metrics = [
        ("max_fitness", max_fitness),
        ("avg_elite_age", avg_elite_age),
        ("max_elite_age", max_elite_age),
        ("generation", generation),
        ("avg_fitness", avg_fitness),
        ("species_age", species_age),
        ("num_resets", num_resets),
        ("num_gen_to_get_best", num_gen_to_get_best),
        ("max_org_avg", max_org_avg),
    ]

    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            name_1, metric_1 = metrics[i]
            name_2, metric_2 = metrics[j]
            cur_pearson_coef = np.corrcoef(metric_1, metric_2)[0, 1]
            print(f"{name_1} & {name_2} pearson coef: {cur_pearson_coef}")

    # print(sum(num_gen_to_get_best) / len(num_gen_to_get_best))
    # print(sorted(num_gen_to_get_best))
    print(f"\nMedian generation to get best {np.median(num_gen_to_get_best)}")
    print(f"Median max org average {np.median(max_org_avg)}")
    print("num_gen_to_get_best", num_gen_to_get_best)
    print("max_org_avg", max_org_avg)
    
    # corr_coef = np.corrcoef(np.array(max_fitness), np.array(avg_elite_age))
    # print("MAX FITNESS & AVG ELITE AGE COEF",corr_coef[0, 1], "\n")
    
    # corr_coef = np.corrcoef(np.array(max_fitness), np.array(max_elite_age))
    # print("MAX FITNESS & MAX ELITE AGE COEF", corr_coef[0, 1], "\n")
    
    # corr_coef = np.corrcoef(np.array(max_fitness), np.array(generation))
    # print("MAX FITNESS & GENERATION COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(avg_fitness), np.array(generation))
    # print("AVG FITNESS & GENERATION COEF", corr_coef[0, 1], "\n")
    
    # corr_coef = np.corrcoef(np.array(avg_fitness), np.array(avg_elite_age))
    # print("AVG FITNESS & AVG ELITE AGE COEF", corr_coef[0, 1], "\n")
    
    # corr_coef = np.corrcoef(np.array(avg_fitness), np.array(max_elite_age))
    # print("AVG FITNESS & MAX ELITE AGE COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(avg_elite_age), np.array(generation))
    # print("AVG ELITE AGE & GENERATION COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(max_elite_age), np.array(generation))
    # print("MAX ELITE AGE & GENERATION COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(species_age), np.array(max_fitness))
    # print("SPECIES AGE & MAX FITNESS COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(species_age), np.array(avg_fitness))
    # print("SPECIES AGE & AVG. FITNESS COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(num_resets), np.array(max_fitness))
    # print("NUM RESETS & MAX FITNESS COEF", corr_coef[0, 1], "\n")

    # corr_coef = np.corrcoef(np.array(num_resets), np.array(avg_fitness))
    # print("NUM RESETS & AVG. FITNESS COEF", corr_coef[0, 1], "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
        help="Directory that contains all the models results.")
    parser.add_argument("--num_species", type=int, default=8,
        help="Number of species.")

    args = parser.parse_args()
    print_stats(args.results_dir, args.num_species)