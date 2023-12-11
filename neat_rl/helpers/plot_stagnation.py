import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np
from neat_rl.helpers.saving import load_stagnation
from neat_rl.neat.stagnation import Stagnation

def load_pop_dict(save_dir):
    pop_file = os.path.join(save_dir, "pop.json")
    
    with open(pop_file) as f:
        pop_dict = json.load(f)
    
    return pop_dict 

def plot_stag_history(pop_dict):
    # Load the stagnation class
    stagnation = Stagnation(None)
    load_stagnation(pop_dict["stagnation"], stagnation)
    
    # Get the best fitnness it got before it stagnated
    for i in stagnation.stagnation_history:
        best_metrics = []
        for stag_point in stagnation.stagnation_history[i]:
            best_metrics.append(stag_point["best_metric"])
    
        plt.plot(best_metrics)
    plt.show()

    # Plot the best total fitness it has gotten at the time of stagnation
    for i in stagnation.stagnation_history:
        max_total_fitness = []
        for stag_point in stagnation.stagnation_history[i]:
            max_total_fitness.append(stag_point["species_snapshot"]["max_total_fitness"])
        plt.plot(max_total_fitness)
    plt.show()


    # Plot the history of when it reached a locally best fitness
    best_metrics = []
    for stag_point in stagnation.best_metric_history[0]:
        best_metrics.append(stag_point["best_metric"])
    
    plt.plot(best_metrics)
    plt.show()


def stagnation_stats(pop_dict):
    """Get statistics over the species for stagnation."""

    # Load the stagnation class
    stag = Stagnation(None)
    load_stagnation(pop_dict["stagnation"], stag)
    
    # Get the average age org age, species age, and generation before resetting
    stag_stats = {}
    for species_id in stag.stagnation_history:
        # Setup the statistics
        species_stats = {
            "num_resets": 0,
            "species_age": [],
            "avg_org_age": [],
            "avg_org_gen": [],
            "max_org_age": [],
        }

        stag_history = stag.stagnation_history[species_id]
        species_stats["num_resets"] = len(stag_history)
        
        
        for snapshot in stag.stagnation_history[species_id]:
            species_snapshot = snapshot["species_snapshot"]

            # Add the species age before resetting
            species_stats["species_age"].append(
                species_snapshot["age"])
            
            # Get the statistics over the organisms
            age_sum = 0
            gen_sum = 0
            max_age = -1e9
            # TODO: FOR AGE YOU NEED TO REMOVE NON-ELITES SINCE THEIR AGE WILL BE 0
            for org_id, org in species_snapshot["orgs"].items():
                age_sum += org["age"]
                gen_sum += org["generation"]
                max_age = max(max_age, org["age"])

            species_stats["avg_org_age"].append(age_sum / len(species_snapshot["orgs"]))
            species_stats["avg_org_gen"].append(gen_sum / len(species_snapshot["orgs"]))
            species_stats["max_org_age"].append(max_age)

        print(species_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True,
        help="Directory that contains the results.")
    
    args = parser.parse_args()
    pop_dict = load_pop_dict(args.save_dir)
    # plot_stag_history(pop_dict)
    stagnation_stats(pop_dict)

