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
    
    for i in stagnation.stagnation_history:
        best_metrics = []
        for stag_point in stagnation.stagnation_history[i]:
            best_metrics.append(stag_point["best_metric"])
    
        plt.plot(best_metrics)
    plt.show()


    best_metrics = []
    for stag_point in stagnation.best_metric_history[i]:
        best_metrics.append(stag_point["best_metric"])
    
    plt.plot(best_metrics)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True,
        help="Directory that contains the results.")
    
    args = parser.parse_args()
    pop_dict = load_pop_dict(args.save_dir)
    plot_stag_history(pop_dict)

