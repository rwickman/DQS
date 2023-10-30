import json, os
import torch

from neat_rl.neat.population import GradientPopulation
from neat_rl.neat.organism import Organism
from neat_rl.neat.stagnation import SpeciesMetrics

def create_org_dict(org, prefix_dir):
    org_dict = org.snapshot()
    model_file = os.path.join(prefix_dir, f"net_{org.id}.pt") 

    net_dict = {
        "model": org.net.state_dict(),
        "optimizer": org.optimizer.state_dict(),
        "lr_scheduler": org.lr_scheduler.state_dict()
    }

    torch.save(net_dict, model_file)
    org_dict["network"] = model_file

    return org_dict

def serialize_stagnation(stagnation):
    stag_dict = {
        "metrics": {}
    } 

    # Save the current performance metrics of each species
    for species_id, metrics in stagnation.species_metrics.items():
        stag_dict["metrics"][species_id] = {
            "last_update": metrics.last_update,
            "best_metric": metrics.best_metric
        }
    
    # Save the history of the performance metrics of each species
    stag_dict["stagnation_history"] = stagnation.stagnation_history

    # Save the history of the time the best_metric threshold was reached
    stag_dict["best_metric_history"] = stagnation.best_metric_history
    
    return stag_dict

def save_population(population, save_dir):
    save_file = os.path.join(save_dir, "pop.json")
    
    prefix_dir = os.path.join(save_dir, "nets")
    if not os.path.exists(prefix_dir):
        os.mkdir(prefix_dir)
    else:
        # Remove all the old models
        for model_file in os.listdir(prefix_dir):
            if ".pt" in model_file:
                os.remove(os.path.join(prefix_dir, model_file))

    orgs = []
    for org in population.orgs:
        org_dict = create_org_dict(org, prefix_dir)
        orgs.append(org_dict)
    
    # Save the species list
    species_list = []
    for species in population.species_list:
        species_dict = {
            "id": species.species_id,
            "org_ids" : [org.id for org in species.orgs],
            "age": species.age,
            "best_avg_fitness": species.best_avg_fitness,
            "best_max_fitness": species.best_max_fitness,
            "max_total_fitness": species.max_total_fitness,
            "best_avg_diversity": species.best_avg_diversity,
        }
        
        species_list.append(species_dict)

    # Save the base organism
    base_org = create_org_dict(population.base_org, prefix_dir)
    serialize_stagnation
    stag_dict = serialize_stagnation(population.stagnation)
    # Aggregate everything into this population dictionary
    pop_dict = {
        "orgs": orgs,
        "generation": population.generation,
        "cur_id": population.cur_id,
        "base_org": base_org,
        "species_list": species_list,
        "org_id_to_species": population.org_id_to_species,
        "stagnation": stag_dict
    }

    with open(save_file, "w") as f:
        json.dump(pop_dict, f)
    
    return pop_dict

def _load_organism(args, org_dict, base_actor):
    net = base_actor.copy(transfer_weights=False)
    net_path = os.path.join(args.save_dir, "nets", org_dict["network"].split("/")[-1])
    
    net_dict = torch.load(net_path)
    net.load_state_dict(net_dict["model"])

    org = Organism(args, net, org_dict["generation"], org_dict["id"])
    org.optimizer.load_state_dict(net_dict["optimizer"])
    if "lr_scheduler" in net_dict:
        org.lr_scheduler.load_state_dict(net_dict["lr_scheduler"])

    org.behavior = org_dict["behavior"]
    if "age" in org_dict:
        org.age = org_dict["age"]
    
    if "best_fitness" in org_dict:
        org.best_fitness = org_dict["best_fitness"]
    
    if "bonus_avg" in org_dict:
        org.bonus_avg = org_dict["bonus_avg"]

    if "bonus_best" in org_dict:
        org.bonus_best = org_dict["bonus_best"]

    if "avg_fitness" in org_dict:
        # Bit of a heuristic to assume avg_fitness is fitness sum
        org._num_updates = org_dict["num_updates"]
        org._fitness_avg = org_dict["avg_fitness"]
    
    if "parents" in org_dict:
        org.parents = org_dict["parents"]
    
    return org

def load_stagnation(stag_dict, stagnation):
    print(stag_dict.keys())
    if "metrics" in stag_dict:
        for species_id, metrics in stag_dict["metrics"].items():
            stagnation.species_metrics[int(species_id)] = SpeciesMetrics(
                metrics["last_update"],
                metrics["best_metric"]
            )
    else:
        for i in range(8):
            stagnation.species_metrics[i] = SpeciesMetrics(
                stag_dict[str(i)]["last_update"],
                stag_dict[str(i)]["best_metric"]
            )

        
    if "stagnation_history" in stag_dict:
        for species_id, history in stag_dict["stagnation_history"].items():
            stagnation.stagnation_history[int(species_id)] = stag_dict["stagnation_history"][species_id]
    
    if "best_metric_history" in stag_dict:
        for species_id, history in stag_dict["best_metric_history"].items():
            stagnation.best_metric_history[int(species_id)] = stag_dict["best_metric_history"][species_id]


def load_population(args, td3ga, base_actor):
    prefix_dir = os.path.join(args.save_dir, "nets")
    save_file = os.path.join(args.save_dir, "pop.json")

    # Load the population dictionary
    with open(save_file) as f:
        pop_dict = json.load(f)
    
    population = GradientPopulation(args, td3ga)
    population.cur_id = pop_dict["cur_id"]
    population.generation = pop_dict["generation"]
    population.base_org = _load_organism(args, pop_dict["base_org"], base_actor)
    if "org_id_to_species" in pop_dict:
        for k, v in pop_dict["org_id_to_species"].items():
            population.org_id_to_species[int(k)] = v

    
    # Load the organisms
    org_index = {} # Used to quickly retrieve organisms
    for org_dict in pop_dict["orgs"]:
        org = _load_organism(args, org_dict, base_actor)
        population.orgs.append(org)
        org_index[int(org.id)] = org

    # Load the species
    for species_dict in pop_dict["species_list"]:
        species = population._create_species()
        species.age = species_dict["age"]
        if "best_avg_fitness" in species_dict:
            species.best_avg_fitness = species_dict["best_avg_fitness"] 
        if "best_max_fitness" in species_dict:
            species.best_max_fitness = species_dict["best_max_fitness"]
        if "max_total_fitness" in species_dict:
            species.max_total_fitness = species_dict["max_total_fitness"]

        if "best_avg_diversity" in species_dict:
            species.best_avg_diversity = species_dict["best_avg_diversity"]
        # Add the organisms to the species
        for org_id in species_dict["org_ids"]:
            species.add(org_index[int(org_id)])

    load_stagnation(pop_dict["stagnation"], population.stagnation)

    return population