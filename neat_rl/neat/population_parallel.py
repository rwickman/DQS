import random
from torch.multiprocessing import Pool
import copy
try:
    import torch_xla.core.xla_model as xm
except:
    pass

from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.util import get_device


class GradientPopulationParallel(GradientPopulation):
    def __init__(self, args, td3ga):
        super().__init__(args, td3ga)
    
    def breed(self, idx, cur_species):
        print("BREED", idx, device)

        new_org = super().breed(cur_species)
        return idx

    def _breed_new_orgs(self, cur_species, num_spawn):
        print("BREEDING NEW ORGS")

        device = get_device()
        
        # Transfer all to CPU
        for i in range(len(cur_species.orgs)):
            cur_species.orgs[i].net = cur_species.orgs[i].net.to("cpu")
        self.td3ga.critic = self.td3ga.critic.to("cpu")
        self.td3ga.replay_buffer.device = "cpu"

        # Create the pool of workers
        results = []
        with Pool(num_spawn) as pool:
            for i in range(num_spawn):                
                worker_args = (i, cur_species)
                print("APPLYING")
                results.append(
                    pool.apply_async(self.breed, worker_args))

            print("WAITING FOR THEM TO BE DONE")
            
            while len(results) > 0:
                unfinished_results = []
                for result in results:
                    if result.ready():
                        print("GETTING RESULT")
                        new_org = result.get()
                        print(new_org)
                        # cur_species.orgs.append(new_org)
                    else:
                        unfinished_results.append(result)
                results = unfinished_results
            self.td3ga.critic = self.td3ga.critic.to(get_device())