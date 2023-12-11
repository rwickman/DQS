import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
from matplotlib import rc,rcParams
from matplotlib.ticker import MaxNLocator

class CustomFormatter(ScalarFormatter):
    def _set_format(self):
        super()._set_format()
        self.format = self.format.replace("mathdefault", "textbf")
        if "1.0" in self.format:
            self.format = self.format.replace("1.0", "1.1")

    def get_offset(self):
        offset = super().get_offset()
        if offset:
            new_offset = offset.replace("mathdefault", "mathbf")
            return new_offset
        else:
            return offset
            
# plt.grid(linestyle="dashed")
# sns.set(style="whitegrid")
sns.set_style("whitegrid", {
    'grid.linestyle': '--'
 })


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#rc('font', weight='bold')
benchmark_results_dirs = [
    "../NEAT_RL/results_PGA/results_PGA-MAP-Elites",
    "../NEAT_RL/results_PGA/results_MAP-Elites-ES",
    "../NEAT_RL/results_PGA/results_TD3",
    "../NEAT_RL/results_PGA/results_CMA-MAP-Elites",
    "../NEAT_RL/results_PGA/results_QDRL"
]
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

my_results = "models/walker"
# env = "Walker"
eval_nums = [20000, 40000, 60000, 80000, 100000]
x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]


def get_my_stats(my_results):    
    my_results_dirs = [os.path.join(my_results, d) for d in os.listdir(my_results)]
    fitness_vals = [[] for _ in range(len(eval_nums))]
    qd_vals = [[] for _ in range(len(eval_nums))]
    coverage_vals = [[] for _ in range(len(eval_nums))]
    min_fitness = 1e9

    for d in my_results_dirs:
        # Open up the train JSON file 
        train_dict_file = [os.path.join(d, f) for f in os.listdir(d) if "train_dict" in f][0]
        with open(train_dict_file) as f:
            train_dict = json.load(f)

        # Open the archive to retrieve minimum fitness
        for i in range(len(eval_nums)):
            cur_idx = (eval_nums[i] - 1) // 64
            if i == len(eval_nums) - 1 or cur_idx > len(train_dict["max_fitness"]):
                cur_idx = len(train_dict["max_fitness"]) - 1
            

            fitness_vals[i].append(train_dict["max_fitness"][cur_idx])
            qd_vals[i].append(train_dict["total_fitness_archive"][cur_idx])
            coverage_vals[i].append(train_dict["coverage"][cur_idx])
    
    if "walker" in my_results.lower():
        print(f"Walker MAX FITNESS {max(fitness_vals[i])} AVG FITNESS {sum(fitness_vals[i])/len(fitness_vals[i])} AVG QD SCORE {sum(qd_vals[i])/len(qd_vals[i])}" )
    elif "halfcheetah" in my_results.lower():
        print(f"HalfCheetah MAX FITNESS: {max(fitness_vals[i])} AVG FITNESS {sum(fitness_vals[i])/len(fitness_vals[i])}")
    elif "hopper" in my_results.lower():
        print(f"HOPPER MAX FITNESS {max(fitness_vals[i])} AVG FITNESS {sum(fitness_vals[i])/len(fitness_vals[i])} AVG QD SCORE {sum(qd_vals[i])/len(qd_vals[i])}" )
    else:
        print(f"ANT MAX FITNESS {max(fitness_vals[i])} AVG FITNESS {sum(fitness_vals[i])/len(fitness_vals[i])} AVG QD SCORE {sum(qd_vals[i])/len(qd_vals[i])}" )
    return fitness_vals, qd_vals, coverage_vals 

def get_their_stats(root_dir, eval_num):
    """Get the metrics from other methods."""
    results_dirs = os.listdir(root_dir)
    min_fitness = 1e-9
    fitness_vals = [[] for _ in range(eval_num)]
    qd_vals = [[] for _ in range(eval_num)]
    min_fitness = [[] for _ in range(eval_num)]
    coverage_vals = [[] for _ in range(eval_num)]
    

    for path_dir in results_dirs[:10]:
        if ".DS_Store" in path_dir:
            continue
        results_path = os.path.join(root_dir, path_dir)
        csv_file = [p for p in os.listdir(results_path) if ".csv" in p][0]
        csv_path = os.path.join(results_path, csv_file)
        if ".swp" in csv_path:
            continue
        df = pd.read_csv(csv_path)

        for i in range(eval_num):
            fitness_vals[i].append(df["max_fitness"][i])
            qd_vals[i].append(df["sum_fitness"][i])
            coverage_vals[i].append(df["coverage"][i])
            min_fitness = min(df["min_fitness"][i], min_fitness)

    return fitness_vals, qd_vals, coverage_vals, min_fitness


def shade_bounds(mean_vals, std_vals, ax, x_vals):
    plots = [r for r in ax.get_children() if type(r)==Line2D]
    line_color = plots[-1].get_color()
    for i in range(len(eval_nums) - 1):
        ax.fill_between(
            np.linspace(x_vals[i], x_vals[i+1]),
            np.linspace(std_vals[i][0], std_vals[i + 1][0]), # Shade lower bound
            np.linspace(std_vals[i][1], std_vals[i+1][1]), # Shade upper bound
            alpha=0.15,
            color=line_color
        )

    
def construct_df(vals, x_key, y_key, x_vals):
    d = {
        y_key: [],
        x_key: []
    }
    for j in range(len(eval_nums)):
        for k in range(len(vals[j])):
            d[y_key].append(vals[j][k])
            d[x_key].append(x_vals[j])

    d[y_key] = np.array(d[y_key])
    d[x_key] = np.array(d[x_key])
    return pd.DataFrame(d)

def plot(max_fitness, qd_scores, coverage, axs, ax_idx):
    x_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    line_plots = []
    
    for i in range(len(max_fitness)):
        max_fitness_df = construct_df(max_fitness[i], "Evaluation", "Max-Fitness", x_vals)
        qd_score_df = construct_df(qd_scores[i], "Evaluation", "QD-Score", x_vals)
        coverage_df = construct_df(coverage[i], "Evaluation", "Coverage", x_vals)


        sns.lineplot(ax=axs[0][ax_idx], data=max_fitness_df, x="Evaluation", y="Max-Fitness", ci="sd")
        
        axes_subplot = sns.lineplot(ax=axs[1][ax_idx], data=qd_score_df, x="Evaluation", y="QD-Score", ci="sd")
        axes_subplot = sns.lineplot(ax=axs[2][ax_idx], data=coverage_df, x="Evaluation", y="Coverage", ci="sd")

        line_plots.append([r for r in axs[2][ax_idx].get_children() if type(r)==Line2D][-1])

    return line_plots

if __name__ == "__main__":
    envs = ["Walker", "HalfCheetah", "Ant", "Hopper"]
    labels = ["DQS (ours)", "PGA-MAP-Elites", "MAP-Elites-ES", "TD3", "CMA-MAP-Elites", "QD-PG"]

    fig, axs = plt.subplots(3, 4, figsize=(20, 11))
    for i, env in enumerate(envs):
        max_fitness_list = []
        qd_score_list = []
        coverage_list = []
        min_fitness = 0

        # Plot DQS results
        my_results = f"models/{env.lower()}"
        fitness, qd_scores, coverage = get_my_stats(my_results)
        
        max_fitness_list.append(fitness)
        qd_score_list.append(qd_scores)
        coverage_list.append(coverage)

        # Plot the benchmarks
        for d in benchmark_results_dirs:
            method = d.split("_")[-1]
                
            d = os.path.join(d, f"results_{env}_{method}")
            max_fitness, qd_score, coverage, their_min_fitness = get_their_stats(d, 5)
            min_fitness = min(min_fitness, their_min_fitness)
            
            max_fitness_list.append(max_fitness)
            qd_score_list.append(qd_score)
            coverage_list.append(coverage)


        # Adjust the QD scores
        # print(qd_score_list)
        # qd_score_list = np.array(qd_score_list)
        # coverage_list = np.array(coverage_list)
        # print(min_fitness)
        # qd_score_list = qd_score_list - coverage_list * min_fitness

        cur_line_plots = plot(max_fitness_list, qd_score_list, coverage_list, axs, i)
        
        if "Hopper" in env:
            line_plots = cur_line_plots

        axs[0][i].set_title("QD" + env, fontsize=24)
        axs[2][i].set_xlabel("Evaluations (x$10^5$)", fontsize=20)
        axs[0][i].set_xlabel(None)
        axs[1][i].set_xlabel(None)
        axs[0][i].tick_params(labelbottom=False, labelsize=12)
        axs[1][i].tick_params(labelbottom=False, labelsize=12)
        axs[2][i].tick_params(labelsize=12)

        for j in range(3):
            axs[j, i].spines['top'].set_visible(False)
            axs[j, i].spines['right'].set_visible(False)
            axs[j, i].spines['left'].set_visible(False)


        axs[0][i].set_xticks(x_ticks)
        axs[1][i].set_xticks(x_ticks)
        axs[2][i].set_xticks(x_ticks)
        
        
 
        # print(axs[0][i].yticks())

        axs[0][i].set_ylabel(None)
        axs[1][i].set_ylabel(None)
        axs[2][i].set_ylabel(None)

        axs[0][i].yaxis.set_major_formatter(CustomFormatter(useMathText=True))
        axs[1][i].yaxis.set_major_formatter(CustomFormatter(useMathText=True))
        axs[2][i].yaxis.set_major_formatter(CustomFormatter(useMathText=True))
        #axs[0][i].yaxis.set_major_locator(MaxNLocator(integer=True))
        
        axs[2][i].xaxis.set_major_formatter(CustomFormatter(useMathText=True))
        axs[2][i].ticklabel_format(axis='y', style='sci', scilimits=(2,4))

    
    axs[0][0].set_ylabel("Max-Fitness", fontsize=20)
    axs[1][0].set_ylabel("QD-Score", fontsize=20)
    axs[2][0].set_ylabel("Coverage", fontsize=20)

    lgd = fig.legend(handles=line_plots, labels=labels, loc="lower center", ncol=len(labels), fontsize=18, frameon=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    plt.savefig("Figures/exp_results.pdf", dpi=500, transparent=False)
