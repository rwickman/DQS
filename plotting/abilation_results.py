import numpy as np 
import argparse, json, os

def avg_results(args):
    model_dirs = [
        os.path.join(args.save_dir, d)
        for d in os.listdir(args.save_dir)
        if args.model_prefix in d
    ]

    print("TAKING AVERAGE OVER", model_dirs)

    fitness_vals = []
    qd_vals = []
    coverage_vals = []


    for model_dir in model_dirs:
        train_dict_file = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if "train_dict" in f][0]

        with open(train_dict_file, "r") as f:
            train_dict = json.load(f)
        fitness_vals.append(train_dict["max_fitness"][-1])
        qd_vals.append(train_dict["total_fitness_archive"][-1])
        coverage_vals.append(train_dict["coverage"][-1])
        print(f"{train_dict_file} MAX FITNESS {fitness_vals[-1]} QD-SCORE {qd_vals[-1]} COVERAGE {coverage_vals[-1]}")
    print(f"MAX FITNESS {round(sum(fitness_vals) / len(fitness_vals), 2)}")
    print(f"QD-Score {round(sum(qd_vals) / len(qd_vals), 2)}")
    print(f"Coverage {round(sum(coverage_vals) / len(coverage_vals), 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_prefix", default="",
        help="Prefix of the models to average the results over.")
    parser.add_argument("--save_dir", required=True,
        help="Directory containing the models.")
    
    args = parser.parse_args()
    avg_results(args)


