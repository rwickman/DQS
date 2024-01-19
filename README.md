# DQS


## Setup
To install this package run:
```shell
pip3 install -e .
pip3 install -r requirements.txt
```
You will also need to install QDGym:
```shell
pip3 install git+https://github.com/ollenilsson19/QDgym.git#egg=QDgym
```

## Training the models
To train the model on the QDHopper environment with a population size of 64 with 8 species:
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --save_dir <path/to/save_dir>
```

To load the model to continue training:
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --save_dir <path/to/save_dir> --load
```


If you want to see population interacting with environment, without training (i.e., render):
```shell
python3 main.py --env QDHopperBulletEnv-v0 --pop_size 64 --num_species 8 --save_dir <path/to/save_dir> --load --render
```

To track the current training results for a model:
```shell
python3 neat_rl/helpers/plot_results.py --save_dir <path/to/model> --env <env_name>
```

For other environments, replace the environment with the one you want to test on.

All environments:
* QDHopperBulletEnv-v0
* QDWalker2DBulletEnv-v0
* QDHalfCheetahBulletEnv-v0
* QDAntBulletEnv-v0

----

## Hyperpameters used in Paper
<p align="center">

| Hyperparameter                               | Value       |
|----------------------------------------------|-------------|
| Population Size                              | 64          |
| Number of Species ($m$)                      | 8           |
| Diversity Reward Scale ($\lambda$)           | 0.05        |
| Species Elites Value ($K$)                   | 4           |
| Maximum Stagnation (max_stag)                   | 16           |
| Expert Buffer Percentage (expert_pct)                   | 0.25           |
| Policy Update Steps (n_grad)                 | 128          |
| Critic Update Frequency (critic_update_freq)     | 8           |
| Policy Hidden Size                           | 128         |
| Species Actor Hidden Size                    | 256         |
| Species Critic Hidden Size                   | 256         |
| Discriminator Hidden Size                    | 256         |
| Species Actor/Critic and Discriminator Learning Rate | 0.003 |
| Policy Learning Rate                         | 0.006       |
| Number of Evaluations (num_eval)             | $10^{5}$    |
| Batch Size ($N$)                             | 256         |
| Discount Factor ($\gamma$)                   | 0.99        |
| Species Target Update Rate ($\tau$)          | 0.005       |
| TD3 Exploration Noise                        | 0.2         |
| TD3 Smoothing Variance ($\sigma$)            | 0.2         |
| TD3 Noise Clip ($c$)                         | 0.5         |
| TD3 Target Update Freq. ($d$)                | 2           |
| Replay Buffer Size                           | $2^{19}$    |

**Table 1:** Hyperparameter values for the DQS algorithm.

</p>

-------
## Cite

```
@article{wickman2023efficient,
  title={Efficient Quality-Diversity Optimization through Diverse Quality Species},
  author={Wickman, Ryan and Poudel, Bibek and Villarreal, Michael and Zhang, Xiaofei and Li, Weizi},
  journal={arXiv preprint arXiv:2304.07425},
  year={2023},
  url={https://arxiv.org/pdf/2304.07425.pdf}
}
```
