# RAPO

RAPO is a model-based reinforcement learning algorithm designed for offline scenarios. This algorithm models state transitions as Gaussian distributions and uses ensemble models to predict the uncertainty of outcomes. Unlike previous algorithms, we innovatively introduce the concept of long-term risk: we fit the uncertainty into the long-term risk of each action, termed as riskQ, through the Bellman equation. Unlike the Q-function, the agent should prefer actions with smaller riskQ. We incorporate this element into the original SAC framework, enabling SAC to consider the risks inherent in offline scenarios during policy updates. Moreover, in the paper, we also demonstrate that this algorithm has an excellent lower bound on performance. For specific implementations and theoretical proofs, please refer to the original paper.



## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate SimpleCQL
```

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments

You can run CQL experiments using the following command:
```
python -m SimpleCQL.conservative_sac_main \
    --env 'halfcheetah-medium-v0' \
    --logging.output_dir './experiment_output'
```

If you want to run on CPU only, just add the `--device='cpu'` option.


## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)



