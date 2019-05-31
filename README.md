# Implementation of Bayesian Generative Adversarial Imitation Learning (BGAIL)

## Requirements
-   python 3.6.6
-   MuJoCo 1.31
-   mujoco-py 0.5.7
-   OpenAI Gym 0.9.0
-   OpenAI Baselines 0.1.5 (in this repository)
-   TensorFlow 1.10.0
-   See [here](SETTING_paper.md) for detailed procedure.

## References
-   [openai/imitation](https://github.com/openai/imitation)
    - GAIL implementation of authors
-   [andrewliao11/gail-tf](https://github.com/andrewliao11/gail-tf)
    - TensorFlow implementation of GAIL
-   [OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/gail)
    - Baseline implementation of GAIL
-   [wsjeon/SVGD](https://github.com/wsjeon/SVGD)
    - TensorFlow implementation of Stein variational gradient descent (SVGD)
-   [BGAIL paper](https://papers.nips.cc/paper/7972-a-bayesian-approach-to-generative-adversarial-imitation-learning.pdf)

## Usage
### Download expert trajectories.
We use the expert trajectories by using the code given by [openai/imitation](https://github.com/openai/imitation).

1.  Download expert trajectories from [this link](https://www.dropbox.com/sh/9uort7161cz93v9/AACJapyvTxDsFC1QLqP1nYNNa?dl=0) to `expert_trajs/`.
2.  Run
    ```bash
    python expert_trajs/convert_h5_to_pkl.py
    ```
    to convert expert trajectories into `*.pkl` files.
    
### Run.
```bash
python run.py 
``` 
will run BGAIL in default setting (see `bgail/defaults.py`.)
