----------------------------------------------------------------

**The codebase is an academic research prototype, and meant to elucidate protocol details and for proofs-of-concept, and benchmarking. It is not meant for deployment currently.**

----------------------------------------------------------------

# F2ED-LEARNING: Attacks and Byzantine-Robust Aggregators in Federated Learning

This repository contains the evaluation code for the following manuscripts.

- Byzantine-Robust Federated Learning with Optimal Statistical Rates and Privacy Guarantees. Banghua Zhu*, Lun Wang*, Qi Pang*, Shuai Wang, Jiantao Jiao, Dawn Song, Michael Jordan.
- Towards Bidirectional Protection in Federated Learning. Lun Wang*, Qi Pang*, Shuai Wang, Dawn Song. SpicyFL Workshop @ NeurIPS 2020.

### Attacks 
We implemented the following attacks in federated learning.

- [DBA](https://openreview.net/forum?id=rkgyS0VFvr)
- [Krum](https://dl.acm.org/doi/abs/10.5555/3489212.3489304)
- [Model Poisoning](https://proceedings.mlr.press/v97/bhagoji19a.html)
- [Model Replacement](https://proceedings.mlr.press/v108/bagdasaryan20a.html)
- [Trimmed Mean](https://dl.acm.org/doi/abs/10.5555/3489212.3489304)

### Byzantine-Robust Aggregators 
We implemented the following Byzantine-robust aggregators in federated learning.

- [Bucketing-filtering]()
- [Bucketing-no-regret]()
- [Bulyan Krum](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Bulyan Median](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Bulyan Trimmed Mean](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Filtering]()
- [GAN]()
- [Krum](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- [Median](https://proceedings.mlr.press/v80/yin18a)
- [No-regret]()
- [Trimmed Mean](https://proceedings.mlr.press/v80/yin18a)

### Dependency

- conda 4.12.0
- Python 3.7.11
- Screen version 4.06.02 (GNU) 23-Oct-17

First, create a conda virtual environment with Python 3.7.11 and activate the environment.

```bash
conda create -n venv python=3.7.11
conda activate venv
```

Run the following command to install all the required python packages.

```bash
pip install -r requirements.txt
```

### Usage

Reproduce the evaluation results by running the following script. You might want to change the GPU index in the script manually. The current script distributes training tasks to 8 Nvidia GPUs indexed by 0-7.

```bash
./train.sh
```

To run a single Byzantine-robust **aggregator** against a single **attack** on a **dataset**, run the following command with the right system arguments:
```bash
python src/simulate.py --dataset='dataset' --attack='attack' --agg='aggregator'
```

For **DBA** attack, we reuse its [official implementation](https://github.com/AI-secure/DBA).
First open a terminal and run the following command to start Visdom monitor:
```bash
python -m visdom.server -p 8098
```
Then start the training with selected **aggregator** and **attack**, which are specified in `utils/X.yaml`, `X` can be `mnist_params` or `fashion_params`.
```bash
cd ./src/DBA
python main.py --params utils/X.yaml
```

For **GAN** aggregator, run the following command to start training in round `X`:
```bash
python src/sim_copy.py --current_round=X --attack='noattack' --dataset='MNIST'
python src/gan.py --next_round=X+1 --gan_lr=1e-5
```
Note that `X` starts from `0`, and you may try different hyper-parameters like learning rate in `gan.py` if you use datasets other than `MNIST` or attacks other than `trimmedmean` and `noattack`.