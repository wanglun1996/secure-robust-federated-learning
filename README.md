----------------------------------------------------------------

**The codebase is an academic research prototype, and meant to elucidate protocol details and for proofs-of-concept, and benchmarking. It is not meant for deployment currently.**

----------------------------------------------------------------

# F2ED-LEARNING: Attacks and Byzantine-Robust Aggregators in Federated Learning

[![CircleCI](https://circleci.com/gh/wanglun1996/secure-robust-federated-learning.svg?style=shield&circle-token=0f78f0ff77f73076dc255f5a8761e1aa8be5abc6)](https://circleci.com/gh/wanglun1996/secure-robust-federated-learning)

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
- [Inner Product Manipulation](https://arxiv.org/abs/1903.03936)

### Byzantine-Robust Aggregators 
We implemented the following Byzantine-robust aggregators in federated learning.

- [Bucketing-filtering](http://arxiv.org/abs/2205.11765)
- [Bucketing-no-regret](http://arxiv.org/abs/2205.11765)
- [Bulyan Krum](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Bulyan Median](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Bulyan Trimmed Mean](http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)
- [Filtering](http://arxiv.org/abs/2205.11765)
- [GAN](http://arxiv.org/abs/2205.11765)
- [Krum](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- [Median](https://proceedings.mlr.press/v80/yin18a)
- [No-regret](http://arxiv.org/abs/2205.11765)
- [Trimmed Mean](https://proceedings.mlr.press/v80/yin18a)
- [Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt)
- [Learning from History](http://proceedings.mlr.press/v139/karimireddy21a.html)
- [Clustering](https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Velicheti2021.pdf)

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
python -m visdom.server -p 8097
```
Then start the training with selected **aggregator** and **attack**, which are specified in `utils/X.yaml`, `X` can be `mnist_params` or `fashion_params`.
```bash
cd ./src/DBA
python main.py --params utils/X.yaml
```

For **GAN** aggregator, run the following command to start training in round `X`:
```bash
python src/simulate_gan.py --current_round=X --attack='noattack' --dataset='MNIST'
python src/gan.py --next_round=X+1 --gan_lr=1e-5
```

Note that `X` starts from `0`, and you may try different hyper-parameters like learning rate in `gan.py` if you use datasets other than `MNIST` or attacks other than `trimmedmean` and `noattack`.

### Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{zhu2022byzantine,
  title={Byzantine-Robust Federated Learning with Optimal Statistical Rates and Privacy Guarantees},
  author={Banghua Zhu and Lun Wang and Qi Pang and Shuai Wang and Jiantao Jiao and Dawn Song and Michael Jordan},
  year={2022},
  url={https://arxiv.org/abs/2205.11765}
}
```

```
@article{wang2020f,
  title={F2ED-LEARNING: Good fences make good neighbors},
  author={Lun Wang and Qi Pang and Shuai Wang and Dawn Song},
  journal={CoRR},
  year={2020},
  url={http://128.1.38.43/wp-content/uploads/2020/12/Lun-Wang-07-paper-Lun.pdf}
}
```

### Acknowledgement

The code of evaluation on DBA attacks largely reuse [the original implementation](https://github.com/AI-secure/DBA) from the authors of DBA.
