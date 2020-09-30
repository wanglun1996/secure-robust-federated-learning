# $F^2ED-LEARNING:$ Good Fences Make Good Neighbors
This repository contains the evaluation code for the corresponding submission to ICLR'21.

### To reproduce the evaluation results
First, set up the running environment with the setup script.

```bash
./setup.sh
```

Enter the python3.7 virtual environment.
```bash
source ./venv/bin/activate
```

Get the dataset ready.
```bash
cd ./src
python data.py
```

Reproduce the evaluation results by filling in the corresponding parameters:
```bash
python simulate.py --dataset= XX --device XX --mal XX --attack XX --agg XX
```