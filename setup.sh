#!/bin/bash

python3.7 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

git submodule init
git submodule update

wget -c https://leon.bottou.org/_media/projects/infimnist.tar.gz
tar -xzvf infimnist.tar.gz
mv ./infimnist/data ./infimnist_py
rm -rf infimnist
rm infimnist.tar.gz

wget -c https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1
unzip Kather_texture_2016_image_tiles_5000.zip?download=1
rm Kather_texture_2016_image_tiles_5000.zip?download=1

cp ./setup.py ./infimnist_py/setup.py
cd infimnist_py
python setup.py build_ext -if
cd ..

deactivate
