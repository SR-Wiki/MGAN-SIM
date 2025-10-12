**MGAN-SIM**

*Long-term, high-fidelity super-resolution structured illumination microscopy via Markovian discriminator*



* Installation 

python 3.9.18 

Pytorch 1.13.1, CUDA11.6 and CUDNN

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch\_stable.html



You can install the required python packages by the following command:

pip install -r requirements.txt





traing

python train.py --dataset "Example" --discriminator "n\_layers46" --save\_path "./result/" --n\_epochs "50" --n\_epochs\_decay "50" --batch\_size "2" 

