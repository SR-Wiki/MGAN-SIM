# MGAN-SIM
*Long-term, high-fidelity super-resolution structured illumination microscopy via Markovian discriminator*

## **1.Installation**

* python 3.9.18 

* Pytorch 1.13.1, CUDA11.6 and CUDNN

    ```
    conda create -n MGAN python=3.9.18
    ```
    ```
    conda activate MGAN
    ```
    ```
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch\_stable.html
    ```

* Python Packages:
  You can install the required python packages by the following command:
  
    ```
    pip install -r requirements.txt
    ```

## **2.training**

### 1. Prepare the data



### 2. Start training

  Users can run directly from the command line:

  ```
  python train.py --dataset "Example" --discriminator "n_layers46" --save_path "./result/" --n_epochs "50" --n_epochs_decay "50" --batch_size "2" 
  ```

  Users can choose three discriminator fields: "n_layers16"、"n_layers46"、"n_layers70"

## **3.inference**

  ```
  python inference.py --dataset "Example" --data_path "Users own path/raw_data"
  ```
