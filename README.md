# MGAN-SIM
Implementation for "Long-term, high-fidelity super-resolution structured illumination microscopy via Markovian discriminator"

## 🔧 Installation

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

## 💻 Training

   ### 1. Prepare the data

   Users can run the Matlab source code for generating the training data.

   ### 2. Start training

   Users can run directly from the command line:

   ```
   python train.py --dataset "Users own dataset" --sample_interval "100" --discriminator "n_layers46" --save_path "./result/" --n_epochs "50" --n_epochs_decay "50" --batch_size "32" 
   ```

   Users can choose three discriminator fields: "n_layers16"、"n_layers46"、"n_layers70"

   ### * transfer learning

   We provide transfer learning code based on pre-trained models of similar scales, which enables rapid fitting on small sample datasets.
  
   Users can run directly from the command line:

   ```
   python transfer_learning.py --dataset "Users own dataset" --sample_interval "100" --discriminator "n_layers46" --save_path "./result/" --n_epochs "50" --n_epochs_decay "50" --batch_size "32" --pretrained_g "./checkpoint/Users own pre-dataset/netG_model_epoch_100.pth" --pretrained_d  "./checkpoint/Users own pre-dataset/netD_model_epoch_100.pth"
   ```

 * Note: The discriminator should be consistent with the pre-trained model.

## ⚡ Inference

  ```
  python inference.py --dataset "Users own dataset" --data_path "Users own path/raw_data"
  ```
