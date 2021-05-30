# Machine-Learning-for-Diagnosis-Irregular-Heart-Rhythms-in-ECG-Signals

Setting and environment
Both Windows and Linux:

Conda environment with Python 3.9:
pandas
scipy
tqdm
sklearn
tensorflow (Windows)
tensorflow-gpu (OzStar)
matplotlib

OzStar environment:
Source activate conda environment
module load anaconda3/5.1.0
GPU modules:
module load cudnn/8.1.0-cuda-11.2.0
module load cuda/11.2.0

Slurm script:
submit_train.sh
72000 required most of the resources.
CNN1D models 500 epoch required 32-64G memory with 1-2 GPU 1-3 hours each.
Bidirectional models 200 epoch required 32-128G memory with 1-2 GPU 3-30 hours each.
2D images pre-processing required 3-5 hours on OzStar environment.
CNN2D with ResNet50 100 epoch required 10-15 hours.

ECG Time-Series Classification
3 pre-processing with 6 classification models were tested: 

Pre-processing:
(3000, 8), Preprocessing.py
(72000, 12), Preprocessing_72000.py
(224, 224, 3), Preprocessing_2D.py

CNNs: 
CNN1D
CNN1D + LSTM
CNN1D + GRU 
Ozstar_training_1D_e500.py
Ozstar_training_72000_1D_e500.py

Bidirectional network:
Bidirectional LSTM
Bidirectional GRU
Ozstar_training_bidirectional_e400.py
Ozstar_training_72000_bidirectional_e400.py

2D:
CNN2D (ResNet50), CNN2D_ResNet50_v2.py
All processes are based on Python only.

Data
China Physiological Signal Challenge 2018 (CPSC2018) Dataset: This dataset consists of 12-channel ECG recordings (lasting from 6s to 60s) from 6877 subjects. Eight kinds of typical CA are included in this dataset. 
http://2018.icbeb.org/Challenge.html
