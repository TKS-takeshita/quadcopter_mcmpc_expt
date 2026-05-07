#Pytorch のインストール

##1. 仮想環境の作成

```bash
python3 -m venv ~/venv_torch
source ~/venv_torch/bin/activate
```

##2. Pytorchのインストール (CUDA 12系)
```bash
pip install --upgrade pip
pip install numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```



