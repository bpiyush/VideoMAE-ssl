# Create conda env
conda create -y --name videomae python=3.6

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate videomae

# Install dependencies
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.9
DS_BUILD_OPS=1 pip install deepspeed
pip install tensorboardx einops decord jupyterlab ipdb matplotlib opencv-python
