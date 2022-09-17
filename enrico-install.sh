# install CUDA 11.7.1 and CUDNN (latest for CUDA 11.x) in the system

# upgrade pip
pip3 install --upgrade pip
echo

# install PyTorch 1.12.1 with support for CUDA 11.6
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
echo

# convert stable-diffusion-webui/environment.yaml into pip installation instructions
pip3 install "albumentations==0.4.3" "opencv-python>=4.1.2.30" "opencv-python-headless>=4.1.2.30" "pudb==2019.2" "imageio==2.9.0" "imageio-ffmpeg==0.4.2" "pytorch-lightning==1.4.2" "omegaconf==2.1.1" "test-tube>=0.7.5" "einops==0.3.0" "torch-fidelity==0.3.0" "transformers==4.19.2" "torchmetrics==0.6.0" "kornia==0.6" "gradio==3.1.6" "accelerate==0.12.0" "pynvml==11.4.1" "basicsr>=1.3.4.0" "facexlib>=0.2.3" "python-slugify>=6.1.2" "streamlit" "retry>=0.9.2"
pip3 install -e "git+https://github.com/CompVis/taming-transformers#egg=taming-transformers"
pip3 install -e "git+https://github.com/openai/CLIP#egg=clip"
pip3 install -e "git+https://github.com/TencentARC/GFPGAN#egg=GFPGAN"
pip3 install -e "git+https://github.com/xinntao/Real-ESRGAN#egg=realesrgan"
pip3 install -e "git+https://github.com/hlky/k-diffusion-sd#egg=k_diffusion"
pip3 install -e .
echo

# download optional models
mkdir -p src/gfpgan/experiments/pretrained_models
echo "Manually download GFPGAN following the instructions"
mkdir -p src/realesrgan/experiments/pretrained_models
echo "Manually download RealESRGAN following the instructions"
echo "Manually download LDSR following the instructions"
echo

# done
echo "Done, now start scripts/relauncher.py"
echo
