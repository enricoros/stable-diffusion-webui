# install CUDA 11.7.1 and CUDNN (latest for CUDA 11.x) in the system

# upgrade pip
pip3 install --upgrade pip
echo

# install PyTorch 1.12.1 with support for CUDA 11.6
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
echo

# convert stable-diffusion-webui/environment.yaml into pip installation instructions
pip3 install "accelerate==0.12.0" "albumentations==0.4.3" "basicsr>=1.3.4.0" "diffusers==0.3.0" "einops==0.3.0" "facexlib>=0.2.3" "gradio==3.1.6" "imageio-ffmpeg==0.4.2" "imageio==2.9.0" "kornia==0.6" "omegaconf==2.1.1" "opencv-python-headless==4.6.0.66" "pandas==1.4.3" "piexif==1.1.3" "pudb==2019.2" "pynvml==11.4.1" "python-slugify>=6.1.2" "pytorch-lightning==1.4.2" "retry>=0.9.2" "streamlit" "streamlit-on-Hover-tabs==1.0.1" "streamlit-option-menu==0.3.2" "streamlit-nested-layout" "test-tube>=0.7.5" "tensorboard" "torch-fidelity==0.3.0" "torchmetrics==0.6.0" "transformers==4.19.2"
pip3 install -e .
pip3 install -e "git+https://github.com/CompVis/taming-transformers#egg=taming-transformers"
pip3 install -e "git+https://github.com/openai/CLIP#egg=clip"
pip3 install -e "git+https://github.com/TencentARC/GFPGAN#egg=GFPGAN"
pip3 install -e "git+https://github.com/xinntao/Real-ESRGAN#egg=realesrgan"
pip3 install -e "git+https://github.com/hlky/k-diffusion-sd#egg=k_diffusion"
pip3 install -e "git+https://github.com/devilismyfriend/latent-diffusion#egg=latent-diffusion"
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
