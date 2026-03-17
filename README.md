# MLOW: Interpretable Low-Rank Frequency Magnitude Decomposition of Multiple Effects for Time Series Forecasting

[![paper preview](./fig/Decomposition.png)](./fig/Decomposition.pdf)

This is a PyTorch implementation of the paper: [] MLOW: Interpretable Low-Rank Frequency Magnitude Decomposition of Multiple Effects for Time Series Forecasting.  MLOW is an interpretable Fourier-based decomposition method that disentangles multiple effects of certain time series data using learned low-rank components, providing an interpretable decomposition in the temporal domain rather than operating in the complex frequency domain.

If you find this project helpful, please don't forget to give it a ⭐ Star. Thank you very much! 

We'll keep updating this repository with new features and news. Please feel free to contact us if you have any questions or find any bugs. 

## 🛠️ Environment Setup

### Basic Installation
```bash
# Basic setup 
conda create -n VLANeXt python=3.10
conda activate VLANeXt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
conda install -c conda-forge ffmpeg
```


A low-rank learning method for Frequency Magnitude, Hyperplane-NMF, is proposed.

A mathematical mechanism allows flexible selection of initial frequency levels and the input horizon.

### Benchmark Installation

**LIBERO**
```bash
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install .
```

**LIBERO-plus** (Separate env needed)
```bash
cd third_party
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus && pip install .
# Dependencies
apt install libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
pip install -r extra_requirements.txt
conda env config vars set LIBERO_CONFIG_PATH=~/.libero_plus
```
We also need to download the asserts, see [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus).


## 🚀 Training
Droid dataset is for robotics pretraining (used in our real-world experiments), and libero dataset is for benchmark evaluation (used in our benchmark evaluation). The default training setting is for our final VLANeXt framework.
<p align="center">
<img src="imgs/framework.png" alt="framework of vlanext" width="80%"/>
</p>

### 🧪 Design Space Exploration
We provide a tutorial-style guide to configuring the **12 design spaces** from our paper.

👉 **Please refer to [DESIGN_SPACE.md](DESIGN_SPACE.md) for detailed configuration instructions.**

### Droid Dataset
For more details, please refer to the [Droid Dataset](https://droid-dataset.github.io).

**Download**:
```bash
gsutil -m rsync -r gs://gresearch/robotics/droid/1.0.1 droid/1.0.1/ 
```

**Run Training**:
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/droid_train_config.yaml

# Multi-GPU (Set distributed=true in config)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29505 -m scripts.train --config config/droid_train_config.yaml
```

### LIBERO Dataset
For more details, please refer to the [OpenVLA](https://github.com/openvla/openvla), which modifies the original dataset in LIBERO for training VLAs.

**Download**:
```bash
hf download openvla/modified_libero_rlds --repo-type dataset --local-dir LIBERO_modified
```

**Run Training**:
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/libero_train_config.yaml

# Multi-GPU (Set distributed=true in config)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29506 -m scripts.train --config config/libero_train_config.yaml
```

## 📊 Evaluation
We have released VLANeXt checkpoints for the four LIBERO or LIBERO-plus suites on [huggingface](https://huggingface.co/DravenALG/VLANeXt). These checkpoints achieve slightly better performance than the results reported in the paper, as the paper reports the average results.

### LIBERO
For more details, please refer to the [official repository of LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/proj/VLANeXt/third_party/LIBERO

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval
```

### LIBERO-plus
For more details, please refer to the [official repository of LIBERO-plus](https://github.com/sylvestf/LIBERO-plus).

```bash
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/proj/VLANeXt/third_party/LIBERO-plus

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_plus_bench_eval
```

### real-world attempts
We also released a checkpoint trained on the Droid dataset, which can be finetuned for your real-world experiments with a Franka robotic arm.


## ⚡ Analysis
**Model Size and Speed**  
Set `CHECKPOINT_PATH` and `INPUT_MODALITY` in `scripts/size_speed_eval.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.size_speed_eval
```

## ❗ Common Issues
If you run into issues, check [COMMON_ISSUES.md](COMMON_ISSUES.md) for known problems and solutions.

## 📚 Citation

If you find VLANeXt useful for your research or applications, please cite our paper using the following BibTeX:

```bibtex
  @article{wu2026vlanext,
      title={VLANeXt: Recipes for Building Strong VLA Models}, 
      author={Xiao-Ming Wu and Bin Fan and Kang Liao and Jian-Jian Jiang and Runze Yang and Yihang Luo and Zhonghua Wu and Wei-Shi Zheng and Chen Change Loy},
      journal={arXiv preprint arXiv:2602.18532},
      year={2026},
  }
```

## 🗞️ License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).
