# MLOW: Interpretable Low-Rank Frequency Magnitude Decomposition of Multiple Effects for Time Series Forecasting


This is a PyTorch implementation of the paper: [] MLOW: Interpretable Low-Rank Frequency Magnitude Decomposition of Multiple Effects for Time Series Forecasting.  MLOW is an interpretable Fourier-based decomposition method that disentangles multiple effects of certain time series data using learned low-rank components, providing an interpretable decomposition in the temporal domain rather than operating in the complex frequency domain.

If you find this project helpful, please don't forget to give it a ⭐ Star. Thank you very much! 

We'll keep updating this repository with new features and news. Please feel free to contact us if you have any questions or find any bugs. 

## 🧠 Inference

### Overview Pipeline
A mathematical mechanism enables flexible selection of initial frequency levels and the input horizon. The inference pipeline allows users to choose the initial frequency level, input horizon, and low-rank components, which are represented as "frequency_level", "seq_len", and "rank" in the run.py file. The visualization of the pipeline is provided as follows:
[![paper preview](./fig/Decomposition.png)](./fig/Decomposition.pdf)

### Low Rank Algorithm
A low-rank learning method for Frequency Magnitude, Hyperplane-NMF, is proposed. A regularization parameter is used, which is represented as "lamb" in the run.py file. For implementation details, please refer to the code in ./data_provider/data_loader.py. The training and inference pseudocode are as follows:

[![paper preview](./fig/Decomposition.png)]

## 🚀 Training
### Environment
You can install the enviornment by runing the following code
```
pip install -r requirements.txt
```
## Dataset
You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry). Then place the downloaded data under the folder `./dataset`. 

## Reproduce the Results 
```
sh ./scripts/long_term_forecast/ETTh1.sh
sh ./scripts/short_term_forecast/PEMS.sh
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
