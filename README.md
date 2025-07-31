# MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2503.20384)
 

<img src="mole.png"/>


## Installation
The code is built using Python 3.10, and can be run under any environment with Python 3.8 and above. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    conda create --name MoLe_VLA python=3.10

Next, clone our repo and install the required packages:
```
    git clone https://github.com/RoyZry98/MoLe-VLA.git
    cd MoLE_VLA
    conda env create -f environment.yml
```

You may refer to [LIFT3D-CVPR2025](https://github.com/PKU-HMI-Lab/LIFT3D?tab=readme-ov-file) to generate the datasets from RLBench.

## Getting Started
The backbone model CogACT, which includes checkpoints, configs, and model cards, is available on [Hugging Face page](https://huggingface.co/CogACT). Refer to the code below for the minimal inference:

    from PIL import Image
    from vla import load_vla
    import torch

    model = load_vla(
          'CogACT/CogACT-Base',                 # choose from [CogACT-Small, CogACT-Base, CogACT-Large] or the local path
          load_for_training=False, 
          action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
          future_action_window_size=15,
        )                                 
    # about 30G Memory in fp32; 
    
    # (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16
    
    model.to('cuda:0').eval()

    image: Image.Image = <input_your_image>     
    prompt = "move sponge near apple"           # input your prompt
    
    # Predict Action (7-DoF; un-normalize for RT-1 google robot data, i.e., fractal20220817_data)
    actions, _ = model.predict_action(
              image,
              prompt,
              unnorm_key='fractal20220817_data', # input your unnorm_key of the dataset
              cfg_scale = 1.5,                   # cfg from 1.5 to 7 also performs well
              use_ddim = True,                   # use DDIM sampling
              num_ddim_steps = 10,               # number of steps for DDIM sampling
            )

    # results in 7-DoF actions of 16 steps with shape [16, 7]

Alternatively, you can use batch inference function ``predict_action_batch`` from [vla/cogactvla.py](./vla/cogactvla.py) to accelerate inference in the simulator.


## Quickly train model:
```bash
cd /path/to/MoLE_VLA
bash train_multi_task10_mix.sh 14 0.5 0.1 0.5 32 0.999 0,1,2,3,4,5,6,7
```

## Citation
Please cite our work if you find it useful.
```bibtex
@article{zhang2025mole,
  title={MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation},
  author={Zhang, Rongyu and Dong, Menghang and Zhang, Yuan and Heng, Liang and Chi, Xiaowei and Dai, Gaole and Du, Li and Wang, Dan and Du, Yuan and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2503.20384},
  year={2025}
}
```
