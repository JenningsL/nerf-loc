# NeRF-Loc


This project the PyTorch implementation of [NeRF-Loc](), a visual-localization pipeline based on conditional NeRF.
![overview](./imgs/overview.jpg)

## Installation

1. install colmap
2. install python packages

```
pip install -r requirements.txt
```

## How To Run?

### Data Preparation

1. Download data for `Cambridge`, `12scenes`, `7scenes` and `Onepose` following their instructions. 
2. Preprocess datasets: 

```
python3 datasets/video/preprocess_cambridge.py ${CAMBRIDGE_DATA_ROOT}
python3 datasets/video/preprocess_12scenes.py ${12SCENES_DATA_ROOT}
python3 datasets/video/preprocess_7scenes.py ${7SCENES_DATA_ROOT}
```

3. Run image retrieval

```
python3 models/image_retrieval/run.py --config ${CONFIG}
```
replace `{CONFIG}` with `configs/cambridge_all.txt` | `configs/12scenes_all.txt` | `configs/7scenes_all.txt` | etc.

### Training

First, train scene-agnostic NeRF-Loc across different scenes: 

```
python3 pl/train.py --config ${CONFIG} --num_nodes ${HOST_NUM}
```

replace `{CONFIG}` with `configs/cambridge_all.txt` | `configs/12scenes_all.txt` | `configs/7scenes_all.txt` | etc.

Then, finetune on a certain scene to get scene-specific NeRF-Loc model.

```
python3 pl/train.py --config ${CONFIG} --num_nodes ${HOST_NUM}
```

replace `{CONFIG}` with `configs/cambridge/KingsCollege.txt` | `configs/12scenes/apt1_kitchen.txt` | `configs/7scenes/chess.txt` | etc.

---

### Evaluation

To evaluate NeRF-Loc: 

```
python3 pl/test.py --config ${CONFIG} --ckpt ${CKPT}
```

replace `{CONFIG}` with `configs/cambridge/KingsCollege.txt` | `configs/12scenes/apt1_kitchen.txt` | `configs/7scenes/chess.txt` | etc.
replace `{CKPT}` with the path of checkpoint file.


### Pre-trained Models
The 2d backbone weights of COTR can be downloaded [here](https://www.cs.ubc.ca/research/kmyi_data/files/2021/cotr/default.zip), please put it in `models/COTR/default/checkpoint.pth.tar`.
You can download the NeRF-Loc pre-trained models [here](). TODO:

## Acknowledgements
Our codes are largely borrowed from the following works, thanks for their excellent contributions!
+ [Neuray](https://github.com/liuyuan-pal/NeuRay) 
+ [LoFTR](https://github.com/zju3dv/LoFTR) 
+ [COTR](https://github.com/ubc-vision/COTR)
+ [HLoc](https://github.com/cvg/Hierarchical-Localization)
+ [DSM](https://github.com/Tangshitao/Dense-Scene-Matching)
+ [Colmap](https://github.com/colmap/colmap)

## Citation
TODO:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
