## [NeurIPS 2025] MTL-KD: Multi-Task Learning via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver

[//]: # (## Copyright Notice)

[//]: # (Copyright &#40;c&#41; 2025 CIAM Group)

[//]: # ()
[//]: # (**The code can only be used for non-commercial purposes. Please contact the authors if you intend to use this code for commercial purposes.**)


## Overview
This repository contains the code implementation of paper [MTL-KD: Multi-Task Learning via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver](https://openreview.net/forum?id=rlH3e7VlY8).
In this paper, we present a unified multi-task neural solver capable of solving 16 different Vehicle Routing Problem (VRP) variants. The model employs a heavy decoder and light encoder architecture, distilled from multiple single-task student models (each with heavy encoder and light decoder). This approach avoids the need for generating extensive supervised datasets and demonstrates excellent performance in both scale generalization and cross-task generalization.

## Dependencies


```bash
Python=3.9.19
torch==2.0.1
numpy==1.24.4
matplotlib==3.7.5
tqdm==4.65.5
pytz==2022.1
```

## Dataset Download

The test datasets can be downloaded from either of the following two links:
```bash
Google Drive：
https://drive.google.com/drive/folders/1E_pu4a7BWdRPhbGcLJkQ86-74B-X2ibh?usp=sharing

Baidu Netdisk:
https://pan.baidu.com/s/1YI9vI5EA64H3SifY6IX-6g?pwd=jjkf

```
After downloading, place the datasets in the `MTLKD/data` directory.

## Training

To train the model, use the following command:
```bash
cd student_model
python run.py
```

## Testing

### Uniform Distribution Dataset
```bash
cd student_model
python test.py
```
You can modify the `test_problem` list and `test_sizes` list in `main()` to select specific problems and sizes for testing. For other configuration changes, please refer to `test.py`.

### LIB Dataset
```bash
cd student_model
python test_lib.py
```
You can change the `PROBLEM` variable in `test_lib.py` to switch between testing CVRP or VRPTW problems. Remember to also switch the dataset path accordingly (e.g., `../data/CVRP-LIB` or `../data/Vrp-Set-Solomon`).

## Citation
If this repository is helpful for your research, please cite our paper:


*"Yuepeng Zheng, Fu Luo, Zhenkun Wang, Yaoxin Wu, Yu Zhou, MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver, The Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025" <br />*

**BibTeX:**


```
@inproceedings{
Zheng2025mtlkd,
title={{MTL}-{KD}: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver},
author={Yuepeng Zheng and Fu Luo and Zhenkun Wang and Yaoxin Wu and Yu Zhou},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=rlH3e7VlY8}
}
```

## Acknowledgements

The MTL-KD code implementation is based on the following works:
[LEHD](https://github.com/CIAM-Group/NCO_code/blob/main/single_objective/LEHD),[MVMoE](https://github.com/RoyalSkye/Routing-MVMoE),[AMDKD](https://github.com/jieyibi/AMDKD).
We express our sincere gratitude to the authors of these projects for their valuable contributions.
