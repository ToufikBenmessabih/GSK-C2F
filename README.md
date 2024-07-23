# GSK-C2F: Graph Skeleton Modelization for Action Segmentation and Recognition using a Coarse-to-Fine Strategy


### Data download and directory structure:
Evaluation was conducted using the benchmark [InHARD](https://paperswithcode.com/dataset/inhard) which offers RGB and skeleton data focused on industrial human action recognition. With 4804 action samples across 14 industrial classes from 16 subjects. It proposes 3 different view angles ("up", "left" and "right"). It uses the Perception Neuron 32 Edition v2 sensor to capture 3D joint positions and rotations at 120 Hz, stored in BVH format. Its main purpose is the analysis and development of learning techniques in industrial environments involving human-robot collaborations.

The data directory is arranged in following structure

- data
   - mapping.csv
   - dataset_name
     - features 
     - groundTruth
     - splits
     - results
        - supervised_C2FTCN
            - split1
              - check_pointfile

## Evaluation Metrics

For evaluation, Mean-over-frames (MoF), segment-wise edit distance (Edit), and F1-scores with IoU thresholds of 0.10, 0.25, and 0.50 (F1@10, 25, 50) are reported. The Edit metric measures the
correctness of predicted temporal ordering of actions, used to penalize over-segmentation, while the F1 score assesses the quality of sequence segmentation coverage across various thresholds.

## Getting Started

### Environment
- Create a virtual environment: `python -m venv tags_env`
- Install Pytorch >=1.9.0 : `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

- show torch version: `pip show torch`
- Show CUDA compiler version: `nvcc --version`
- Show GPU informaation: `nvidia-smi`
- For more information read this: [How to check if your GPU card supports a particular CUDA version.ipynb](https://github.com/ToufikBenmessabih/GSK-ED/blob/1b95908f603063f7370d63bf74b6eb5feebe8f44/How%20to%20check%20if%20your%20GPU%20card%20supports%20a%20particular%20CUDA%20version.ipynb)
  
## Load Dataset

Load 3D joint positions from the InHARD-13 dataset: [skeleton_pose_inhard13.ipynb](https://github.com/ToufikBenmessabih/GSK-ED/blob/85dfadcdbcbd5bce91bd8c76443894cf3cca76d5/skeleton_pose_inhard13.ipynb)

## Train
`python train.py --dataset_name <dataset> --cudad 0 --base_dir <dataset_path> --split <nbr>`

Example: `python train.py --dataset_name InHARD-13 --cudad 0 --base_dir ./data/InHARD-13/ --split 1`

## Evaluation
`python eval.py --dataset_name <dataset>  --base_dir <dataset_path>`

Example: `python eval.py --dataset_name InHARD-13 --cudad 0 --base_dir ./data/InHARD-13/ --compile_result`

[FlowChart](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1#G1ZXNX5TX5S1Y__h-Z_5sWsbrjy9MwEDAM)


[Test table](https://cesifr.sharepoint.com/:x:/s/HumanActionRecognition/ERxhL1xm9yNGquZTXgwo0VcBwAlLfIaYgj7Fxr5bgHjdNw?e=ZilQLi)
