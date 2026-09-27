<p align="center">
  <h1 align="center">PMVSplat: Pyramidal Multi-View Stereo-Based Generalizable 3D
Gaussian Splatting</h1>

## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

```bash
cd PMVSplat
conda create -n PMVSplat python=3.10
conda activate PMVSplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

Our PMVSplat uses the same training datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions] on getting datasets.

### DTU (For Testing Only)

* Download the preprocessed DTU data [dtu_training.rar].
* Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`

### Cross-Datasets (For Testing Only)
* Download [NeRF Synthetic], [ScanNet], and [Tanks and Temples] datasets.
## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models], and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=/path/to/checkpoint \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# acid
python -m src.main +experiment=acid \
checkpointing.load=/path/to/checkpoint \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid_nctx3.json \
dataset.view_sampler.num_context_views=3 \
test.compute_scores=true


```

* the rendered novel views will be stored under `outputs/test`

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimatch and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train RCVTSplat
python -m src.main +experiment=re10k data_loader.train.batch_size=1
```

Our models are trained with a single 4090(D) (24GB) GPU. 

### Cross-Dataset Generalization

We use the default model trained on RealEstate10K to conduct cross-dataset evaluations. To evaluate them, *e.g.*, on DTU, run the following command

```bash
# RealEstate10K -> DTU
python -m src.main +experiment=dtu \
checkpointing.load=/path/to/checkpoint \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
test.compute_scores=true

# RealEstate10K -> NeRF Synthetic
python -m src.main +experiment=some \
checkpointing.load=/path/to/checkpoint \
mode=test \
dataset.view_sampler.num_context_views=3 \
test.compute_scores=true
```

## Acknowledgements

The project is largely based on [pixelSplat] and [MVSplat]. Many thanks to these two projects for their excellent contributions!
