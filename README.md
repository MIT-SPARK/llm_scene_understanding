# Leveraging Language Models for Robot 3D Scene Understanding

William Chen, Siyi Hu, Rajat Talak, Luca Carlone

## Overview
This repo contains code for the paper _Leveraging Language Models for Robot 3D Scene Understanding_. We present several methods for leveraging language models for 3D scene understanding on scene graphs, like those produced by the [Hydra spatial perception system](https://arxiv.org/abs/2201.13360). We test our algorithms on scene graphs generated from the [Matterport3D semantic mesh dataset](https://niessner.github.io/Matterport/).

## Requirements
Before starting, you will need:
- A CUDA-enabled GPU (we used an RTX 3080 with 16 GB of memory)
- A corresponding version of CUDA (we used v11.1)
- Python 3.8 with venv
- Pip package manager

After cloning this repo: 
- Create a virtual environment `python3 -m venv /path/to/llm_su_venv`
- Source the environment `source /path/to/llm_su_venv/bin/activate`
- Enter this repo and install all requirements: `pip install -r requirements.txt`
  - Note that some libraries listed in that file are no longer necessary, as it was procedurally generated. One can alternatively go through the scripts one wishes to run and install their individual dependencies.
  - Such dependencies include: `numpy, scipy, torch, torch_geometric, torchvision, matplotlib, transformers, tqdm, pandas, gensim, sympy`.

## Running Code
- `python zero_shot_<rooms/bldgs>.py` runs our zero-shot language approach on the entire Matterport3D dataset to predict either rooms given objects or buildings given rooms.
- `python <ff/contrastive>_train.py` runs our feed-forward or contrastive training approaches respectively.
  - Run `python data_generator.py` prior to the above to generate the bootstrapped data needed for training and evaluation.
- `python bldg_ff_train.py` and `python bldg_data_generator_comparison.py` are the equivalents for building-prediction. Note that said data generator does _not_ bootstrap datapoints for the test set, instead just using the same test set as `zero_shot_bldgs.py` for easier comparison.
- `python <ff/contrastive>_holdout_tests.py` runs training on a dataset with certain objects withheld, then evaluating on datapoints with those previously-unseen objects.
- `python <ff/contrastive>_label_space_test.py` runs training on the mpcat40 label space dataset, then evaluates on the larger nyuClass label space dataset.
- Some other utility functions and scripts are included as well, such as `compute_cooccurrencies.py`, which generates co-occurrency matrices (i.e. counting frequencies of room-object pairs)

## Real Scene Graph Labelling Visualization
![visualization](https://github.com/verityw/llm_scene_understanding/blob/main/images/RealDSGExample.png)
We ran our zero-shot room-labelling approach on three real scene graphs created using the [Hydra](https://arxiv.org/abs/2201.13360) robot spatial perception system. We provide the visualizations displaying the room bounding box floors, object nodes, and room nodes (which have ground truth and inferred labels attached as well). To open them, either download and open the HTML files found [here](https://github.com/verityw/llm_scene_understanding/tree/main/real_dsg_vis) in a local browser or use the following links. Note that the latter option may take some time to load.
- [College Dorm](https://htmlpreview.github.io/?https://github.com/verityw/llm_scene_understanding/blob/main/real_dsg_vis/sidpac_floor1_3_vis.html)
- [Apartment](https://github.com/verityw/llm_scene_understanding/tree/main/real_dsg_vis)
- [Office](https://htmlpreview.github.io/?https://github.com/verityw/llm_scene_understanding/blob/main/real_dsg_vis/uh2_office_vis.html)

## Citation
We have yet to publish this paper. Once it is up on Arxiv, we will update this section accordingly.
