# Leveraging Language Models for Robot 3D Scene Understanding

William Chen, Siyi Hu, Rajat Talak, Luca Carlone

# Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Running Code](#running-code)
4. [Real Scene Graph Labelling Visualization](#real-scene-graph-labelling-visualization)
5. [Citation](#citation)

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
We ran our zero-shot room-labelling approach on three real scene graphs created using [Hydra](https://arxiv.org/abs/2201.13360). We provide the visualizations displaying the room bounding box floors, object nodes, and room nodes (which have ground truth and inferred labels attached as well). To open them, either download and open the HTML files found [here](https://github.com/verityw/llm_scene_understanding/tree/main/real_dsg_vis) in a local browser or use the following links. Note that the latter option may take some time to load.
- [College Dorm](https://htmlpreview.github.io/?https://github.com/verityw/llm_scene_understanding/blob/main/real_dsg_vis/sidpac_floor1_3_vis.html)
- [Apartment](https://github.com/verityw/llm_scene_understanding/tree/main/real_dsg_vis)
- [Office](https://htmlpreview.github.io/?https://github.com/verityw/llm_scene_understanding/blob/main/real_dsg_vis/uh2_office_vis.html)

Alternatively, we provide all the rooms' query strings, ground truth labels, and inferred labels in the dropdown below. Note some rooms are omitted due to abnormal ground truth room labels or lack of objects contained within.

<details>
  <summary><b>Real Scene Graph Labelling Results</b></summary>
  <br>
  <pre>
  <code>
######################################################
################## Starting: sidpac ##################
######################################################
--------- 0 ---------
A room containing tables, chairs, and cabinets is called a
predicted: kitchen - ground truth: lounge;seminar room
--------- 4 ---------
A room containing tables and cabinets is called a
predicted: kitchen - ground truth: hallway
--------- 5 ---------
A room containing tables, chairs, and televisions is called a
predicted: lounge - ground truth: lounge;game room
--------- 7 ---------
A room containing chairs is called a
predicted: lounge - ground truth: hallway
--------- 8 ---------
A room containing stairs and railing is called a
predicted: stairwell - ground truth: stairwell
--------- 9 ---------
A room containing stairs and railing is called a
predicted: stairwell - ground truth: stairwell
--------- 10 ---------
A room containing stairs and railing is called a
predicted: stairwell - ground truth: stairwell
--------- 11 ---------
A room containing stairs is called a
predicted: stairwell - ground truth: stairwell
--------- 13 ---------
A room containing tables, chairs, and refrigerators is called a
predicted: kitchen - ground truth: hallway
--------- 14 ---------
A room containing beds, tables, and chairs is called a
predicted: bedroom - ground truth: bedroom
--------- 15 ---------
A room containing stoves, tables, and cabinets is called a
predicted: kitchen - ground truth: kitchen
--------- 17 ---------
A room containing tables and chairs is called a
predicted: lounge - ground truth: lounge;hallway
--------- 18 ---------
A room containing cabinets is called a
predicted: bedroom - ground truth: hallway
--------- 22 ---------
A room containing stairs and railing is called a
predicted: stairwell - ground truth: stairwell
--------- 23 ---------
A room containing stairs is called a
predicted: stairwell - ground truth: stairwell
--------- 24 ---------
A room containing stairs and railing is called a
predicted: stairwell - ground truth: stairwell
#########################################################
################## Starting: apartment ##################
#########################################################
--------- 0 ---------
A room containing stairs, tables, and chairs is called a
predicted: hallway - ground truth: dining room;kitchen
--------- 1 ---------
A room containing beds, chairs, and wardrobes is called a
predicted: bedroom - ground truth: bedroom
--------- 2 ---------
A room containing tables, cabinets, and counters is called a
predicted: kitchen - ground truth: office
--------- 3 ---------
A room containing beds, chairs, and mirrors is called a
predicted: bedroom - ground truth: bedroom
######################################################
################## Starting: office ##################
######################################################
--------- 0 ---------
A room containing tables, chairs, and wardrobes is called a
predicted: bedroom - ground truth: hallway;office
--------- 1 ---------
A room containing computers, tables, and chairs is called a
predicted: lounge - ground truth: office
--------- 2 ---------
A room containing tables, chairs, and wardrobes is called a
predicted: bedroom - ground truth: office
--------- 3 ---------
A room containing tables and chairs is called a
predicted: lounge - ground truth: conference room
  </code>
  </pre>
</details>

## Citation
We have yet to publish this paper. Once it is up on Arxiv, we will update this section accordingly.
