# CARATE
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%20-blue) 
![Style Black](https://warehouse-camo.ingress.cmh1.psfhosted.org/fbfdc7754183ecf079bc71ddeabaf88f6cbc5c00/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64652532307374796c652d626c61636b2d3030303030302e737667) 

# Why 

Molecular representation is wrecked. Seriously! We chemists talk with an ancient language about something we can't comprehend with that language for decades. It has to stop. 

# What 

The success of transformer models is evident. Applied to molecules we need a graph-based transformer. Such models can then learn hidden representations of a molecule better suited to describe a molecule. 

For a chemist it is quite intuitive but seldomly modelled as such: A molecule exhibits properties through its combined *electronic and structural features*

* Evidence of this perspective  was given in [chembee](https://codeberg.org/sail.black/chembee.git). 

* Mathematical equivalence of the variational principle and neural networks was given in the thesis [Markov-chain modelling of dynmaic interation patterns in supramolecular complexes](https://www.researchgate.net/publication/360107521_Markov-chain_modelling_of_dynamic_interaction_patterns_in_supramolecular_complexes). 

* The failure of the BOA is described in the case of diatomic tranistion metal fluorides is described in a [Can Fluorine form triple bonds?](https://chemrxiv.org/engage/chemrxiv/article-details/620f745121686706d17ac316)

* Evidence of quantum-mechanical simulations via molecular dynamics is given in a seminal work [Direct Simulation of Bose-Einstein-Condensates using molecular dynmaics and the Lennard-Jones potential](https://www.researchgate.net/publication/360560870_Direct_simulation_of_Bose-Einstein_condesates_using_molecular_dynamics_and_the_Lennard-Jones_potential)
# Scope

The aim is to implement the algorithm in a reusable way, e.g. for the [chembee](https://codeberg.org/sail.black/chembee.git) pattern. Actually, the chembee pattern is mimicked in this project to provide a stand alone tool. The overall structure of the program is reusable for other deep-learning projects and will be transferred to an own project that should work similar to opinionated frameworks. 

# Installation on CPU 

Prepare system 
```bash
sudo apt-get install python3-dev libffi-dev
```

```bash 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
pip install torch-scatter torch-sparse torch-geometric rdkit-pypi networkx[default] matplotlib
pip install torch-cluster 
pip install torch-spline-conv 
``` 

# Usage 

```bash 
bash install.sh
```

```bash
carate -c path_to_config_file.py
```
Example when CLI is not working
```bash
 python mcf.py
```

## Training results 

Most of the training results are saved in pairs. The reason for this data structure is simply that the training can be interrupted for any reason. However the current result may still be saved or sent across a 
given network. 

Therefore any ETL or data processing might not be affected by any interruption on the training machine.

# Outlook 

The program is meant to be run as a simple CLI. Not quite there yet. 

# Cite 

There is a preprint available on bioRxiv. Read the [preprint](https://www.biorxiv.org/content/10.1101/2022.02.12.470636v1)