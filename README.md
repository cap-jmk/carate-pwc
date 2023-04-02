# CARATE
[![Downloads](https://static.pepy.tech/personalized-badge/carate?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/carate)
[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%20-blue) 
![Style Black](https://warehouse-camo.ingress.cmh1.psfhosted.org/fbfdc7754183ecf079bc71ddeabaf88f6cbc5c00/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64652532307374796c652d626c61636b2d3030303030302e737667) 

![Bert goes into the karate club](bert_goes_into_the_karate_club.png)


# Why 

Molecular representation is wrecked. Seriously! We chemists talked for decades with an ancient language about something we can't comprehend with that language. We have to stop it, now!

# What 

The success of transformer models is evident. Applied to molecules we need a graph-based transformer. Such models can then learn hidden representations of a molecule better suited to describe a molecule. 

For a chemist it is quite intuitive but seldomly modelled as such: A molecule exhibits properties through its combined *electronic and structural features*

* Evidence of this perspective  was given in [chembee](https://codeberg.org/sail.black/chembee.git). 

* Mathematical equivalence of the variational principle and neural networks was given in the thesis [Markov-chain modelling of dynmaic interation patterns in supramolecular complexes](https://www.researchgate.net/publication/360107521_Markov-chain_modelling_of_dynamic_interaction_patterns_in_supramolecular_complexes). 

* The failure of the BOA is described in the case of diatomic tranistion metal fluorides is described in the preprint: [Can Fluorine form triple bonds?](https://chemrxiv.org/engage/chemrxiv/article-details/620f745121686706d17ac316)

* Evidence of quantum-mechanical simulations via molecular dynamics is given in a seminal work [Direct Simulation of Bose-Einstein-Condensates using molecular dynmaics and the Lennard-Jones potential](https://www.researchgate.net/publication/360560870_Direct_simulation_of_Bose-Einstein_condesates_using_molecular_dynamics_and_the_Lennard-Jones_potential)
# Scope

The aim is to implement the algorithm in a reusable way, e.g. for the [chembee](https://codeberg.org/sail.black/chembee.git) pattern. Actually, the chembee pattern is mimicked in this project to provide a stand alone tool. The overall structure of the program is reusable for other deep-learning projects and will be transferred to an own project that should work similar to opinionated frameworks. 



# Installation on CPU 

Prepare system 
```bash
sudo apt-get install python3-dev libffi-dev
```

## Build manually

```bash 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
pip install torch-scatter torch-sparse torch-geometric rdkit-pypi networkx[default] matplotlib
pip install torch-cluster 
pip install torch-spline-conv 
``` 

# Faster way 

Inside the directory of your git-clone:

```bash
pip install -e .
```

# Usage 

The program is meant to be run as a simple CLI. You can specify the configuration either via a `JSON` and use the program as a microservice, or you may run it locally from the command line. It is up to you. 

Finally, with the new `pyproject.toml` it is possible to 
```bash 
pip install carate
```
The installation will install torch with CUDA, so the decision of the library what hardware to use goes JIT (just-in-time). At the moment only CPU/GPU is implemented and FPGA/TPU and others are ignored. Further development of the package will then focus on avoiding special library APIs but make the pattern adaptable to an arbitrary algorithmic/numerical backend.

```bash
carate -c path_to_config_file.py
```

## Start a run

To start a run you need to define the configuration. You can do so by defining a `.json` or a `config.py` file

Examples for `config.py` files are given in `config_files`

Or you can check the the `tutorial.ipynb` in `notebooks` how to use the package with a `.json` file 

## Training results 

Most of the training results are saved in a accumulative json on the disk. The reason is to have enough redundancy in case of data failure. 

Previous experiments suggest to harden the machine for training to avoid unwanted side-effects as shutdowns, data loss, or data diffusion. You may still send intermediate results through the network, but store the large chunks on the hardened device.

Therefore, any ETL or data processing might not be affected by any interruption on the training machine.

# Build on the project

Building on the code is not recommended as the project will be continued in another library (building with that would make most sense). 

However, you may still use the models as they are by the means of the library production ready.

In case you can't wait for the picky scientist in me, you can still build on my intermediate results. You can find them in the following locations 

* [Google Drive](https://drive.google.com/drive/folders/1ikY_EW-Uadkybb--TvxXFgoZtCQtniyH?usp=sharing)

We have to admit it though: There was a security incident on 31st of March 2023, so the results from 
Alchemy and ZINC are still waiting. I logged all experiments  I did and uploaded the log, such 
that any picky reviewer can check where there happened weird stuff, I made a mistake, or there 
was an incident and requests more reproductions. That should solve the issue for now!

# Support the development

If you are happy about substantial progress in chemistry and life sciences that is not commercial first but citizen first, well then just

<a href="https://www.buymeacoffee.com/capjmk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>



# Cite 

There is a preprint available on bioRxiv. Read the [preprint](https://www.biorxiv.org/content/10.1101/2022.02.12.470636v1)