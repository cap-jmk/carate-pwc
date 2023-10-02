# 1. CARATE

[![Downloads](https://static.pepy.tech/personalized-badge/carate?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/carate)
[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%20-blue)
[![Documentation Status](https://readthedocs.org/projects/carate/badge/?version=latest)](https://carate.readthedocs.io/en/latest/?badge=latest)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![PyPI - Version](https://img.shields.io/pypi/v/carate.svg)](https://pypi.org/project/carate)
![Bert goes into the karate club](bert_goes_into_the_karate_club.png)

<!-- TOC -->

- [1. CARATE](#1-carate)
- [2. Why](#2-why)
- [3. What](#3-what)
- [4. Scope](#4-scope)
- [5. Quickstart](#5-quickstart)
  - [5.1. From CLI](#51-from-cli)
  - [5.2. From notebook/.py file](#52-from-notebookpy-file)
  - [5.3. Analysing runs](#53-analysing-runs)
  - [5.4. Build manually](#54-build-manually)
  - [5.5. Installation from repo](#55-installation-from-repo)
  - [5.6. build the docs](#56-build-the-docs)
  - [5.7. Training results](#57-training-results)
- [6. Reproduce publication](#6-reproduce-publication)
- [7. Build on the project](#7-build-on-the-project)
- [8. Support the development](#8-support-the-development)
- [9. Cite](#9-cite)

<!-- /TOC -->
# 2. Why

Molecular representation is wrecked. Seriously! We chemists talked for decades with an ancient language about something we can't comprehend with that language. We have to stop it, now!

# 3. What

The success of transformer models is evident. Applied to molecules we need a graph-based transformer. Such models can then learn hidden representations of a molecule better suited to describe a molecule.

For a chemist it is quite intuitive but seldomly modelled as such: A molecule exhibits properties through its combined *electronic and structural features*

- Evidence of this perspective  was given in [chembee](https://codeberg.org/sail.black/chembee.git).

- Mathematical equivalence of the variational principle and neural networks was given in the thesis [Markov-chain modelling of dynmaic interation patterns in supramolecular complexes](https://www.researchgate.net/publication/360107521_Markov-chain_modelling_of_dynamic_interaction_patterns_in_supramolecular_complexes).

- The failure of the BOA is described in the case of diatomic tranistion metal fluorides is described in the preprint: [Can Fluorine form triple bonds?](https://chemrxiv.org/engage/chemrxiv/article-details/620f745121686706d17ac316)

- Evidence of quantum-mechanical simulations via molecular dynamics is given in a seminal work [Direct Simulation of Bose-Einstein-Condensates using molecular dynmaics and the Lennard-Jones potential](https://www.researchgate.net/publication/360560870_Direct_simulation_of_Bose-Einstein_condesates_using_molecular_dynamics_and_the_Lennard-Jones_potential)

# 4. Scope

The aim is to implement the algorithm in a reusable way, e.g. for the [chembee](https://codeberg.org/sail.black/chembee.git) pattern. Actually, the chembee pattern is mimicked in this project to provide a stand alone tool. The overall structure of the program is reusable for other deep-learning projects and will be transferred to an own project that should work similar to opinionated frameworks.

# 5. Quickstart 

Quickly have a look over the [documentation](https://carate.readthedocs.io/en/latest/).

First install carate via 
```bash
pip install carate
```
The installation will install torch with CUDA, so the decision of the library what hardware to use goes JIT (just-in-time). At the moment only CPU/GPU is implemented and FPGA/TPU and others are ignored. Further development of the package will then focus on avoiding special library APIs but make the pattern adaptable to an arbitrary algorithmic/numerical backend.

## 5.1. From CLI

For a single file run

```bash
carate -c file_path
```

For a directory of runs you can use 
```bash
carate -d directoy_path
```

## 5.2. From notebook/.py file

You can start runs from [notebooks](./notebooks/). It might be handy for a clean analysis and communication in your team. Check out the [Quickstart notebook](./notebooks/Quickstart.ipynb)

## 5.3. Analysing runs 

I provided some basic functions to analyse runs. With the notebooks you should be able to reproduce
my plots. Check the [Analysis notebook](./notebooks/Analysis.ipynb)

## 5.4. Build manually

The vision is to move away from PyTorch as it frequently creates problems in maintainance. 

The numpy interface of Jax seems to be more promising and robust against problems. By using the numpy
interface the package would become more independent and one might as well implement the algorithm 
in numpy or a similar package. 

To install the package make sure you install all correct verions mentioned in requirements.txt for 
debugging or in pyproject.toml for production use. See below on how to install the package. 

## 5.5. Installation from repo

Inside the directory of your git-clone:

```bash
pip install -e .
```

## 5.6. build the docs

```bash
pip install spawn-lia
lia mkdocs -d carate
```

## 5.7. Training results

Most of the training results are saved in a accumulative json on the disk. The reason is to have enough redundancy in case of data failure.

Previous experiments suggest to harden the machine for training to avoid unwanted side-effects as shutdowns, data loss, or data diffusion. You may still send intermediate results through the network, but store the large chunks on the hardened device.

Therefore, any ETL or data processing might not be affected by any interruption on the training machine.

The models can be used for inference. 

# 6. Reproduce publication

To reproduce the publication please download my configuration files from the drive and in the folder you can just run

```bash
carate -d . 
```

Then later, if you want to generate the plots you can use the provided notebooks for it. Please 
especially refer to the [Analysis notebook](./notebooks/Analysis.ipynb)

# 7. Build on the project

Building on the code is not recommended as the project will be continued in another library (building with that would make most sense).

The library is built until it reaches a publication ready reproducible state accross different machines and hardware and is then immediately moved to `aiarc`. 

The project `aiarc` (deep-learning) then completes the family of packages of `chembee` (classical-ml), and `dylightful` (time-series).

However, you may still use the models as they are by the means of the library production ready.

In case you can't wait for the picky scientist in me, you can still build on my intermediate results. You can find them in the following locations

- [Google Drive](https://drive.google.com/drive/folders/1ikY_EW-Uadkybb--TvxXFgoZtCQtniyH?usp=sharing)

We have to admit it though: There was a security incident on 31st of March 2023, so the results from
Alchemy and ZINC are still waiting. I logged all experiments  


# 8. Support the development

If you are happy about substantial progress in chemistry and life sciences that is not commercial first but citizen first, well then just

<a href="https://www.buymeacoffee.com/capjmk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

Or you can of start join the development of the code. 

# 9. Cite

There is a preprint available on bioRxiv. Read the [preprint](https://www.biorxiv.org/content/10.1101/2022.02.12.470636v1)
