# 🛒 WebShop

[![Python version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-3813/)
[![License](https://img.shields.io/badge/License-Princeton-orange)](https://copyright.princeton.edu/policy)
[![PyPI version](https://badge.fury.io/py/webshop.svg)](https://badge.fury.io/py/webshop)
![Pytest workflow](https://github.com/princeton-nlp/webshop/actions/workflows/pytest.yml/badge.svg)

Implementation of the WebShop environment and search agents for the paper:

**[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://webshop-pnlp.github.io/)**  
[Shunyu Yao*](https://ysymyth.github.io/), [Howard Chen*](https://howard50b.github.io/), [John Yang](https://john-b-yang.github.io/), [Karthik Narasimhan](https://www.cs.princeton.edu/~karthikn/)

<p float="left">
  <img src="assets/demo.gif">
</p>

This repository contains code for reproducing results. If you find this work useful in your research, please cite:

```
@inproceedings{yao2022webshop,
  bibtex_show = {true},
  title = {WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents},
  author = {Yao, Shunyu and Chen, Howard and Yang, John and Narasimhan, Karthik},
  booktitle = {ArXiv},
  year = {preprint},
  html = {https://arxiv.org/abs/2207.01206},
  tag = {NLP}
}
```
## 📖 Table of Contents <!-- omit in toc -->
* [👋 Overview](#-overview)
* [🚀 Setup](#-setup)
* [🛠️ Usage](#-usage)
* [💫 Contributions](#-contributions)
* [🪪 License](#-license)
## 👋 Overview
WebShop is a simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. In this environment, an agent needs to navigate multiple types of webpages and issue diverse actions to find, customize, and purchase a product given an instruction. WebShop provides several challenges including understanding compositional instructions, query (re-)formulation, dealing with noisy text in webpages, and performing strategic exploration.

**Hugging Face Demo**: Devise your own natural language query for a product and ask for an agent trained with WebShop to find it on Amazon or eBay, deployed as a 🤗 Hugging Face space [here](https://huggingface.co/spaces/webshop/amazon_shop)!

**Python Package**: If you would like to interact with the `WebShop` environment interface instead of installing and setting up the source code, you can install the `webshop` python package directly. The [project page](https://pypi.org/project/webshop/) contains (WIP) documentation on how to use the library, and the library can be installed via `pip install webshop`.
## 🚀 Setup
Our code is implemented in Python. To setup, do the following:
1. Install [Python 3.8.13](https://www.python.org/downloads/release/python-3813/)
2. Install [Java](https://www.java.com/en/download/)
3. Download the source code:
```sh
> git clone https://github.com/princeton-nlp/webshop.git webshop
```
4. Create a virtual environment using [Anaconda](https://anaconda.org/anaconda/python) and activate it
```sh
> conda create -n webshop python=3.8.13
> conda activate webshop
```
4. Install requirements into the `webshop` virtual environment via the `setup.sh` script
```sh
> ./setup.sh [-d small|all]
```
The setup script performs several actions in the following order:
* Installs Python dependencies listed in `requirements.txt`
* Downloads product and instruction data for populating WebShop
* Downloads `spaCy en_core_web_lg` model
* Construct search engine index from product, instruction data
* Downloads 50 randomly chosen trajectories generated by MTurk workers
The `-d` flag argument allows you to specify whether you would like to pull the entire product + instruction data set (`-d all`) or a subset of 1000 random products (`-d small`).

## 🛠️ Usage
The WebShop environment can be rendered in two modes - `html` and `simple` - each of which offer a different observation space. The `simple` mode strips away the extraneous meta-data that the `html` mode includes to make model training and evaluation easier.
### Webpage Environment (`html` mode)
Launch the `WebShop` webpage:
```sh
> ./run_dev.sh
```
The site should then be viewable in the browser. Go to http://localhost:3000/ABC, where you should land on the search home page with a random instruction.

Navigating the website will automatically generate a corresponding trajectory file in the `user_session_logs/mturk` folder. Each file corresponds to a single instruction/web session, and each step of the file corresponds to a single action (i.e. `search[...]`, `click[...]`).

The current WebShop build comes with two flags:
* `--log`: Include this flag to create a trajectory `.jsonl` log file of actions on WebShop
* `--attrs`: Include this flag to display an `Attributes` tab on the `item_page` of WebShop

### Text Environment (`simple` mode)
The `simple` mode of the WebShop environment is packaged and readily available as an OpenAI environment. The OpenAI gym definitions of the text environment can be found in the `web_agent_site/envs` folder.

To start using the gym and building agents that interact with the WebShop environment, include the following statements in your Python file:
```python
import gym
from web_agent_site.envs import WebAgentTextEnv

env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=...)
```
Now, you can write your own agent that interacts with the environment via the standard OpenAI gym [interface](https://www.gymlibrary.ml/content/api/).

Examples of a `RandomPolicy` agent interacting with the WebShop environment in both `html` and `simple` mode can be found in the `run_envs` folder. To run these examples locally, run the `run_web_agent_text_env.sh` or `run_web_agent_site_env.sh` script:
```sh
> ./run_web_agent_text_env.sh
Products loaded.
Keys Cleaned.
Attributes Loaded.
100%|██████████████████| 1000/1000
Loaded 6910 goals.
Amazon Shopping Game [SEP] Instruction: [SEP] Find me slim f...
Available actions: {'has_search_bar': True, 'clickables': ['search']}
Taking action "search[shoes]" -> Reward = 0.0
...
```
In order to run the `run_web_agent_site_env.sh` script, you must download a version of [ChromeDriver](https://chromedriver.chromium.org/downloads) compatible with your Chrome browser version. Once you have downloaded and unzipped the executable, rename it `chromedriver` and place it in the `webshop/envs` folder.

### Baseline Models
To run baseline models (rule, IL, RL, IL+RL) from the paper, please refer to the `README.md` in the [baseline_models](https://github.com/princeton-nlp/webshop/tree/master/baseline_models) folder.

### Sim-to-real Transfer
To read more about how the sim-to-real transfer of agents trained on WebShop to other environments works, please refer to the `README.md` in the [transfer](https://github.com/princeton-nlp/webshop/tree/master/transfer) folder.

## 💫 Contributions
We would love to hear from the broader NLP and Machine Learning community, and we welcome any contributions, pull requests, or issues! To do so, please either file a new pull request or issue and fill in the corresponding templates accordingly. We'll be sure to follow up shortly!

## 🪪 License
Check `LICENSE.md`
