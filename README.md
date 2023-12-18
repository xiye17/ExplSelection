# ExplSelection
Code for [Explanation Selection Using Unlabeled Data for Chain-of-Thought Prompting
](https://arxiv.org/abs/2302.04813) (EMNLP 2023).


## Setup
* python==3.8
* requirements: `pip install -r requirements.txt`
* Set OPENAI KEY: `export KEY=yourkey`
* Run `mkdir misc expl_search/misc expl_search/annotations expl_search/candidates `

## TLDR
We directly provide pairs of seed few-shot CoT prompts and searched CoT prompts in prompt pairs. (Note these prompts are optimized for `code-davinci-002`).

Run evaluation to compare seed CoTs and optimized CoTs on `code-002` \
`SETS="0" sh exp_scripts/exp_result.sh gsm # datasets include gsm, ecqa, esnli, strategyqa`

## Run Optimization Experiments
We take GSM as an example dataset

```
# SEED
sh exp_scripts/main/gsm.sh SEED

# OSACC
sh exp_scripts/main/gsm.sh OSACC

# OSLL
sh exp_scripts/main/gsm.sh OSLL
```

## Citation

```
@InProceedings{Ye-Durrett:2023:explselect,
  title = {Explanation Selection Using Unlabeled Data for In-Context Learning},
  author = {Xi Ye and Greg Durrett},
  booktitle = {Proceedings of EMNLP},
  year = {2023},
}
```