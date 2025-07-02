# WHoW: A Cross-domain Approach for Analysing Conversation Moderation

This repository contains information about the Whow corpus with two subsets from two different moderated conversation scenarios (Debata and Panel). The content include the description of the data, the corpus, the analysis codes, and the modelling code.

The corpus is described in Arxiv paper "WHoW: A Cross-domain Approach for Analysing Conversation
Moderation" Ming-Bin Chen and Jey Han Lau and Lea Frermann from the University of Melbourne.

<img src="./material/demo.png">
Example of a moderated conversation and annotation using the WHoW framework. Blue, green, and red colors represent the supporting team, moderator, and opposing team in one of the Debate subset conversation, respectively. The peach-colored boxes contain the annotations for the corresponding moderator sentences.

<br>

## The Whow Framework

We introduce WHoW: an analytical framework that breaks down the moderation decision-making process into three key components: motives (Why), dialogue acts (How), and target speaker (Who).

<img src="./material/label_def.png">


<br>

## The corpus and annotation

Based on the framework, we annotated moderated multi-party conversations in two domains: TV debates and radio panel discussions. Our dataset comprises a total of 5,657 human-annotated sentences (Test and Dev) and model-annotated 15,494 sentences (GPT-4o) (Train).

<img src="./material/descriptive.png">

Descriptive statistics for the Debate and Panel. M denotes Moderator; share the proportion of words uttered by the moderator; and turn the full utterance (which contains multiple sentences).


## Questions

For any questio please contact mingbin {At} unimelb dot edu dot au.

# Dialogue-Analysis Toolkit

A **modular, command-line driven toolkit** for preparing data, running OpenAI batch jobs, evaluating GPT vs. human annotations, and performing statistical analyses for the *INSQ* and *Roundtable* corpora.

---

## Contents

| Folder / File                  | Purpose                                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------------------|
| `batch_processing.py`          | Build & upload OpenAI **batch jobs** (`create-batches`), monitor them, download results, repair invalid JSON, and more. |
| `evaluation_pipeline.py`       | Episode-level inference + corpus-level **evaluation** of GPT predictions against human labels (classification reports, vote-weighted accuracy). |
| `aggregate_eval.py`            | Post-hoc utilities: joint evaluation of 5-label GPT output, single-task evaluation, and XLSX **comparison** sheets. |
| `data_processing_cli.py`       | All remaining data-munging helpers (train-set build, aggregation, tie-breaking, anonymisation, meta generation) grouped under one CLI. |
| `analysis_cli.py`              | Descriptive **analysis & visualisation**: token stats, motive×DA matrices with t-tests, transition heat-maps. |
| `requirements.txt`             | Python dependencies (see below).                                                               |
| `figure/`, `results/`          | Default output locations for plots & CSV/JSON artefacts.                                       |

---

## Quick start


# 1️⃣ create virtual environment & install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣ set your OpenAI credentials
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org_..."   # optional

# 3️⃣ build and upload batch jobs (dev split)
python batch_processing.py create-batches \
  --corpus insq --mode dev --model gpt-4o \
  --input ./data/insq/dev_data --output ./data/insq/output

# 4️⃣ download & auto-repair once batches complete
python batch_processing.py download-output \
  --record ./log/batch_upload_record.json --repair

# 5️⃣ aggregate GPT + human labels → test.json
python data_processing_cli.py aggregate \
  --corpus roundtable --mode test \
  --gpt-output ./data/roundtable/output/gpt-4o_test.json

# 6️⃣ evaluate GPT vs human
python evaluation_pipeline.py \
  --corpus roundtable --mode test --model gpt-4o

# 7️⃣ token statistics
python analysis_cli.py token-stats --corpus roundtable --splits dev test
