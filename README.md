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


<br>

## Data Availability and repo status

We are currently anonymizing the annotators' private information and refactoring the modeling and analysis codes. A more comprehensive update to the repository will be available soon.

<br>


## Questions

For any questio please contact mingbin {At} unimelb dot edu dot au.