# Hangman with RL

This repository contains the code for a quick fun little project I coded trying to solve the game of hangman using Reinforcement Learning

## Dataset
For the dataset I used a file downloaded from [here](https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears-medium.txt) which should contain all words in English (I've not checked)

## State
In order to simplicy the problem, since the hangman is based on such dataset, I just went and check the longest word, and considered the shorter ones with additional padding at the end.

At that point, the number of possible letters is fixed, the length of the words is fixed, so the final state it's just the concatenation of those 2 infos + how many lives are left to the agent.

## Action
For the action, since the set of possible letters is fixed, I just outputed a distribution over list, masked out the ones already tried, and re-normalized

## Algorithm
For the learning I used PPO implemented from scratch using TD(0)

## Result
The learning is successful and rewards can be checked at the end of the `ipynb` notebook

## Possible improvement
Consider the prior of humans in the sampling of the words (estimated from a big corpora of text?)
