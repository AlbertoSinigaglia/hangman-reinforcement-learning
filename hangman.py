import random
from functools import cache

import tensorflow as tf
import numpy as np

def num_chars(d):
    set_ = set()
    for s in d:
        for l in s:
            set_.add(l)
    return len(set_), set_

@cache
def get_dataset(path):
    print("loading dataset")
    with open(path) as file:
        dataset = [s.strip() for s in file.readlines()]
    return dataset, *num_chars(dataset)

class Environment:
    def __init__(self, path, max_lives = 10):
        self.dataset, self.letters, self.letters_list = get_dataset(path)
        self.max_len = max([len(s) for s in self.dataset])
        self.letters_list = list(self.letters_list) + ["_"] # used for placeholder for guessed letters
        self.lives = 0
        self.max_lives = max_lives
        self.current_word = self.dataset[random.randrange(0, len(self.dataset))]
        self.current_word_remaining = self.current_word
        self.already_chosen_letters = []

    def act(self, letter:str):
        if letter in self.already_chosen_letters or letter == "_":
            raise Exception("smth wrong")

        self.already_chosen_letters.append(letter)
        if letter not in self.current_word:
            self.lives += 1
            if self.lives == self.max_lives:
                self.reset()
                return -1, 1
            return -1, 0

        self.current_word_remaining = self.current_word_remaining.replace(letter, "_")
        if len(set(list(self.current_word_remaining))) == 1:
            return 1, 1

        return 1, 0

    def reset(self):
        self.lives = 0
        self.current_word = self.dataset[random.randrange(0, len(self.dataset))]
        self.current_word_remaining = self.current_word
        self.already_chosen_letters = []


class Agent:
    def __init__(self, env:Environment):
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.selu),
            tf.keras.layers.Dense(64, activation=tf.nn.selu),
            tf.keras.layers.Dense(len(env.letters_list) + 1, activation="softmax")
        ])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.selu),
            tf.keras.layers.Dense(64, activation=tf.nn.selu),
            tf.keras.layers.Dense(1, activation="linear")
        ])

    def mask(self, env:Environment):
        mask = np.ones(len(env.letters_list) + 1)
        for l in env.already_chosen_letters:
            mask[self.letter_to_int(env, l)] = 1
        return mask

    def state(self, env:Environment):
        w = np.zeros((env.max_len, len(env.letters_list) + 1))
        for i in range(env.max_len):
            if i < len(env.current_word):
                w[i, self.letter_to_int(env, env.current_word[i])] = 1
            else:
                w[i, 0] = 1

        l = np.zeros(len(env.letters_list) + 1)
        for el in env.already_chosen_letters:
            l[self.letter_to_int(env, el)] = 1

        d = np.zeros(env.max_lives)
        d[env.lives] = 1

        return np.concatenate((
            w.reshape((-1)),
            l.reshape((-1)),
            d.reshape((-1))
        ))

    def letter_to_int(self, env: Environment, letter):
        return env.letters_list.index(letter) + 1





def main():
    envs = [Environment("dataset.txt") for _ in range(256)]
    agent = Agent(envs[0])

    for e in range(1000):
        states = []
        masks = []
        for env in envs:
            states.append(agent.state(env))
            masks.append(agent.mask(env))

        states = np.array(states)
        masks = np.array(masks)
        with tf.GradientTape() as tape:
            probs = agent.actor(states)
            print(probs.shape, masks.shape)
            raise Exception




if __name__ == "__main__":
    main()