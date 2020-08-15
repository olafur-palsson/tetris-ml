#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Agent:

    def make_move(self, state, current_block):
        raise NotImplementedError()

    def give_reward(self, reward):
        raise NotImplementedError()

