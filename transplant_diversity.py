#!/usr/bin/env python3
#
# transplant_diversity.py :: Version 0.1
# This file calculates the efficacy of FMT for different transplant
# compositions. Specifically, we study single-species transplants (akin to
# probiotics) versus community transplants (akin to FMT). We hypothesize that
# the community transplants will be more effective at shifting a microbiome
# composition from one steady state to another.
#
# Send questions to Eric Jones at ewj@physics.ucsb.edu
#
###############################################################################

import barebones_CDI as bb
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import integrate

class System:
    """ Container class that stores relevant system parameters """
    def __init__(s):
        s.steady_states = bb.get_all_ss()
        _, s.mu, s.M, _ = bb.get_stein_params()
        s.N = len(s.mu)


    def integrand(s, x, t):
        """ Return N-dimensional gLV equations """
        dxdt = ( np.dot(np.diag(s.mu), x)
                 + np.dot(np.diag(x), np.dot(s.M, x)) )
        for i in range(len(x)):
            if abs(x[i]) < 1e-8:
                dxdt[i] = 0
        return dxdt

def get_steady_state_name(s, y):
    eps = 1e-6
    ss_name = None
    for name in s.steady_states:
        if np.linalg.norm(s.steady_states[name] - y) < eps:
            ss_name = name

    return ss_name

def test_community_transplants(s, size=1, verbose=False):
    """ Calculate how a given state responds to the introduction of a community
    (stable steady state) of microbial species of size 'size'"""
    starts = [name for name in s.steady_states]
    ss_names = [name for name in s.steady_states]
    switches = {name: 0 for name in ss_names}
    possible_switches = 0

    for i,start in enumerate(starts):
        xa = s.steady_states[start]

        for ss_name in ss_names:
            transplant = s.steady_states[ss_name]
            transplant = size * transplant / sum(transplant)

            ic = xa + transplant
            t = np.linspace(0, 5000, 1001)
            y = integrate.odeint(s.integrand, ic, t)
            end = get_steady_state_name(s, y[-1])

            if end is not start: switches[ss_name] += 1
            possible_switches += 1

            if verbose:
                print('adding steady state {}: start = {}, end = {}'.
                      format(ss_name, start, end))
    if verbose:
        print(switches)
    return switches, possible_switches

def test_single_species_transplants(s, size=1, verbose=False):
    """ Calculate how a given state responds to the introduction of a single
    microbial species of size 'size'"""
    starts = [name for name in s.steady_states]
    microbes = list(range(s.N))
    switches = np.zeros(s.N)
    possible_switches = 0

    for i,start in enumerate(starts):
        xa = s.steady_states[start]

        for microbe in microbes:
            transplant = np.zeros(s.N)
            transplant[microbe] = size

            ic = xa + transplant
            t = np.linspace(0, 5000, 1001)
            y = integrate.odeint(s.integrand, ic, t)
            end = get_steady_state_name(s, y[-1])
            if end is not start: switches[microbe] += 1
            possible_switches += 1

            if verbose:
                print('adding microbe {}: start = {}, end = {}'.
                      format(microbe, start, end))
    if verbose:
        print(switches)
    return switches, possible_switches

def compare_single_species_and_community_transplants(s, verbose=False):
    num_points = 101
    max_val = 50
    xs = np.linspace(0, max_val, num_points)
    single_rates = []
    single_storage = []
    community_rates = []
    community_storage = []

    read_data = True 
    filename = 'transplant_success_rates_{}_{}'.format(max_val, num_points)
    if not read_data:
        for x in xs:
            print(x)
            single_switches, single_total = test_single_species_transplants(s,
                                            size=x, verbose=verbose)
            community_switches, community_total = test_community_transplants(s,
                                                  size=x, verbose=verbose)
            single_rates.append(sum(single_switches)/single_total)
            community_rates.append(sum(community_switches.values())/community_total)
            single_storage.append(single_switches)
            community_storage.append(list(community_switches.values()))
        single_storage, community_storage = np.array(single_storage), np.array(community_storage)
        with open('data/{}'.format(filename), 'wb') as f:
            pickle.dump((single_storage, single_rates, community_storage,
                         community_rates), f)
            print('... SAVED transplant success rates to data/{}'
                  .format(filename))
    else:
        with open('data/{}'.format(filename), 'rb') as f:
            (single_storage, single_rates, community_storage,
             community_rates) = pickle.load(f)
            print('... LOADED transplant success rates from data/{}'
                  .format(filename))
    per_single_rate = single_storage / len(single_storage[0])
    per_community_rate = community_storage / len(community_storage[0])

    fig, ax = plt.subplots(figsize=(6,6))
    ax1, = ax.plot(xs, single_rates, color='red')
    for i in range(len(per_single_rate[0])):
        ax2, = ax.plot(xs, per_single_rate[:, i], color='red', alpha=.5, lw=1,
                      ls='--')
    ax3, = ax.plot(xs, community_rates, color='blue')
    for i in range(len(per_community_rate[0])):
        ax4, = ax.plot(xs, per_community_rate[:, i], color='blue', alpha=.5,
                      lw=1, ls='--')
    ax.set_xlabel('transplant size ($10^{11}$ microbes)')
    ax.set_ylabel('P(transplant success)')
    ax.set_ylim([0, 1]); ax.set_xlim([0, None])
    ax.legend([ax1, ax2, ax3, ax4],
              ['single species (mean)',
               'single species (individual) ',
               'steady state (mean)',
               'steady state (individual)'], ncol=1, fontsize=12,
              loc='upper right', framealpha=1)

    filename = 'figs/transplant_success_rate_v0.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print('... SAVED figure to {}'.format(filename))










s = System()
compare_single_species_and_community_transplants(s, verbose=False)
