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
import itertools
from scipy import integrate
import time

class System:
    """ Container class that stores relevant system parameters """
    def __init__(s):
        s.steady_states = bb.get_all_ss()
        _, s.mu, s.M, _ = bb.get_stein_params()
        s.N = len(s.mu)
        s.unstable_fps = {i: s.get_sane_steady_states(num_unstable=i)
                          for i in range(5)}
        s.unstable_fps['stein'] = s.steady_states.values()

    def integrand(s, t, x):
        """ Return N-dimensional gLV equations """
        dxdt = ( np.dot(np.diag(s.mu), x)
                 + np.dot(np.diag(x), np.dot(s.M, x)) )
        for i in range(len(x)):
            if x[i] < 1e-8:
                dxdt[i] = 0
        #print('t', t)
        #print('x', x)
        #print('dxdt', dxdt)
        return dxdt

    def jacobian(s, t, x):
        jac = np.zeros((s.N, s.N))
        for i in range(s.N):
            for j in range(s.N):
                if i is j:
                    val = s.mu[i] + np.dot(s.M, x)[i] + s.M[i,i]*x[i]
                    jac[i, j] = val
                else:
                    val = x[i]*s.M[i,j]
                    jac[i, j] = val
        return jac

    def get_sane_steady_states(s, num_unstable=0):
        """ Return a list of nonnegative fixed points that all have no more
        than num_unstable unstable eigenvectors """
        all_fps = s.get_all_steady_states()
        eps = 1e-8
        nonnegative_steady_states = np.array(
                                    [fp for fp in all_fps if all(fp  >= -eps)])
        sane_steady_states = np.array(
                             [fp for fp in nonnegative_steady_states
                              if s.get_stability(fp) <= num_unstable])
        return sane_steady_states

    def get_stability(s, fp):
        """ Return the number of unstable eigenvectors of the jacobian
        evaluated at fp """
        jac = s.jacobian(0, fp)
        eig_vals, eig_vecs = np.linalg.eig(jac)
        num_unstable_dirs = sum(eig_vals > 0)
        return num_unstable_dirs

    def get_all_steady_states(s):
        """ Return a list of all steady states of the gLV equations
        of the system s """

        combs = []
        for i in range(s.N+1):
            sub_combs = [list(x) for x in itertools.combinations(range(s.N),i)]
            combs.extend(sub_combs)

        fixed_points = []
        for comb in combs:
            # generate subset matrices/vectors of M and mu that correspond to gLV
            # solutions where some populations are 0
            temp_M = s.M[comb, :][:, comb]
            temp_mu = s.mu[comb]
            # solve the the fixed points where some populations are 0
            temp_fp = np.linalg.solve(temp_M, -temp_mu)
            full_fp = np.zeros(s.N)
            for i,elem in enumerate(comb):
                full_fp[elem] = temp_fp[i]
            fixed_points.append(full_fp)

        fixed_points = np.array(fixed_points)
        return fixed_points

def get_steady_state_name(s, y):
    eps = 1e-6
    ss_name = None
    for name in s.steady_states:
        if np.linalg.norm(s.steady_states[name] - y) < eps:
            ss_name = name
    return ss_name

def get_steady_state_number(s, y, num_unstable=2, language=None):
    eps = 1e-4
    ss_name = None
    for i,fp in enumerate(s.unstable_fps[num_unstable]):
        if np.linalg.norm(fp - y) < eps:
            ss_name = i

    if language:
        lang_name = None
        for lang_index in range(len(s.unstable_fps[language])):
            if all(s.unstable_fps[language][lang_index] ==
                    s.unstable_fps[num_unstable][ss_name]):
                lang_name = lang_index

        if lang_name == None and ss_name != None:
            ss_name = 'more unstable SS'
        else:
            ss_name = lang_name

    return ss_name

def solve_til_steady_state(s, ic, t_end=100, verbose=False):
    """ Simulates the gLV system s from initial condition ic until it reaches a
    steady state. """
    solver = integrate.ode(s.integrand, jac=s.jacobian)
    solver.set_integrator('vode', method='bdf', atol=1e-12, rtol=1e-12)
    solver.set_initial_value(ic, 0)

    tvals = [0]
    yvals = [ic]
    deriv = s.integrand(0, ic)

    is_divergent = False
    is_steady_state = None
    divergent_size = 1e5
    while (solver.successful() and is_steady_state == None):
        vals = solver.integrate(t_end, step=True)
        solver.integrate
        tvals.append(solver.t)
        yvals.append(solver.y)
        deriv = s.integrand(0, solver.y)

        if sum(yvals[-1]) > divergent_size:
            is_divergent = True
            break

        if tvals[-1] > 1e8:
            is_divergent = True
            break

        is_steady_state = get_steady_state_number(s, yvals[-1], num_unstable=4)

    tvals = np.array(tvals)
    yvals = np.array(yvals)
    return yvals, tvals, is_divergent

def test_community_transplants(s, size=1, verbose=False, num_unstable=2,
        transplants=None):
    """ Calculate how a given state responds to the introduction of a community
    (stable steady state) of microbial species of size 'size'"""
    if transplants == 'stein':
        #starts = [name for name in s.steady_states]
        print('here')
        print(s.steady_states)
        starts = s.steady_states
        ss_names = [name for name in s.steady_states]
        transplant_list = s.steady_states
        switches = {name: 0 for name in ss_names}
        possible_switches_per = {name: 0 for name in ss_names}
    else:
        starts = s.unstable_fps[num_unstable]
        ss_names = [i for i,x in enumerate(starts)]
        transplant_list = s.unstable_fps[num_unstable]
        switches = {name: 0 for i,name in enumerate(ss_names)}
        possible_switches_per = {name: 0 for i,name in enumerate(ss_names)}

    for i,start in enumerate(starts):
        if transplants == 'stein':
            xa = starts[start]
            start_name = i
        else:
            xa = start
            start_name = i

        for ss_name in ss_names:
            #transplant = s.steady_states[ss_name]
            transplant = starts[ss_name]
            transplant = size * transplant / sum(transplant)
            print('xa',xa)
            ic = xa + transplant
            y, t, is_divergent = solve_til_steady_state(s, ic)
            if is_divergent:
                continue
            end_name = get_steady_state_number(s, y[-1], num_unstable=num_unstable)

            if end_name is not start_name:
                switches[ss_name] += 1
            possible_switches_per[ss_name] += 1

            if verbose:
                print('{}: adding steady state {}: start = {}, end = {}'.
                      format(size, ss_name, start_name, end_name))
    if verbose:
        print(switches)
        print(possible_switches_per)
    return switches, possible_switches_per

def test_single_species_transplants(s, size=1, verbose=False, num_unstable=2):
    """ Calculate how a given state responds to the introduction of a single
    microbial species of size 'size'"""
    #starts = [name for name in s.steady_states]
    starts = s.unstable_fps[num_unstable]
    microbes = list(range(s.N))
    switches = np.zeros(s.N)
    possible_switches_per = np.zeros(s.N)

    for i,start in enumerate(starts):
        xa = start
        start_name = i
        for microbe in microbes:
            transplant = np.zeros(s.N)
            transplant[microbe] = size

            ic = xa + transplant

            y, t, is_divergent = solve_til_steady_state(s, ic)
            if is_divergent:
                continue

            end_name = get_steady_state_number(s, y[-1],
                    num_unstable=num_unstable)
            possible_switches_per[microbe] += 1
            if end_name is not start_name:
                switches[microbe] += 1

            if verbose:
                print('{}: adding microbe {}: start = {}, end = {}'.
                      format(size, microbe, start_name, end_name))
                if end_name == None:
                    print(y[-1])
    if verbose:
        print(switches)
        print(possible_switches_per)
    return switches, possible_switches_per

def test_phage_species_transplants(s, size=1, verbose=False, num_unstable=2):
    """ Calculate how a given state responds to the introduction of a single
    phage that targets a specific species and decrements it by size 'size'"""
    #starts = [name for name in s.steady_states]
    starts = s.unstable_fps[num_unstable]
    microbes = list(range(s.N))
    switches = np.zeros(s.N)
    possible_switches_per = np.zeros(s.N)

    for i,start in enumerate(starts):
        xa = start
        start_name = i
        for microbe in microbes:
            transplant = np.zeros(s.N)
            transplant[microbe] = size

            ic = xa - transplant

            if any(ic < 0):
                continue

            y, t, is_divergent = solve_til_steady_state(s, ic)
            if is_divergent:
                continue

            end_name = get_steady_state_number(s, y[-1],
                    num_unstable=num_unstable)
            possible_switches_per[microbe] += 1
            if end_name is not start_name:
                switches[microbe] += 1

            if verbose:
                print('{}: adding microbe {}: start = {}, end = {}'.
                      format(size, microbe, start_name, end_name))
                if end_name == None:
                    print(y[-1])
    if verbose:
        print(switches)
        print(possible_switches_per)
    return switches, possible_switches_per

def compare_single_species_and_community_transplants(s, verbose=False, num_unstable=1):
    # num_unstable = 0, 1, 2, or 'stein'
    plt.figure()
    #num_unstable = 1 #'stein'
    num_points = 31
    max_val = 15
    xs = np.linspace(0, max_val, num_points)
    single_rates = []
    single_storage = []
    single_totals = []
    community_rates = []
    community_storage = []
    community_totals = []

    read_data = True
    filename = ('transplant_success_rates_{}_{}_unstable_{}'
                 .format(max_val, num_points, num_unstable))
    if not read_data:
        for x in xs:
            print(x)
            single_switches, single_total = test_single_species_transplants(s,
                                            size=x, verbose=verbose,
                                            num_unstable=num_unstable)
            community_switches, community_total = test_community_transplants(s,
                                                  size=x, verbose=verbose,
                                                  num_unstable=num_unstable,
                                                  transplants=num_unstable)
            single_rates.append(sum(single_switches)/sum(single_total))
            single_storage.append(single_switches)
            single_totals.append(single_total)
            community_rates.append(sum(community_switches.values())/sum(community_total.values()))
            community_storage.append(list(community_switches.values()))
            community_totals.append(list(community_total.values()))
        single_storage, community_storage = np.array(single_storage), np.array(community_storage)
        single_totals, community_totals = np.array(single_totals), np.array(community_totals)
        with open('data/{}'.format(filename), 'wb') as f:
            pickle.dump((single_storage, single_rates, single_totals,
                community_storage, community_rates, community_totals), f)
            print('... SAVED transplant success rates to data/{}'
                  .format(filename))
    else:
        with open('data/{}'.format(filename), 'rb') as f:
            (single_storage, single_rates, single_totals, community_storage,
             community_rates, community_totals) = pickle.load(f)
            print('... LOADED transplant success rates from data/{}'
                  .format(filename))


    phage_storage = []; phage_totals = []; phage_rates = []
    read_phage_data = False
    num_points = 81
    max_val = 8
    phage_xs = np.linspace(0, max_val, num_points)
    filename = ('transplant_success_rates_{}_{}_unstable_{}_phage'
                 .format(max_val, num_points, num_unstable))
    if not read_phage_data:
        for x in phage_xs:
            print(x)
            phage_switches, phage_total = test_phage_species_transplants(s,
                                            size=x, verbose=verbose,
                                            num_unstable=num_unstable)
            phage_rates.append(sum(phage_switches)/sum(phage_total))
            print(phage_rates[-1])
            phage_storage.append(phage_switches)
            phage_totals.append(phage_total)
        phage_storage, phage_totals = np.array(phage_storage), np.array(phage_totals)

        with open('data/{}'.format(filename), 'wb') as f:
            pickle.dump((phage_storage, phage_rates, phage_totals), f)
            print('... SAVED transplant success rates to data/{}'
                  .format(filename))
    else:
        with open('data/{}'.format(filename), 'rb') as f:
            phage_storage, phage_rates, phage_totals = pickle.load(f)
            print('... LOADED transplant success rates from data/{}'
                  .format(filename))

    per_single_rate = np.array([[switches/total for switches,total in zip(storage,totals)]
                                 for storage,totals in zip(single_storage, single_totals)])
    per_community_rate = np.array([[switches/total for switches,total in zip(storage,totals)]
                                   for storage,totals in zip(community_storage, community_totals)])
    per_phage_rate = np.array([[switches/total for switches,total in zip(storage,totals)]
                                 for storage,totals in zip(phage_storage, phage_totals)])

    fig, ax = plt.subplots(figsize=(6,6))
    ax1, = ax.plot(xs, single_rates, color='red')
    for i in range(len(per_single_rate[0])):
        ax2, = ax.plot(xs, per_single_rate[:, i], color='red', alpha=.5, lw=1,
                      ls='--')
    ax3, = ax.plot(xs, community_rates, color='blue')
    for i in range(len(per_community_rate[0])):
        ax4, = ax.plot(xs, per_community_rate[:, i], color='blue', alpha=.5,
                      lw=1, ls='--')
    ax5, = ax.plot(phage_xs, phage_rates, color='green')
    for i in range(len(per_phage_rate[0])):
        ax6, = ax.plot(phage_xs, per_phage_rate[:, i], color='green', alpha=.5,
                      lw=1, ls='--')
    ax.set_xlabel('transplant size ($10^{11}$ microbes)')
    ax.set_ylabel('P(transplant success)')
    ax.set_ylim([0, 1]); ax.set_xlim([0, None])
    ax.legend([ax1, ax3, ax5, ax2, ax4, ax6],
              ['single species (mean)',
               'steady state (mean)',
               'phage (mean)',
               'single species (individual) ',
               'steady state (individual)',
               'phage (individual)',
               ], ncol=2, fontsize=8,
               framealpha=1)

    filename = 'figs/transplant_success_rate_S1_origins_{}_unstable_directions.pdf'.format(num_unstable)
    plt.savefig(filename, bbox_inches='tight')
    print('... SAVED figure to {}'.format(filename))


if __name__ == '__main__':
    s = System()
    #print(len(s.unstable_fps[2]))
    #test_single_species_transplants(s, 1, verbose=True)
    #test_community_transplants(s, .01, verbose=True)
    for num_unstable in [0, 'stein', 1, 2]:
        compare_single_species_and_community_transplants(s, verbose=True,
            num_unstable=num_unstable)
    #s.get_sane_steady_states(1)
