#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/1 9:13
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MOPSO.py
# @Statement : Multi-Objective Particle Swarm Optimization
# @Reference : Coello C ,  Pulido G T ,  Lechuga M S . Handling multiple objectives with particle swarm optimization[J]. IEEE Transactions on Evolutionary Computation.
import random
import math
import matplotlib.pyplot as plt


def obj(x):
    # Dimension: 30
    # Range: [0, 1]
    # Optimal solution: x_1 \in [0, 1], x_i = 0, i = 2, ..., n
    # Description: convex, disconnected
    for i in range(len(x)):  # boundary check
        if not 0 <= x[i] <= 1:
            return [1e6, 1e6]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))
    return [f1, f2]


def boundary_check(value, lb, ub):
    """
    The boundary check
    :param value:
    :param lb: lower bound
    :param ub: upper bound
    :return:
    """
    for i in range(len(value)):
        value[i] = max(value[i], lb[i])
        value[i] = min(value[i], ub[i])
    return value


def dominates(s1, s2):
    """
    Determine whether s1 dominates s2
    :param s1: score 1
    :param s2: score 2
    :return:
    """
    sum_less = 0
    for i in range(len(s1)):
        if s1[i] > s2[i]:
            return False
        elif s1[i] < s2[i]:
            sum_less += 1
    if sum_less != 0:
        return True
    return False


def domination_pop(score):
    """
    Determine the dominated and non-dominated particles
    :param score: the score of all particles
    :return: dom[i] = True means particle[i] is dominated, otherwise dom[i] = False
    """
    dom = [False for _ in range(len(score))]
    for i in range(len(score) - 1):
        for j in range(i + 1, len(score)):
            if not dom[j] and dominates(score[i], score[j]):
                dom[j] = True
            elif not dom[i] and dominates(score[j], score[i]):
                dom[i] = True
    return dom


def create_grid(rep_score, ngrid, alpha):
    """
    Create the grid on the objective space for repository
    The total number of girds equals ngrid ^ dim
    :param rep_score: the repository to store the score of all non-dominated particles
    :param ngrid: the grid number on each dimension
    :param alpha: the inflation rate of grid
    :return:
    """
    nobj = len(rep_score[0])
    smin = [1e6] * nobj
    smax = [-1e6] * nobj
    for score in rep_score:
        for i in range(nobj):
            smin[i] = min(smin[i], score[i])
            smax[i] = max(smax[i], score[i])
    for i in range(nobj):  # inflation
        dc = (smax[i] - smin[i]) * alpha
        smin[i] -= dc
        smax[i] += dc
    grid = []
    for i in range(nobj):
        grid.append([])
        dx = (smax[i] - smin[i]) / ngrid
        for j in range(ngrid):
            grid[i].append((j + 1) * dx + smin[i])
    return grid


def grid_index(rep_score, grid):
    """
    Determine the grid index of each element in repository
    :param rep_score: the repository to store the score of all non-dominated particles
    :param grid:
    :return:
    """
    nobj = len(rep_score[0])
    rep_ind = []  # the grid index of each element in the repository
    ngrid = len(grid[0])
    for score in rep_score:
        index = []
        for i in range(nobj):
            for j in range(ngrid):
                if grid[i][j] >= score[i]:
                    index.append(j)
                    break
        rep_ind.append(str(index))
    return rep_ind


def mutate(pos, pm, dim, lb, ub):
    """
    Mutation operation
    :param pos: the position of the particle
    :param pm: the probability of mutation
    :param dim: the dimension
    :param lb: the lower bound
    :param ub: the upper bound
    :return:
    """
    k = random.randint(0, dim - 1)
    dx = pm * (ub[k] - lb[k])
    lb1 = max(lb[k], pos[k] - dx)
    ub1 = min(ub[k], pos[k] + dx)
    new_pos = pos.copy()
    new_pos[k] = random.uniform(lb1, ub1)
    return new_pos


def select_global_best(rep_index, beta):
    """
    Select the global best
    :param rep_index: the grid index of each element in repository
    :param beta: the pressure of global best selection
    :return: the index of the global best
    """
    pro = []
    rep_set = set(rep_index)
    for item in rep_set:
        pro.append(math.exp(-beta * rep_index.count(item)))
    ind = roulette_selection(pro)
    rep_set = list(rep_set)
    rep_ind = rep_set[ind]
    best_index = []
    for i in range(len(rep_index)):
        if rep_index[i] == rep_ind:
            best_index.append(i)
    return random.choice(best_index)


def select_deletion(rep_index, gamma):
    """
    Select one element from the repository to delete
    :param rep_index: probability
    :param gamma: the pressure of deletion selection
    :return: the index of the element to be deleted from the repository
    """
    pro = []
    rep_set = set(rep_index)
    for item in rep_set:
        pro.append(math.exp(gamma * rep_index.count(item)))
    ind = roulette_selection(pro)
    rep_set = list(rep_set)
    rep_ind = rep_set[ind]
    del_index = []
    for i in range(len(rep_index)):
        if rep_index[i] == rep_ind:
            del_index.append(i)
    return random.choice(del_index)


def roulette_selection(pro):
    """
    The roulette selection
    :param pro: probability
    :return:
    """
    r = random.random()
    probability = 0
    sum_pro = sum(pro)
    for i in range(len(pro)):
        probability += pro[i] / sum_pro
        if probability >= r:
            return i


def main(pop, iter, nrep, ngrid, omega, c1, c2, alpha, beta, gamma, mu, lb, ub, vmin, vmax):
    """
    The main function of the MOPSO
    :param pop: the population size
    :param iter: the iteration number
    :param nrep: the repository size
    :param ngrid: the grid number on each dimension
    :param omega: inertia weight
    :param c1: personal learning coefficient
    :param c2: global learning coefficient
    :param alpha: the inflation rate of grid
    :param beta: the pressure of global best selection
    :param gamma: the pressure of deletion selection
    :param mu: mutation rate
    :param lb: the lower bound (list)
    :param ub: the upper bound (list)
    :param vmin: the minimum velocity (list)
    :param vmax: the maximum velocity (list)
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # the dimension of the objective function
    pos = []  # the position of particles
    score = []  # the score of particles
    vel = []  # the velocity of particles
    for _ in range(pop):
        temp_pos = [random.uniform(lb[i], ub[i]) for i in range(dim)]
        pos.append(temp_pos)
        score.append(obj(temp_pos))
        vel.append([0] * dim)
    p_best = score.copy()  # the personal best score of each particle
    p_best_pos = pos.copy()  # the position of personal best
    rep = []  # the repository to store the position of all non-dominated particles
    rep_score = []  # the repository to store the score of all non-dominated particles
    dom = domination_pop(score)
    for i in range(len(dom)):
        if not dom[i]:
            rep.append(pos[i])
            rep_score.append(score[i])
    grid = create_grid(rep_score, ngrid, alpha)  # create the grid
    rep_index = grid_index(rep_score, grid)  # determine the grid index of each element in respository

    # Step 2. The main loop
    for t in range(iter):
        for i in range(pop):
            # Step 2.1. Update the position
            best_ind = select_global_best(rep_index, beta)
            g_best_pos = rep[best_ind]
            vel[i] = [omega * vel[i][j] + c1 * random.random() * (g_best_pos[j] - pos[i][j]) + c2 * random.random() * (
                        p_best_pos[i][j] - pos[i][j]) for j in range(dim)]
            vel[i] = boundary_check(vel[i], vmin, vmax)
            pos[i] = [pos[i][j] + vel[i][j] for j in range(dim)]
            pos[i] = boundary_check(pos[i], lb, ub)
            score[i] = obj(pos[i])

            # Step 2.2. mutation
            pm = (1 - t / (iter - 1)) ** (1 / mu)  # the probability of mutation
            if random.random() < pm:  # perform mutation
                new_pos = mutate(pos[i], pm, dim, lb, ub)
                new_score = obj(new_pos)
                if dominates(new_score, score[i]):  # accept the mutated position
                    pos[i] = new_pos
                    score[i] = new_score
                elif not dominates(score[i], new_score):
                    if random.random() < 0.5:  # accept the mutated position
                        pos[i] = new_pos
                        score[i] = new_score

            # Step 2.3. Update person best
            if dominates(score[i], p_best[i]):
                p_best[i] = score[i].copy()
                p_best_pos[i] = pos[i].copy()
            elif not dominates(p_best[i], score[i]):
                if random.random() < 0.5:
                    p_best[i] = score[i].copy()
                    p_best_pos[i] = pos[i].copy()

        # Step 2.4. Update the repository
        rep.extend(pos)
        rep_score.extend(score)
        dom = domination_pop(rep_score)
        for i in range(len(dom) - 1, -1, -1):
            if dom[i]:
                rep.pop(i)
                rep_score.pop(i)

        # Step 2.5. Update the grid
        grid = create_grid(rep_score, ngrid, alpha)
        rep_index = grid_index(rep_score, grid)

        # Step 2.6. Keep the repository size does not exceed the nrep
        if len(rep) > nrep:
            exceed = len(rep) - nrep
            for _ in range(exceed):
                del_ind = select_deletion(rep_index, gamma)
                rep.pop(del_ind)
                rep_score.pop(del_ind)
                rep_index.pop(del_ind)
        omega *= 0.99

        # Step 2.7. Display information
        plt.figure()
        pos_x = [s[0] for s in score]
        pos_y = [s[1] for s in score]
        rep_x = [s[0] for s in rep_score]
        rep_y = [s[1] for s in rep_score]
        plt.scatter(pos_x, pos_y, marker='x', color='black', label='Particles')
        plt.scatter(rep_x, rep_y, marker='o', color='red', label='Pareto-optimal particles')
        plt.xlabel('f(1)')
        plt.ylabel('f(2)')
        plt.legend(loc='lower left')
        plt.show()
        if (t + 1) % 10 == 0:
            print('Iteration: ' + str(t + 1) + ', Pareto-optimal particles: ' + str(len(rep)))

    # Step 3. Sort the results
    result = {'Pareto-optimal solutions': rep, 'Pareto points': rep_score}
    plt.figure()
    x = [s[0] for s in rep_score]
    y = [s[1] for s in rep_score]
    plt.scatter(x, y)
    plt.show()
    return result


if __name__ == '__main__':
    pop = 200
    iter = 100
    nrep = 100
    ngrid = 7
    omega = 0.5
    c1 = 1
    c2 = 2
    alpha = 0.1
    beta = 2
    gamma = 2
    mu = 0.1
    lb = [0] * 5
    ub = [1] * 5
    vmin = [(lb[i] - ub[i]) * 0.15 for i in range(len(lb))]
    vmax = [(ub[i] - lb[i]) * 0.15 for i in range(len(lb))]
    print(main(pop, iter, nrep, ngrid, omega, c1, c2, alpha, beta, gamma, mu, lb, ub, vmin, vmax))
