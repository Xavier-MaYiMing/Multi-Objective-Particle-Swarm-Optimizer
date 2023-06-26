#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:45
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MOPSO.py
# @Statement : Multi-Objective Particle Swarm Optimizer (MOPSO)
# @Reference : Coello C,  Pulido G T,  Lechuga M S. Handling multiple objectives with particle swarm optimization[J]. IEEE Transactions on Evolutionary Computation.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def domination_pop(objs):
    # determine the dominated and non-dominated particles
    dom = [False for _ in range(len(objs))]
    for i in range(len(objs) - 1):
        for j in range(i + 1, len(objs)):
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
            elif not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
    return dom


def create_grid(rep_obj, nobj, ngrid, alpha):
    # create the grid on the objective space of repository
    smin = np.min(rep_obj, axis=0)
    smax = np.max(rep_obj, axis=0)
    dc = (smax - smin) * alpha  # inflation
    smin -= dc
    smax += dc
    grid = np.zeros((nobj, ngrid))
    dx = (smax - smin) / ngrid
    for j in range(ngrid):
        grid[:, j] = (j + 1) * dx + smin
    return grid


def grid_index(rep_obj, grid, nobj, ngrid):
    # determine the index of grids
    rep_index = []
    for obj in rep_obj:
        index = []
        for i in range(nobj):
            for j in range(ngrid):
                if grid[i, j] >= obj[i]:
                    index.append(j)
                    break
        rep_index.append(str(index))
    return rep_index


def select_gbest(rep_index, beta):
    # select the global best from the repository
    pro = []
    rep_set = set(rep_index)
    for item in rep_set:
        pro.append(np.exp(-beta * rep_index.count(item)))
    ind = roulette(pro)
    rep_set = list(rep_set)
    rep_ind = rep_set[ind]
    gbest_ind = []
    for i in range(len(rep_index)):
        if rep_index[i] == rep_ind:
            gbest_ind.append(i)
    return np.random.choice(gbest_ind)


def select_deletion(rep_index, gamma):
    # select the deletion from the repository
    pro = []
    rep_set = set(rep_index)
    for item in rep_set:
        pro.append(np.exp(gamma * rep_index.count(item)))
    ind = roulette(pro)
    rep_set = list(rep_set)
    rep_ind = rep_set[ind]
    del_ind = []
    for i in range(len(rep_index)):
        if rep_index[i] == rep_ind:
            del_ind.append(i)
    return np.random.choice(del_ind)


def boundary_check(value, lb, ub):
    # ensure the value within the lb and ub
    new_value = np.where(value < ub, value, ub)
    new_value = np.where(new_value > lb, value, lb)
    return new_value


def mutate(pos, dim, lb, ub, pm):
    # mutation operation
    k = np.random.randint(0, dim)
    dx = pm * (ub[k] - lb[k])
    lb1 = max(lb[k], pos[k] - dx)
    ub1 = min(ub[k], pos[k] + dx)
    new_pos = pos.copy()
    new_pos[k] = np.random.uniform(lb1, ub1)
    return new_pos


def roulette(prob):
    # roulette wheel selection based on probability prob
    prob = prob / sum(prob)
    r = np.random.random()
    cumsum = np.cumsum(prob)
    return min(np.where(r <= cumsum)[0])


def main(npop, iter, lb, ub, nrep, ngrid, alpha, beta, gamma, mu, omega=0.5, c1=1, c2=2):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nrep: repository size
    :param ngrid: grid number on each dimension
    :param alpha: the inflation rate of grid
    :param beta: the pressure of global best selection
    :param gamma: the pressure of deletion selection
    :param mu: mutation rate
    :param omega: inertia weight (default = 0.5)
    :param c1: personal learning coefficient (default = 1)
    :param c2: global learning coefficient (default = 2)
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # dimension
    pos = np.random.uniform(lb, ub, (npop, dim))  # positions
    vel = np.zeros((npop, dim))  # velocity
    objs = [cal_obj(x) for x in pos]  # objectives
    nobj = len(objs[0])  # objective number
    vmin = (lb - ub) * 0.15  # minimum velocity
    vmax = (ub - lb) * 0.15  # maximum velocity
    pbest = objs.copy()  # personal best
    pbest_pos = pos.copy()  # personal best position
    rep = []  # the repository to store the positions of all non-dominated particles
    rep_obj = []  # the repository to store the objectives of all non-dominated particles
    dom = domination_pop(objs)
    for i in range(npop):
        if not dom[i]:
            rep.append(pos[i])
            rep_obj.append(objs[i])
    grid = create_grid(rep_obj, nobj, ngrid, alpha)
    rep_index = grid_index(rep_obj, grid, nobj, ngrid)

    # Step 2. The main loop
    for t in range(iter):
        pm = (1 - t / (iter - 1)) ** (1 / mu)  # mutation probability

        # Step 2.1. Update position
        for i in range(npop):
            gbest_pos = rep[select_gbest(rep_index, beta)]
            vel[i] = omega * vel[i] + c1 * np.random.random() * (gbest_pos - pos[i]) + c2 * np.random.random() * (pbest_pos[i] - pos[i])
            vel[i] = boundary_check(vel[i], vmin, vmax)
            pos[i] += vel[i]
            pos[i] = boundary_check(pos[i], lb, ub)
            objs[i] = cal_obj(pos[i])

            # Step 2.2 Mutation
            if np.random.random() < pm:
                new_pos = mutate(pos[i], dim, lb, ub, pm)
                new_obj = cal_obj(new_pos)
                if dominates(new_obj, objs[i]):
                    pos[i] = new_pos
                    objs[i] = new_obj
                elif not dominates(objs[i], new_obj) and np.random.random() < 0.5:
                    pos[i] = new_pos
                    objs[i] = new_obj

            # Step 2.3. Update personal best
            if dominates(objs[i], pbest[i]):
                pbest[i] = objs[i].copy()
                pbest_pos[i] = pos[i].copy()
            elif not dominates(pbest[i], objs[i]) and np.random.random() < 0.5:
                pbest[i] = objs[i].copy()
                pbest_pos[i] = pos[i].copy()

        # Step 2.4. Update the repository and grid
        rep.extend(pos)
        rep_obj.extend(objs)
        dom = domination_pop(rep_obj)
        for i in range(len(dom) - 1, -1, -1):
            if dom[i]:
                rep.pop(i)
                rep_obj.pop(i)
        grid = create_grid(rep_obj, nobj, ngrid, alpha)
        rep_index = grid_index(rep_obj, grid, nobj, ngrid)
        exceed = max(0, len(rep) - nrep)
        for _ in range(exceed):
            del_ind = select_deletion(rep_index, gamma)
            rep.pop(del_ind)
            rep_obj.pop(del_ind)
            rep_index.pop(del_ind)

        # Step 2.5. Display information
        plt.figure()
        pos_x = [o[0] for o in objs]
        pos_y = [o[1] for o in objs]
        rep_x = [o[0] for o in rep_obj]
        rep_y = [o[1] for o in rep_obj]
        plt.scatter(pos_x, pos_y, marker='x', color='black', label='Particles')
        plt.scatter(rep_x, rep_y, marker='o', color='red', label='Pareto-optimal particles')
        plt.xlabel('objective 1')
        plt.ylabel('objective 2')
        plt.legend(loc='lower left')
        name = 'iteration = ' + str(t + 1) + '.png'
        plt.title('Iteration ' + str(t + 1))
        plt.savefig(name)
        plt.show()
        if (t + 1) % 10 == 0:
            print('Iteration: ' + str(t + 1) + ', Pareto-optimal particles: ' + str(len(rep)))

    # Step 3. Sort the results
    result = {'Pareto-optimal solutions': rep, 'Pareto points': rep_obj}
    plt.figure()
    x = [o[0] for o in rep_obj]
    y = [o[1] for o in rep_obj]
    plt.scatter(x, y)
    plt.show()
    return result


if __name__ == '__main__':
    main(200, 200, np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1]), 100, 7, 0.1, 2, 2, 0.1)
