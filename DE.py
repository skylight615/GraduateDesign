import math
import random
import time
import cost_function as cf
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import matplotlib.pyplot as plt
import logging
import multiprocessing as mp

from DataLoader import DataLoader, DataParser

min_vec = list()  # record the sub vector that has min cost value now
min_value = 99999  # record the current min value during evolution
cost_list = list()
cost_list_w = list()
CRm = 0.5
SaDE_p = 0.5
age = 0
sub_num = 0
ns1, ns2, nf1, nf2 = 0, 0, 0, 0
f_rec = list()
CR_list = list()
CR_rec = list()


# init the population, size = NP, and the member is a vector (1,num_parameter).
# The value in vector should in the range of IPR.
def init_population(init_code: list):
    global cost_list
    population = list()
    for n in range(NP):
        item = list()
        for code in init_code:
            group = dataset.code2group[code]
            sets = dataset.groups[group]
            rd_num = random.randint(0, len(sets) - 1)
            item.append(dataset.str2code[sets[rd_num]])
        population.append(np.array(item))
    cost_list = cf.cost(population, dataset)
    return population


def generate_f():
    global SaDE_p
    fp = np.random.uniform(0, 1, 1)
    if fp > SaDE_p:
        F = np.random.standard_cauchy(1)
    else:
        F = np.random.normal(0.5, 0.3, 1)
    return F


def update_SaDE_p():
    global SaDE_p, ns1, ns2, nf1, nf2, age
    if (ns2*(ns1+nf1)+ns1*(ns2+nf2)) == 0:
        SaDE_p = 0.5
    else:
        SaDE_p = ns1*(ns2+nf2)/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
    logger.info(f"echo:{age} ns1: {ns1} ns2: {ns2} nf1: {nf1} nf2: {nf2} SaDE_p: {SaDE_p}")
    ns1, ns2, nf1, nf2 = 0, 0, 0, 0


def update_CRm():
    global CRm, CR_rec, f_rec
    if len(CR_rec) == 0:
        CRm = 0.5
    else:
        w_sum = sum(f_rec)
        w = [i / w_sum for i in f_rec]
        CRm = sum([w[i] * CR_rec[i] for i in range(len(w))])
        if CRm < 0:
            CRm = 0
        CR_rec.clear()
        f_rec.clear()
    logger.info(f"CRm: {CRm}")


# generate the v_i using x_i. The difference is calculated by random, x_i also a random member
# population is the sub groups, and generation is the origin sequence
def evolution(population: list, index: list, generation: list):
    # do permutation for population[index]
    F = generate_f()
    next_generation, function_index = evolve(population, F, index)
    next_generation = select(population, next_generation, function_index, index, generation)
    process_recorder.append(min_value)
    return next_generation


def evolve(population: list, F: float, groups_index):
    next_generation = list()
    function_index = list()
    for index in range(NP):
        function_index.append(1)
        diff = np.zeros(shape=len(population[0]), dtype=float)
        rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2), replace=False)
        for pair in rand_i:
            diff = diff + population[pair[0]] - population[pair[1]]
        # SaDE
        p = np.random.uniform(0, 1, 1)
        if p > SaDE_p:
            diff += min_vec[groups_index] - population[index]
            function_index[index] = 3
        new_seq = np.floor(population[index] + F * diff)
        for n in range(len(new_seq)):
            tem = population[index][n]
            # (id - base) % size + base
            base = dataset.base[tem]
            new_seq[n] = (new_seq[n] - base[0]) % base[1] + base[0]
        next_generation.append(crossover(population[index], new_seq, index))
    return next_generation, function_index


# generate the u_i using v_i and x_i
def crossover(x, v, index):
    global CRm, CR_list
    u_j = np.zeros(len(x))
    r = range(len(x))
    if age % 5 == 0:
        CR = np.random.normal(CRm, 0.1, 1)
        CR_list[index] = CR
    else:
        CR = CR_list[index]
    for j in r:
        rand = np.random.uniform(0, 1, 1)
        save_index = np.random.choice(r, size=1)
        if rand <= CR or j == save_index:
            u_j[j] = v[j]
        else:
            u_j[j] = x[j]
    return u_j


# select between u_i and x_i into next generation, and criterion is cost function
def select(population: list, next_generation: list, function_index: list, index: list, generation: list):
    global min_value, min_vec, cost_list, ns1, ns2, nf1, nf2, f_rec, CR_rec
    new_generation = generation.copy()
    for i in range(NP):
        new_generation[i][index] = next_generation[i]
    next_cost_list = cf.cost(new_generation, dataset)
    res = list()
    for index in range(NP):
        if cost_list[index] > next_cost_list[index]:
            res.append(next_generation[index])
            CR_rec.append(CR_list[index])
            f_rec.append(cost_list[index] - next_cost_list[index])
            # SaDE
            if function_index[index] == 1:
                ns1 += 1
            elif function_index[index] == 3:
                ns2 += 1
            cost_list[index] = next_cost_list[index]
        else:
            res.append(population[index])
            if function_index[index] == 1:
                nf1 += 1
            elif function_index[index] == 3:
                nf2 += 1
        if cost_list[index] < min_value:
            min_value = cost_list[index]
            min_vec = new_generation[index]
    return res


def SaNSDE(generation, index):
    global age, ns1, ns2, nf1, nf2
    sub_gen = [item[index] for item in generation]
    for i in tqdm(range(config["inner_loop"])):
        sub_gen = evolution(sub_gen, index, generation)
        age = i
        if i != 0 and i % 20 == 0:
            update_CRm()
        if i != 0 and i % 50 == 0:
            update_SaDE_p()
        if i % 100 == 0:
            logger.info(f"now loop is to {i}")
    age, ns1, ns2, nf1, nf2 = 0, 0, 0, 0, 0
    return sub_gen


def divide_groups(length):
    global sub_num
    index = np.random.choice(range(0, length), size=length, replace=False)
    return np.array_split(index, sub_num)


def DE_w(groups_index: list, item: list, w: list):
    global cost_list_w
    w_generation = w.copy()
    cost_list_w = weight_eval(w_generation, item, groups_index)
    # evolve the w population
    for _ in tqdm(range(config["inner_loop"])):
        next_generation = evolve_w(w_generation)
        w_generation = select_w(w_generation, next_generation, item, groups_index)
    return w_generation


def weight_eval(w_generation: list, item: list, groups_index: list):
    global min_value, min_vec
    pop = list()
    for i in range(NP):
        tem = item.copy()
        for j in range(sub_num):
            for k in range(len(groups_index[j])):
                tem[groups_index[j][k]] = item[groups_index[j][k]] * w_generation[i][j]
        for n in range(len(tem)):
            # (id - base) % size + base
            base = dataset.base[item[n]]
            tem[n] = (tem[n] - base[0]) % base[1] + base[0]
        pop.append(tem)
    costs = cf.cost(pop, dataset)
    for i in range(NP):
        if costs[i] < min_value:
            min_value = costs[i]
            min_vec = pop[i]
    return costs


def evolve_w(generation: list):
    next_generation = list()
    for index in range(NP):
        diff = np.zeros(shape=sub_num, dtype=float)
        rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2), replace=False)
        for pair in rand_i:
            diff = diff + generation[pair[0]] - generation[pair[1]]
        new_seq = np.floor(generation[index] + config["F_w"] * diff)
        next_generation.append(crossover_w(generation[index], new_seq))
    return next_generation


def crossover_w(x, v):
    u_j = np.zeros(len(x))
    r = range(len(x))
    for j in r:
        rand = np.random.uniform(0, 1, 1)
        save_index = np.random.choice(r, size=1)
        if rand <= config["CR_w"] or j == save_index:
            u_j[j] = v[j]
        else:
            u_j[j] = x[j]
    return u_j


def select_w(generation: list, next_generation: list, item: list, groups_index: list):
    global cost_list_w
    res = list()
    costs = weight_eval(next_generation, item, groups_index)
    for index in range(NP):
        if cost_list_w[index] > costs[index]:
            cost_list_w[index] = costs[index]
            res.append(next_generation[index])
        else:
            res.append(generation[index])
    return res


def find_candidates(generation: list, value_list: list):
    worst_vec = list()
    worst_value = -99999
    best_vec = list()
    best_value = 99999
    for i in range(NP):
        if value_list[i] > worst_value:
            worst_value = value_list[i]
            worst_vec = generation[i]
        if value_list[i] < best_value:
            best_value = value_list[i]
            best_vec = generation[i]
    rand_value = random.randint(0, NP-1)
    rand_vec = generation[rand_value]
    return best_vec, worst_vec, rand_vec


def DECC_G(code_seq):
    global sub_num, min_vec, min_value
    w = np.ones((NP, sub_num))
    min_vec = np.zeros(len(code_seq))
    generation = init_population(code_seq)
    for _ in tqdm(range(config["outer_loop"])):
        groups_index = divide_groups(len(code_seq))
        for k in range(sub_num):
            w[:, k] = np.random.normal(loc=0, scale=1, size=NP)
            next_generation = SaNSDE(generation, groups_index[k])
            for i in range(len(groups_index[k])):
                for j in range(NP):
                    generation[j][groups_index[k][i]] = next_generation[j][i]
        value_list = cf.cost(generation, dataset)
        best, worst, rand = find_candidates(generation, value_list)
        DE_w(groups_index, best, w)
        DE_w(groups_index, worst, w)
        DE_w(groups_index, rand, w)
        process_recorder.append(min_value)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    date = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args = argparse.ArgumentParser()
    args.add_argument("type", type=str, help="input stream is mRNA or protein structure")
    args.add_argument("input", type=str, help="the mRNA code stream or protein code stream")
    arg = args.parse_args()
    with open('./testConfig/mRNA.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset = DataLoader()
    parser = DataParser()
    process_recorder = list()
    seq = str()
    test_name = str(config["seed"])+"-"+str(config["outer_loop"])+"-"+date
    handler = logging.FileHandler(f'./log/{test_name}.log')
    logger.addHandler(handler)
    random.seed(config["seed"])
    if arg.type == "protein":
        seq = parser.generate_random_seq(arg.input, dataset)
    elif arg.type == "mrna":
        seq = arg.input
    code_seq = dataset.convert2code(seq)
    origin_value = cf.mfe_cost(seq)
    min_value, min_vec = origin_value, code_seq
    NP = config["NP"]
    CR_list = [0.5 for _ in range(NP)]
    sub_num = math.ceil(len(code_seq)/config["sub_size"])
    DECC_G(code_seq)
    p = parser.get_protein(code_seq, dataset)
    logger.info(f"origin sequence mfe: {origin_value:6.2f}")
    logger.info(f"min_mfe: {min_value:6.2f} min seq: {dataset.recover2str(min_vec)}")
    logger.info(f"origin sequence code: {code_seq}")
    logger.info(f"modified sequence code: {min_vec}")
    logger.info(f"If modified mrna has same structure with origin mrna : {dataset.check_type(code_seq, min_vec)}")
    logger.info(f"The protein in Baidu style is : {p}")
    plt.plot(range(len(process_recorder)), process_recorder)
    plt.show()
    plt.savefig(f"./evo_image/{test_name}.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
