import random
import time
import cost_function as cf
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import matplotlib.pyplot as plt
import logging

from DataLoader import DataLoader, DataParser

min_vec = 0  # record the vector that has min cost value now
min_value = 99999  # record the current min value during evolution
cost_list = list()
CRm = 0.5
SaDE_p = 0.5
age = 0
unused = 0
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


def evolve(population: list, F: float):
    global min_value, min_vec
    next_generation = list()
    function_index = list()
    for index in range(NP):
        function_index.append(1)
        if min_vec != population[index]:
            diff = np.zeros(shape=num, dtype=float)
            rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2), replace=False)
            for pair in rand_i:
                diff = diff + population[pair[0]] - population[pair[1]]
            # SaDE
            p = np.random.uniform(0, 1, 1)
            if p > SaDE_p:
                diff += min_vec - population[index]
                function_index[index] = 3
            new_seq = np.floor(population[index] + F * diff)
            for n in range(len(new_seq)):
                tem = population[index][n]
                # (id - base) % size + base
                base = dataset.base[tem]
                new_seq[n] = (new_seq[n] - base[0]) % base[1] + base[0]
            next_generation.append(crossover(population[index], new_seq, index))
        else:
            # do the GTDE for the best member
            for _ in range(config["NGT"]):
                bottleneck_dims = target_bottleneck(len(population[0]))
                new_best = construct_vec(bottleneck_dims, population)
                new_cost = cf.mfe_cost(dataset.recover2str(new_best))
                if new_cost < min_value:
                    min_value = new_cost
                    min_vec = new_best
                    next_generation.append(new_best)
                else:
                    next_generation.append(population[index])
    return next_generation, function_index


def target_bottleneck(dim: int):
    res = []
    for i in range(dim):
        # p = Gaussian(0.01, 0.01)
        p = np.random.normal(0.01, 0.01, 1)
        rand = np.random.uniform(0, 1)
        if rand < p:
            res.append(i)
    return res


def construct_vec(bottleneck_dims: list, population: list):
    F_c = np.random.normal(0.5, 0.1, 1)
    item1, item2 = np.random.choice(population, size=2, replace=False)
    new_best = min_vec.copy()
    for i in bottleneck_dims:
        rand = np.random.uniform(0, 1)
        if rand < config["P_m"]:
            # rand_item != item1 and item2
            while True:
                rand_item = np.random.choice(population, size=1, replace=False)
                if rand_item != item1 and rand_item != item2:
                    break
            new_best[i] = min_vec[i] + F_c * (item1[i] - rand_item[i])
        else:
            new_best[i] = min_vec[i] + F_c * (item1[i] - item2[i])
        base = dataset.base[new_best[i]]
        new_best[i] = (new_best[i] - base[0]) % base[1] + base[0]
    return new_best


# generate the v_i using x_i. The difference is calculated by random, x_i also a random member
def evolution(population: list):
    # do permutation for population[index]
    global min_value, min_vec, unused
    F = generate_f()
    tmp = min_value
    next_generation, function_index = evolve(population, F)
    next_generation = select(population, next_generation, function_index)
    if tmp == min_value:
        unused += 1
    else:
        unused = 0
    process_recorder.append(min_value)
    return next_generation


# generate the u_i using v_i and x_i
def crossover(x, v, index):
    global CRm, CR_list
    u_j = np.zeros(num)
    r = range(num)
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
def select(population: list, next_generation: list, function_index: list):
    global min_value, min_vec, cost_list, ns1, ns2, nf1, nf2, f_rec, CR_rec
    next_cost_list = cf.cost(next_generation, dataset)
    res = list()
    for index in range(NP):
        if next_cost_list[index] < min_value:
            min_value = next_cost_list[index]
            min_vec = next_generation[index]
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
    return res


def DE(code_seq):
    global age
    generation = init_population(code_seq)
    for i in tqdm(range(config["max_gen"])):
        generation = evolution(generation)
        age = i
        if i != 0 and i % 20 == 0:
            update_CRm()
        if i != 0 and i % 50 == 0:
            update_SaDE_p()
        if i % 100 == 0:
            logger.info(f"now loop is to {i}")
        if unused > 1000:
            break


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
    test_name = str(config["seed"])+"-"+str(config["max_gen"])+"-"+date
    handler = logging.FileHandler(f'./log/{test_name}.log')
    logger.addHandler(handler)
    random.seed(config["seed"])
    if arg.type == "protein":
        seq = parser.generate_random_seq(arg.input, dataset)
    elif arg.type == "mrna":
        seq = arg.input
    code_seq = dataset.convert2code(seq)
    num = len(code_seq)
    origin_value = cf.mfe_cost(seq)
    min_value, min_vec = origin_value, code_seq
    NP = config["NP"]
    CR_list = [0.5 for _ in range(NP)]
    DE(code_seq)
    p = parser.get_protein(code_seq, dataset)
    logger.info(f"origin sequence mfe: {origin_value:6.2f}")
    logger.info(f"min_mfe: {min_value:6.2f} min seq: {dataset.recover2str(min_vec)}")
    logger.info(f"origin sequence code: {code_seq}")
    logger.info(f"modified sequence code: {min_vec}")
    logger.info(f"If modified mrna has same structure with origin mrna : {dataset.check_type(code_seq, min_vec)}")
    logger.info(f"The protein in Baidu style is : {p}")
    plt.plot(range(config["max_gen"]), process_recorder)
    plt.show()
    plt.savefig(f"./evo_image/{test_name}.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
