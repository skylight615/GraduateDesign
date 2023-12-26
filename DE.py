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


# generate the v_i using x_i. The difference is calculated by random, x_i also a random member
def evolution(population: list):
    # do permutation for population[index]
    next_generation = list()
    for index in range(NP):
        diff = np.zeros(shape=num, dtype=float)
        rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2), replace=False)
        # rand_i = np.random.choice(range(NP), size=config["y"], replace=False)
        for pair in rand_i:
            diff = diff + population[pair[0]] - population[pair[1]]
        new_seq = np.floor(population[index] + F * diff)
        for n in range(len(new_seq)):
            tem = population[index][n]
            # (id - base) % size + base
            base = dataset.base[tem]
            new_seq[n] = (new_seq[n] - base[0]) % base[1] + base[0]
        next_generation.append(crossover(population[index], new_seq))
    next_generation = select(population, next_generation)
    process_recorder.append(min_value)
    return next_generation


# generate the u_i using v_i and x_i
def crossover(x, v):
    u_j = np.zeros(num)
    r = range(num)
    for j in r:
        rand = np.random.uniform(0, 1, 1)
        save_index = np.random.choice(r, size=1)
        if rand <= CR or j == save_index:
            u_j[j] = v[j]
        else:
            u_j[j] = x[j]
    return u_j


# select between u_i and x_i into next generation, and criterion is cost function
def select(population: list, next_generation: list):
    global min_value, min_vec, cost_list
    next_cost_list = cf.cost(next_generation, dataset)
    res = list()
    for index in range(NP):
        if next_cost_list[index] < min_value:
            min_value = next_cost_list[index]
            min_vec = next_generation[index]
            res.append(next_generation[index])
            cost_list[index] = next_cost_list[index]
            continue
        if cost_list[index] > next_cost_list[index]:
            res.append(next_generation[index])
            cost_list[index] = next_cost_list[index]
        else:
            res.append(population[index])
    return res


def DE(code_seq):
    generation = init_population(code_seq)
    for i in tqdm(range(config["max_gen"])):
        generation = evolution(generation)
        if i % 100 == 0:
            logger.info(f"now loop is to {i}")


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
    NP, F, CR = config["NP"], config["F"], config["CR"]
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
