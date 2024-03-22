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
age = 0
unused = 0
dataRecorder = list()


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
    cost_list = cf.cost(population, dataset, config["lambda"])
    return population


def evolve(population: list, F: float):
    global min_value, min_vec
    next_generation = list()
    for index in range(NP):
        if min_value != cost_list[index]:
            diff = np.zeros(shape=num, dtype=float)
            rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2), replace=False)
            for pair in rand_i:
                diff = diff + population[pair[0]] - population[pair[1]]
            new_seq = np.floor(population[index] + F * diff)
            for n in range(len(new_seq)):
                tem = population[index][n]
                # (id - base) % size + base
                base = dataset.base[tem]
                new_seq[n] = (new_seq[n] - base[0]) % base[1] + base[0]
            next_generation.append(crossover(population[index], new_seq))
        else:
            # do the GTDE for the best member
            if age % 50 == 0:
                buffer = list()
                for _ in range(config["NGT"]):
                    bottleneck_dims = list()
                    while len(bottleneck_dims) == 0:
                        bottleneck_dims = target_bottleneck(len(population[0]))
                    new_best = construct_vec(bottleneck_dims, population, index)
                    buffer.append(new_best)
                costs = cf.cost(buffer, dataset, config["lambda"])
                for i in range(len(buffer)):
                    if config["save"] == 1:
                        dataRecorder.append((buffer[i], costs[i]))
                    if costs[i] < min_value:
                        min_value = costs[i]
                        min_vec = buffer[i]
                        cost_list[index] = min_value
            next_generation.append(min_vec)
    return next_generation


def target_bottleneck(dim: int):
    res = []
    for i in range(dim):
        # p = Gaussian(0.01, 0.01)
        p = np.random.normal(0.01, 0.01, 1)
        rand = np.random.uniform(0, 1)
        if rand < p:
            res.append(i)
    return res


def construct_vec(bottleneck_dims: list, population: list, index: int):
    F_c = np.random.normal(0.5, 0.1, 1)
    item1_id, item2_id = np.random.choice(range(len(population)), size=2, replace=False)
    item1, item2 = population[item1_id], population[item2_id]
    new_best = min_vec.copy()
    for i in bottleneck_dims:
        rand = np.random.uniform(0, 1)
        if rand < config["P_m"]:
            # rand_item != item1 and item2
            while True:
                rand_item_id = np.random.choice(range(len(population)))
                rand_item = population[rand_item_id]
                if rand_item_id != item1_id and rand_item_id != item2_id:
                    break
            new_best[i] = np.floor(min_vec[i] + F_c * (item1[i] - rand_item[i]))
        else:
            new_best[i] = np.floor(min_vec[i] + F_c * (item1[i] - item2[i]))
        base = dataset.base[population[index][i]]
        new_best[i] = (new_best[i] - base[0]) % base[1] + base[0]
    return new_best


# generate the v_i using x_i. The difference is calculated by random, x_i also a random member
def evolution(population: list):
    # do permutation for population[index]
    global min_value, min_vec, unused
    tmp = min_value
    next_generation = evolve(population, F)
    next_generation = select(population, next_generation)
    if tmp == min_value:
        unused += 1
    else:
        unused = 0
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
    next_cost_list = cf.cost(next_generation, dataset, config["lambda"])
    if config["save"] == 1:
        for i in range(len(next_cost_list)):
            dataRecorder.append((next_generation[i], next_cost_list[i]))
    res = list()
    for index in range(NP):
        if next_cost_list[index] < min_value:
            min_value = next_cost_list[index]
            min_vec = next_generation[index]
        if cost_list[index] > next_cost_list[index]:
            res.append(next_generation[index])
            cost_list[index] = next_cost_list[index]
        else:
            res.append(population[index])
    return res


def DE(code_seq):
    global age
    generation = init_population(code_seq)
    for i in tqdm(range(config["max_gen"])):
        generation = evolution(generation)
        age = i
        logger.info(f"echo:{age} min_mfe: {min_value:6.2f}")
        if i % 100 == 0:
            logger.info(f"now loop is to {i}")
        if unused > config["stop"]:
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
    handler = logging.FileHandler(f'./log/GTDE-O/{test_name}.log')
    logger.addHandler(handler)
    random.seed(config["seed"])
    if arg.type == "protein":
        seq = parser.generate_random_seq(arg.input, dataset)
    elif arg.type == "mrna":
        seq = arg.input
    code_seq = dataset.convert2code(seq)
    num = len(code_seq)
    origin_value = cf.cost([code_seq], dataset, config["lambda"])[0]
    min_value, min_vec = origin_value, code_seq
    F, NP, CR = config["F"], config["NP"], config["CR"]
    CR_list = [0.5 for _ in range(NP)]
    logger.info(f"NP: {NP} stop: {config['stop']} lambda: {config['lambda']}")
    DE(code_seq)
    if config["save"] == 1:
        np.save(f"./data/data.npy", dataRecorder)
    p = parser.get_protein(code_seq, dataset)
    logger.info(f"origin sequence mfe: {origin_value:6.2f}")
    logger.info(f"min_cost: {min_value:6.2f} min_mfe:{cf.mfe_cost(dataset.recover2str(min_vec))}"
                f"min_cai: {cf.CAI_cost(min_vec, dataset)} min seq: {dataset.recover2str(min_vec)}")
    logger.info(f"origin sequence code: {code_seq}")
    logger.info(f"modified sequence code: {min_vec}")
    logger.info(f"If modified mrna has same structure with origin mrna : {dataset.check_type(code_seq, min_vec)}")
    logger.info(f"The protein in Baidu style is : {p}")
    plt.plot(range(len(process_recorder)), process_recorder)
    plt.savefig(f"./evo_image/GTDE-O/{test_name}.png")
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
