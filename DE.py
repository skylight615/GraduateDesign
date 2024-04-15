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
GT_p = 0.5
candidates = list()
p_list, success = list(), list()
age = 0
unused = 0
dataRecorder = list()
ns1, ns2, nf1, nf2 = 0, 0, 0, 0
f_rec = list()
CR_list = list()
CR_rec = list()
GT_rec = list()


# init the population, size = NP, and the member is a vector (1,num_parameter).
# The value in vector should in the range of IPR.
def init_population(init_code: list):
    global cost_list, candidates
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
    candidates = np.argpartition(cost_list, -5)[-5:]
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
    if (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)) == 0:
        SaDE_p = 0.5
    else:
        SaDE_p = ns1 * (ns2 + nf2) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2))
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


def update_GT():
    global GT_p, p_list, success
    sum_diff = sum(success)
    w = [i / sum_diff for i in success]
    if GT_p <= 0.01:
        GT_p = 0.1
    else:
        GT_p = sum([w[i] * p_list[i] for i in range(len(w))])
    p_list.clear()
    success.clear()
    logger.info(f"GT_p: {GT_p}")


def evolve(population: list, F: float):
    global min_value, min_vec, GT_rec, p_list, success
    next_generation = list()
    function_index = list()
    for index in range(NP):
        function_index.append(1)
        GT_target = np.random.choice(candidates, size=1)
        if GT_target != index:
            diff = np.zeros(shape=num, dtype=float)
            rand_i = np.random.choice(list(range(0, index)) + list(range(index, NP)), size=(config["y"], 2),
                                      replace=False)
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
            buffer = list()
            ss = cf.get_structure(dataset.recover2str(population[index]))
            unpaired = [i for i in range(len(ss)) if ss[i] == '.']
            for _ in range(config["NGT"]):
                bottleneck_dims = target_bottleneck(unpaired)
                if len(bottleneck_dims) == 0:
                    bottleneck_dims = [np.random.randint(0, len(population[0]))]
                new_item = construct_vec(bottleneck_dims, population, index)
                buffer.append(new_item)
            costs = cf.cost(buffer, dataset, config["lambda"])
            diffs = [costs[i] - cost_list[index] for i in range(len(buffer))]
            tmp = population[index].copy()
            for i in range(len(buffer)):
                if diffs[i] < 0:
                    success.append(diffs[i])
                    p_list.append(GT_rec[i])
                if costs[i] < cost_list[index]:
                    cost_list[index] = costs[i]
                    tmp = buffer[i]
            GT_rec.clear()
            next_generation.append(tmp)
    return next_generation, function_index


def target_bottleneck(unpaired: list):
    global GT_p, GT_rec
    res = []
    p = 0
    while p <= 0:
        p = np.random.normal(GT_p, 0.1, 1)
    for i in unpaired:
        # p = Gaussian(0.01, 0.01)
        rand = np.random.uniform(0, 1)
        if rand < p:
            res.append(i)
    GT_rec.append(p)
    return res


def construct_vec(bottleneck_dims: list, population: list, index: int):
    F_c = np.random.normal(0.5, 0.1, 1)
    item1_id, item2_id = np.random.choice(range(len(population)), size=2, replace=False)
    item1, item2 = population[item1_id], population[item2_id]
    new_item = population[index].copy()
    for i in bottleneck_dims:
        rand = np.random.uniform(0, 1)
        if rand < config["P_m"]:
            # rand_item != item1 and item2
            while True:
                rand_item_id = np.random.choice(range(len(population)))
                rand_item = population[rand_item_id]
                if rand_item_id != item1_id and rand_item_id != item2_id:
                    break
            new_item[i] = np.floor(new_item[i] + F_c * (item1[i] - rand_item[i]))
        else:
            new_item[i] = np.floor(new_item[i] + F_c * (item1[i] - item2[i]))
        base = dataset.base[population[index][i]]
        new_item[i] = (new_item[i] - base[0]) % base[1] + base[0]
    return new_item


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
    global min_value, min_vec, cost_list, ns1, ns2, nf1, nf2, f_rec, CR_rec, candidates
    next_cost_list = cf.cost(next_generation, dataset, config["lambda"])
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
    candidates = np.argpartition(cost_list, -5)[-5:]
    return res


def DE(code_seq):
    global age
    generation = init_population(code_seq)
    for i in tqdm(range(config["max_gen"])):
        generation = evolution(generation)
        age = i
        logger.info(f"echo:{age} min_mfe: {min_value:6.2f}")
        if i != 0 and i % 10 == 0:
            update_CRm()
        if i != 0 and i % 50 == 0:
            update_SaDE_p()
        if i != 0 and i % 10 == 0:
            update_GT()
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
    test_name = str(config["seed"]) + "-" + str(config["max_gen"]) + "-" + date
    handler = logging.FileHandler(f'./log/GTDE/{test_name}.log')
    logger.addHandler(handler)
    # random.seed(config["seed"])
    if arg.type == "protein":
        seq = parser.generate_random_seq(arg.input, dataset)
    elif arg.type == "mrna":
        seq = arg.input
    code_seq = dataset.convert2code(seq)
    num = len(code_seq)
    origin_value = cf.cost([code_seq], dataset, config["lambda"])[0]
    min_value, min_vec = origin_value, code_seq
    NP = config["NP"]
    CR_list = [0.5 for _ in range(NP)]
    logger.info(f"NP: {NP} stop: {config['stop']} lambda: {config['lambda']}")
    DE(code_seq)
    p = parser.get_protein(code_seq, dataset)
    logger.info(f"origin sequence mfe: {origin_value:6.2f}")
    logger.info(f"min_cost: {min_value:6.2f} min_mfe:{cf.mfe_cost(dataset.recover2str(min_vec))}"
                f"min_cai: {cf.CAI_cost(min_vec, dataset)} min seq: {dataset.recover2str(min_vec)}")
    logger.info(f"origin sequence code: {code_seq}")
    logger.info(f"modified sequence code: {min_vec}")
    logger.info(f"If modified mrna has same structure with origin mrna : {dataset.check_type(code_seq, min_vec)}")
    logger.info(f"The protein in Baidu style is : {p}")
    # plt.plot(range(len(process_recorder)), process_recorder)
    # plt.savefig(f"./evo_image/GTDE/{test_name}.png")
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
