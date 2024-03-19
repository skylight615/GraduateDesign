import random
import RNA
import math
from DataLoader import DataLoader
import multiprocessing as mp


def mfe_cost(seq):
    fc = RNA.fold_compound(seq)
    (ss, mfe) = fc.mfe()
    return mfe


def cai_cost(item, dataset: DataLoader):
    res = 0
    code_len = int(len(item))
    for i in range(0, code_len):
        code = dataset.code2str[item[i]]
        res += math.log2(dataset.codon_usage[code])
    return res


def CAI_cost(item, dataset: DataLoader):
    res = 1
    code_len = int(len(item))
    for i in range(0, code_len):
        code = dataset.code2str[item[i]]
        res *= dataset.codon_usage[code]
    res = pow(res, 1/code_len)
    return res


def cost(population: list, loader: DataLoader, lamda: float):
    # begin = time.time()
    # print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    str_population = [loader.recover2str(vec) for vec in population]
    results = pool.map(mfe_cost, str_population)
    pool.close()
    pool.join()
    if lamda != 0:
        for i in range(len(results)):
            results[i] = results[i] - lamda * cai_cost(population[i], loader)
    # end = time.time()
    # print(f"cost time: {end - begin}")
    return results


if __name__ == '__main__':
    loader = DataLoader()
    # lambda = 10
    # [-886.1536694955073]
    # -899.9000244140625
    # 0.9848222246527519
    seq = "AUGCUGGAUCAGGUGAACAAGCUGAAGUACCCCGAGGUGAGCCUGACCUGA"
    codes = loader.convert2code(seq)
    result = cost([codes], loader, 1)
    mfe = mfe_cost(seq)
    cai = CAI_cost(codes, loader)
    print(result)
    print(mfe)
    print(cai)
