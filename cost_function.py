import random
import RNA
import time
from DataLoader import DataLoader
import multiprocessing as mp


def mfe_cost(seq):
    fc = RNA.fold_compound(seq)
    (ss, mfe) = fc.mfe()
    return mfe


def cost(population: list, loader: DataLoader):
    # begin = time.time()
    # print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(mfe_cost, [loader.recover2str(vec) for vec in population])
    pool.close()
    pool.join()
    # end = time.time()
    # print(f"cost time: {end - begin}")
    return results
