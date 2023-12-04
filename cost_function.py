import RNA

from DataLoader import DataLoader


def mfe_cost(seq):
    fc = RNA.fold_compound(seq)
    (ss, mfe) = fc.mfe()
    return mfe


def cost(population: list, loader: DataLoader):
    results = list()
    for vec in population:
        results.append(mfe_cost(loader.recover2str(vec)))
    return results
