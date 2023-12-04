import random

import yaml


class DataLoader:

    def __init__(self):
        with open('testConfig/synonym.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        # [group_id, 密码子集合]
        self.groups = dict()
        # [index, group_id]
        self.code2group = dict()
        # [密码子， index]
        self.str2code = dict()
        # [index, 密码子]
        self.code2str = dict()
        # [index, (base, size)]
        self.base = dict()
        index = 1
        base = 1
        for i in range(1, 22):
            size = len(config[i])
            self.groups[i] = config[i]
            for code in self.groups[i]:
                self.code2group[index] = i
                self.str2code[code] = index
                self.code2str[index] = code
                self.base[index] = (base, size)
                index += 1
            base += size

    # convert the mRNA string to the digit code
    def convert2code(self, mrna: str):
        code = list()
        for i in range(0, int(len(mrna) / 3)):
            code.append(self.str2code[mrna[i * 3:i * 3 + 3]])
        return code

    # convert the digit code to mRNA string(used to calculate the mfe)
    def recover2str(self, code: list):
        mrna = ""
        for index in code:
            mrna = mrna + self.code2str[index]
        return mrna

    def check_type(self, origin, modified):
        for i in range(len(origin)):
            if self.code2group[origin[i]] != self.code2group[modified[i]]:
                return False
        return True


class DataParser:

    def __init__(self):
        self.mapping = dict()
        self.p2code = dict()
        with open('testConfig/fasta_mapping.yaml', 'r', encoding='utf-8') as f:
            self.mapping = yaml.load(f.read(), Loader=yaml.FullLoader)
        for (key, value) in self.mapping.items():
            self.p2code[value] = key

    def get_protein(self, seq, dateset: DataLoader):
        protein = ""
        for code in seq:
            protein = protein + self.mapping[dateset.code2group[code]]
        return protein

    def generate_random_seq(self, protein: str, dataset: DataLoader):
        seq = ""
        for c in protein:
            group = dataset.groups[self.p2code[c]]
            seq = seq + group[random.randint(0, len(group)-1)]
        return seq
