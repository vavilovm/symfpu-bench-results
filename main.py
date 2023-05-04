import os
import pickle
from itertools import repeat
from multiprocessing import Pool
from statistics import variance, mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pip install pandas, beautifulsoup4, lxml, html5lib

solvers = ["Z3", "SymfpuZ3", "SymfpuYices", "Bitwuzla", "SymfpuBitwuzla"]
solvers_no_z3 = ["SymfpuZ3", "SymfpuYices", "Bitwuzla", "SymfpuBitwuzla"]


class TestResult:
    # 'sample name', 'theory', 'solver', 'assert time', 'check time', 'time', 'status'
    def __init__(self, sample_name: str, theory: str, solver: str, assert_time: np.ndarray, check_time: np.ndarray,
                 time: np.ndarray,
                 status: np.ndarray):
        self.sample_name = sample_name
        self.theory = theory
        self.solver = solver
        self.assert_time = assert_time
        self.check_time = check_time
        self.time = time
        self.status = status

    def __str__(self):
        return f'TestResult: ({self.sample_name} ' \
               f'{self.theory} {self.solver} {self.assert_time} {self.check_time} {self.time} {self.status})'

    def all_unknown(self):
        return np.all(self.status == "UNKNOWN")

    def array_without_unknowns(self, array: np.ndarray):
        return np.asarray([x for i, x in enumerate(array) if self.status[i] != "UNKNOWN"])

    def get_average_time(self):
        # mean of not unknown status
        if self.all_unknown():
            return None
        return mean(self.array_without_unknowns(self.check_time))

    def get_max_time(self):
        if self.all_unknown():
            return None
        return max(self.array_without_unknowns(self.time))

    def get_variance_time(self):
        if self.all_unknown():
            return None
        arr = self.array_without_unknowns(self.time)
        if len(arr) < 2:
            return None
        return variance(self.array_without_unknowns(self.time))

    def get_known_count(self):
        return np.count_nonzero(self.status != "UNKNOWN")

    #     constructor from list
    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6])


def f(solver, data_dir, show_z3):
    if solver == "Z3" and not show_z3:
        return None, None

    d = np.load(f'{data_dir}/{solver}.npy')
    x = np.linspace(0.0, 100.0, num=d.size, endpoint=True)
    d = np.percentile(d, x)

    # plt.plot(x, d, label=solver)
    return x, d

def show_graph(data_dir='data/np-comp-check', show_z3=False, log=True):
    with Pool() as pool:
        xd = pool.starmap(f, zip(solvers, repeat(data_dir), repeat(show_z3)))
        for (x, d), solver in zip(xd, solvers):
            if x is not None:
                plt.plot(x, d, label=solver)
    if log:
        plt.yscale('log', base=2)
    # plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.ylabel("Relative time")
    plt.xlabel("Percentile")
    plt.title("Relative time to Z3")
    plt.savefig('compare.png')
    plt.show()


def load_dict(solver: str) -> dict[str, TestResult]:
    with open(f'data/dicts/{solver}.pkl', 'rb') as f:
        return pickle.load(f)


def create_and_save_dict(solver: str):
    file = f'data/{solver}.csv'
    df = pd.read_csv(file, index_col=False)

    #     create dict from non-unique test names to list of TestResults
    def function(x):
        # sample_name, theory, solver, assert_time, check_time, time, status
        sample_name = x['sample name'].iloc[0]
        theory = x['theory'].iloc[0]
        solver = x['solver'].iloc[0]
        assert_time = x['assert time'].to_numpy()
        check_time = x['check time'].to_numpy()
        time = x['time'].to_numpy()
        status = x['status'].to_numpy()
        return TestResult(sample_name, theory, solver, assert_time, check_time, time, status)

    res = df.groupby('sample name').apply(function).to_dict()

    # mkdirs
    os.makedirs('data/dicts', exist_ok=True)
    with open(f'data/dicts/{solver}.pkl', 'wb') as f:
        pickle.dump(res, f)
    return res


def save_comp_to_z3(solver, z3_dict, out_dir='data/np-comp-check'):
    print(f'Computing {solver}')
    if solver == 'Z3':
        return
    compare_to_z3 = []  # solver -> np array of relative times
    d = load_dict(solver)
    print(f'Loaded {solver}')
    for sample, results in d.items():
        if sample not in z3_dict:
            continue
        z3_results = z3_dict[sample]
        if results.all_unknown() or z3_results.all_unknown():
            continue
        a = results.get_average_time()
        z3 = z3_results.get_average_time()
        compare_to_z3.append(a / z3)
    np.save(f'{out_dir}/{solver}.npy', np.sort(np.asarray(compare_to_z3)))
    print(f'Saved {solver}')


def save_np_array(solver, out_dir, f=lambda x: x.get_average_time()):
    print(f'Computing {solver}')
    compare_to_z3 = []  # solver -> np array
    d = load_dict(solver)
    print(f'Loaded {solver}')
    for sample, results in d.items():
        a = f(results)
        if a is not None:
            compare_to_z3.append(a)
    np.save(f'{out_dir}/{solver}.npy', np.sort(np.asarray(compare_to_z3)))
    print(f'Saved {solver}')


def print_percentes(dir, show_z3=False):
    for solver in solvers:
        if solver == "Z3" and not show_z3:
            continue
        d = np.load(f'{dir}/{solver}.npy')

        print(f'{solver} size: {d.size}')
        percentiles = [0, 5, 15, 20, 25, 50, 90, 95, 100]
        for p in percentiles:
            print(f'    {p}: {np.percentile(d, p)}')


if __name__ == '__main__':
    # merge all csv files in ./data folder
    files = [f'data/{solver}.csv' for solver in solvers]
    header = ['sample name', 'theory', 'solver', 'assert time', 'check time', 'time', 'status']

    # to add header and change sep in all csv files
    # for solver in solvers:
    # solver = 'Bitwuzla'
    # f = f'data/{solver}.csv'
    # print(f)
    # df = pd.read_csv(f, header=None, sep=' \| ', index_col=False, skipinitialspace=True)
    # df.to_csv(f'data/{solver}.csv', index=False, header=header)

    # to create dict for each solver
    # with Pool() as pool:
    #     result = pool.map(create_and_save_dict, solvers)

    print('Creating Z3 dict')

    z3_dict = load_dict('Z3')
    path = 'data/np-comp-check'
    os.makedirs(path, exist_ok=True)
    print('start save to np')

    with Pool() as pool:
        pool.starmap(save_comp_to_z3, zip(solvers, repeat(z3_dict), repeat(path)))

    print('start show graph')
    show_graph(path, show_z3=False, log=True)

    print('start print percentiles')
    print_percentes(path, show_z3=False)

    pass
