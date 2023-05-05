import os
import pickle
from itertools import repeat
from multiprocessing import Pool
from statistics import variance, mean
from typing import Any

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

    def all_unknown(self) -> bool:
        return np.all(self.status == "UNKNOWN")

    def array_without_unknowns(self, array: np.ndarray):
        return np.asarray([x for i, x in enumerate(array) if self.status[i] != "UNKNOWN"])

    def get_average_time(self):
        # mean of not unknown status
        if self.all_unknown():
            return None
        a = self.array_without_unknowns(self.time)
        return mean(a)

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
    with Pool() as p:
        xd = p.starmap(f, zip(solvers, repeat(data_dir), repeat(show_z3)))
        for (x, d), solver in zip(xd, solvers):
            if x is not None:
                plt.plot(x, d, label=solver)
    plt.axhline(y=1, label='Z3', color='purple')
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


DICTS_DIR = 'data/dicts'


def load_dict(solver: str) -> dict[str, TestResult]:
    with open(f'{DICTS_DIR}/{solver}.pkl', 'rb') as file:
        return pickle.load(file)


def string_np_array_to_int(a: np.ndarray):
    return np.array([int(x) for x in a])


def create_and_save_dict(solver: str):
    file = f'data/{solver}.csv'
    df = pd.read_csv(file, index_col=False)
    print(f'create dict {solver}')

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
    os.makedirs(DICTS_DIR, exist_ok=True)
    with open(f'{DICTS_DIR}/{solver}.pkl', 'wb') as out_file:
        pickle.dump(res, out_file)
    return res


def save_comp_to_z3(solver: str, z3_dict_to_comp: dict[str, TestResult], out_dir: str = 'data/np-comp-check'):
    if solver == 'Z3':
        return
    print(f'Computing {solver}')
    compare_to_z3: list[Any] = []  # solver -> np array of relative times
    d: dict[str, TestResult] = load_dict(solver)
    print(f'Loaded {solver}')
    for sample, results in d.items():
        if sample not in z3_dict_to_comp:
            continue
        z3_results = z3_dict_to_comp[sample]
        if results.all_unknown() or z3_results.all_unknown():
            continue
        a = results.get_average_time()
        z3 = z3_results.get_average_time()
        compare_to_z3.append(a / z3)
    np.save(f'{out_dir}/{solver}.npy', np.sort(np.asarray(compare_to_z3)))
    print(f'Saved {solver}')


def save_np_array(solver, out_dir, method=lambda x: x.get_average_time()):
    print(f'Computing {solver}')
    compare_to_z3 = []  # solver -> np array
    d = load_dict(solver)
    print(f'Loaded {solver}')
    for sample, results in d.items():
        a = method(results)
        if a is not None:
            compare_to_z3.append(a)
    np.save(f'{out_dir}/{solver}.npy', np.sort(np.asarray(compare_to_z3)))
    print(f'Saved {solver}')


def print_percentes(input_dir, show_z3=False):
    for solver in solvers:
        if solver == "Z3" and not show_z3:
            continue
        d = np.load(f'{input_dir}/{solver}.npy')

        print(f'{solver} size: {d.size}')
        percentiles = [0, 5, 15, 20, 25, 50, 90, 95, 100]
        for p in percentiles:
            print(f'    {p}: {np.percentile(d, p)}')


if __name__ == '__main__':
    # merge all csv files in ./data folder
    files = [f'data/{solver}.csv' for solver in solvers]
    header = ['sample name', 'theory', 'solver', 'assert time', 'check time', 'time', 'status']

    # for solver in solvers:
    #     f = f'data/{solver}.csv'
    #     print(f)
    #     df = pd.read_csv(f, index_col=False)
    #     df.loc[df["sample name"] != 'sample name'].to_csv(f'data/{solver}.csv', index=False, header=header)


    # to create dict for each solver
    # with Pool() as pool:
    #     result = pool.map(create_and_save_dict, solvers)

    # print('Creating Z3 dict')
    #

    z3_dict = load_dict('Z3')
    path = 'data/np-comp-check'
    os.makedirs(path, exist_ok=True)
    # print('start save to np')
    # with Pool() as pool:
    #     pool.starmap(save_comp_to_z3, zip(solvers, repeat(z3_dict), repeat(path)))

    # print('start show graph')
    show_graph(path, show_z3=False, log=True)

    # print('start print percentiles')
    # print_percentes(path, show_z3=False)

    # print first from z3_dict
    # for sample, results in z3_dict.items():
    #     print(sample)
    #     print(results)
    #     break

    pass
