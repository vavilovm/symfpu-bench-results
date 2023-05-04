import glob
import os
import pickle
import shutil
from multiprocessing import Pool
from statistics import variance, mean

import pandas as pd
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# pip install pandas, beautifulsoup4, lxml, html5lib

solvers = ["Z3", "SymfpuZ3", "SymfpuYices", "Bitwuzla", "SymfpuBitwuzla",
           # "Z3TransformedNoBvOpt", "YicesNoBvOpt", "BitwuzlaTransformedNoBvOpt"
           ]


# def print_aggregate_stats():
#     global solver, test_name, results, res
#     cnt_failed = {}
#     cnt_unknown = {}
#     time_sum = {}
#     all_successful_time_sum = {}
#     cnt_best = {}
#     cnt_worst = {}
#     for solver in solvers:
#         cnt_failed[solver] = (data[(data["Method name"] == solver) & (data["Result"] == "failed")].size)
#         cnt_unknown[solver] = (data[(data["Method name"] == solver) & (data["Result"] == "ignored")].size)
#         time_sum[solver] = data[data["Method name"] == solver]["Duration"].sum()
#         cnt_best[solver] = 0
#         cnt_worst[solver] = 0
#         all_successful_time_sum[solver] = 0
#     results_copy = deepcopy(test_results)
#     for test_name, results in results_copy.items():
#         # if all results are successful
#         all_ok = True
#         for res in results:
#             if res[2] != "passed":
#                 all_ok = False
#         if not all_ok:
#             continue
#
#         results.sort(key=lambda x: x[1])
#         cnt_best[results[0][0]] += 1
#         cnt_worst[results[-1][0]] += 1
#         for res in results:
#             all_successful_time_sum[res[0]] += res[1]
#     print("Time sum:")
#     print(time_sum)
#     print("All successful Time sum:")
#     print(all_successful_time_sum)
#     print("Best:")
#     print(cnt_best)
#     print("Worst:")
#     print(cnt_worst)
#     print("Failed:")
#     print(cnt_failed)
#     print("Unknown:")
#     print(cnt_unknown)


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
        return mean(self.array_without_unknowns(self.time))

    def get_variance_time(self):
        if self.all_unknown():
            return None
        return variance(self.array_without_unknowns(self.time))

    #     constructor from list
    @classmethod
    def from_list(cls, l):
        return cls(l[0], l[1], l[2], l[3], l[4], l[5], l[6])


def show_graph():
    compare_to_z3 = {}

    # # load np arrays
    for solver in solvers:
        if solver == "Z3":
            continue
        path = f'data/np-comp/{solver}.npy'
        compare_to_z3[solver] = np.load(path)

        # skip z3_dict, bitwuzla
        # if solver == "Z3" or solver == "Bitwuzla":
        #     continue
        d = compare_to_z3[solver]
        x = np.linspace(0.0, 100.0, num=d.size, endpoint=True)
        d = np.percentile(d, x)

        plt.plot(x, d, label=solver)

    plt.yscale('log')
    # plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.ylabel("Relative time")
    plt.xlabel("Percentile")
    plt.title("Relative time to Z3")
    plt.savefig('compare.png')
    plt.show()


def load_dict(solver: str):
    with open(f'data/dicts/{solver}.pkl', 'rb') as f:
        return pickle.load(f)


def get_dict(solver: str):
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

    d = df.groupby('sample name').apply(function).to_dict()

    # mkdirs
    os.makedirs('data/dicts', exist_ok=True)
    with open(f'data/dicts/{solver}.pkl', 'wb') as f:
        pickle.dump(d, f)
    return d


def save_comp_to_z3(solver):
    print(f'Computing {solver}')
    if solver == 'Z3':
        return
    z3_dict = load_dict('Z3')
    compare_to_z3 = []  # solver -> np array of relative times
    d = load_dict(solver)
    # d = dicts[solver]
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
        np.save(f'data/np-comp/{solver}.npy', np.asarray(compare_to_z3))
    print(f'Saved {solver}')



if __name__ == '__main__':
    # merge all csv files in ./data folder
    files = [f'data/{solver}.csv' for solver in solvers]
    header = ['sample name', 'theory', 'solver', 'assert time', 'check time', 'time', 'status']

    # print(files)
    # with open('data/all.csv', 'wb') as wfd:
    #     for f in files:
    #         with open(f, 'rb') as fd:
    #             shutil.copyfileobj(fd, wfd)

    # for solver in solvers:
    #     f = f'data/{solver}.csv'
    #     print(f)
    #     df = pd.read_csv(f, header=None, sep=' \| ', index_col=False, skipinitialspace=True)
    #     df.to_csv(f'data/{solver}.csv', index=False, header=header)

    # for file in files:
    # file = 'data/Z3.csv'
    # print(file)
    # get_dict()
    # for key, value in d.items():
    #     print(key, value)
    #     break

    # with Pool() as pool:
    #     result = pool.map(get_dict, solvers)
    test_name = 'QF_FP_sqrt-has-no-other-solution-18743.smt2'



    #     test_results = {}
    # for index, row in df.iterrows():
    #     test_result = TestResult.from_list(row.tolist())
    #     if test_result.sample_name not in test_results:
    #         test_results[test_result.sample_name] = []
    #     test_results[test_result.sample_name].append(test_result)

    # df = pd.read_csv('data/all.csv', index_col=False)
    # # create dict: solver -> test_name -> [TestResults]
    #
    #
    # print(df[df['sample name'] == test_name])
    #
    # print("dict:")
    # d = df.set_index('sample name').T.to_dict('list')
    # for key, value in d.items():
    #     print(key, value)
    #     break
    # print(d[test_name])

    dicts = {}
    for solver in solvers:
        d = load_dict(solver)
        dicts[solver] = d

    # z3_dict = dicts['Z3']

    # os.makedirs('data/np-comp', exist_ok=True)
    # with Pool() as pool:
    #     pool.map(save_comp_to_z3, solvers)

    # show_graph()

    # for solver in solvers:
    #     if solver == "Z3":
    #         continue
    #     path = f'data/np-comp/{solver}.npy'
    #     d = np.load(path)
    #
    #     print(f'{solver} size: {d.size}')
    #     print(f'    25: {np.percentile(d, 25)}')
    #     print(f'    50: {np.percentile(d, 50)}')
    #     print(f'    90: {np.percentile(d, 90)}')
    #     print(f'    95: {np.percentile(d, 95)}')


        # skip z3_dict, bitwuzla
        # if solver == "Z3" or solver == "Bitwuzla":
        #     continue


    pass
