import os
from statistics import variance, mean

import pandas as pd
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# pip install pandas, beautifulsoup4, lxml, html5lib

solvers = ["Z3", "Z3Transformed", "Yices", "Bitwuzla", "BitwuzlaTransformed",
           # "Z3TransformedNoBvOpt", "YicesNoBvOpt", "BitwuzlaTransformedNoBvOpt"
           ]


def get_solver_name(str: str):
    if str.startswith("testSolverZ3TransformedNoBvOpt"):
        return "Z3TransformedNoBvOpt"
    elif str.startswith("testSolverZ3Transformed"):
        return "Z3Transformed"
    elif str.startswith("testSolverZ3"):
        return "Z3"
    elif str.startswith("testSolverYicesNoBvOpt"):
        return "YicesNoBvOpt"
    elif str.startswith("testSolverYices"):
        return "Yices"
    elif str.startswith("testSolverBitwuzlaTransformedNoBvOpt"):
        return "BitwuzlaTransformedNoBvOpt"
    elif str.startswith("testSolverBitwuzlaTransformed"):
        return "BitwuzlaTransformed"
    elif str.startswith("testSolverBitwuzla"):
        return "Bitwuzla"
    else:
        raise Exception(f"Unknown solver name: {str}")


def html_to_csv(filename: str):
    # read html tables using pandas
    tables = pd.read_html(f'{filename}.html')
    # get last table
    table = tables[-1]
    # remove last char from Duration column
    table["Duration"] = table["Duration"].str[:-1]
    table["Method name"] = table["Method name"].map(get_solver_name)

    print("save to csv")
    # save it to csv
    table.to_csv(f'{filename}.csv', index=False)


def print_aggregate_stats():
    global solver, test_name, results, res
    cnt_failed = {}
    cnt_unknown = {}
    time_sum = {}
    all_successful_time_sum = {}
    cnt_best = {}
    cnt_worst = {}
    for solver in solvers:
        cnt_failed[solver] = (data[(data["Method name"] == solver) & (data["Result"] == "failed")].size)
        cnt_unknown[solver] = (data[(data["Method name"] == solver) & (data["Result"] == "ignored")].size)
        time_sum[solver] = data[data["Method name"] == solver]["Duration"].sum()
        cnt_best[solver] = 0
        cnt_worst[solver] = 0
        all_successful_time_sum[solver] = 0
    results_copy = deepcopy(test_results)
    for test_name, results in results_copy.items():
        # if all results are successful
        all_ok = True
        for res in results:
            if res[2] != "passed":
                all_ok = False
        if not all_ok:
            continue

        results.sort(key=lambda x: x[1])
        cnt_best[results[0][0]] += 1
        cnt_worst[results[-1][0]] += 1
        for res in results:
            all_successful_time_sum[res[0]] += res[1]
    print("Time sum:")
    print(time_sum)
    print("All successful Time sum:")
    print(all_successful_time_sum)
    print("Best:")
    print(cnt_best)
    print("Worst:")
    print(cnt_worst)
    print("Failed:")
    print(cnt_failed)
    print("Unknown:")
    print(cnt_unknown)


# class testResult
class TestResult:
    def __init__(self, solver, time, result):
        self.solver = solver
        self.time = time
        self.result = result

    #     constructor from list
    @classmethod
    def from_list(cls, l):
        return cls(l[0], l[1], l[2])


def show_graph(dirname: str):
    global solver
    compare_to_z3 = {}

    # # load np arrays
    for solver in solvers:
        path = f'np-comp/{dirname}/{solver}.npy'
        compare_to_z3[solver] = np.load(path)

    for solver in solvers:
        # skip z3, bitwuzla
        if solver == "Z3" or solver == "Bitwuzla":
            continue
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


def read_and_save(filename: str):
    global data, test_results, results, test_name, solver, compare_to_z3, res
    datas = [pd.read_csv(f"{filename}0.csv"), pd.read_csv(f"{filename}1.csv"), pd.read_csv(f"{filename}2.csv")]
    # [dict of test_name -> (dict solver_name -> TestResult(solver_name, time, result))]
    results = [get_results_dict(data) for data in datas]
    compare_to_z3 = {}
    for solver in solvers:
        compare_to_z3[solver] = {}
        for i in range(3):
            compare_to_z3[solver][i] = np.array([])

    # calculate relative difference with Z3 for every test
    # for i, ith_results in enumerate(results):
    for test_name, test_results in results[0].items():
        for test_results in results:
            if test_name not in test_results:
                continue


        z3_res: TestResult = test_results['Z3']

        if z3_res.result != "passed":
            continue
        for solver, res in test_results.items():
            if res.result != "passed":
                continue
            compare_to_z3[res.solver][i] = np.append(compare_to_z3[res.solver][i], res.time / z3_res.time)


    for solver in solvers:
        # filter if not all tests are successful or difference is too large

        # [np array]
        results = compare_to_z3[solver]
        if len(results) != 3:
            print(f"Skipping {solver} because not all tests are successful")
            continue
        # filter large variance
        if variance(results) > 0.25:
            print(f"Skipping {solver} because variance is too large")
            continue

        # compare_to_z3[i][solver].sort()
        os.makedirs(f'np-comp/{filename}', exist_ok=True)
        np.save(f'np-comp/{filename}/{solver}', mean(compare_to_z3[solver]))


def get_results_dict(data: pd.DataFrame) -> dict[str, dict[str, TestResult]]:
    results = {}
    global test_name, solver
    for row in data.itertuples(index=False):
        test_name = row[0]
        results.setdefault(test_name, dict())
        solver = row[1]
        results[test_name][solver] = TestResult.from_list(row[1:])
    return results


if __name__ == '__main__':
    filenames = [
        # "data/merged/data0", "data/merged/data1", "data/merged/data2",
        # "data/all/data0", "data/all/data1", "data/all/data2",
        "data/all/data",
        # "data/no-bv-opt/data0", "data/no-bv-opt/data1","data/no-bv-opt/data2"
    ]
    # if running several times comment this line:
    # for filename in filenames:
    #     # html_to_csv(filename)
    #     read_and_save(filename)
    #     show_graph(filename)
    read_and_save(filenames[0])
    show_graph(filenames[0])

    # html_to_csv()
    # read_and_save()

    # print_aggregate_stats()

    # show longest tests
    # for test_name, results in test_results.items():
    #
    #     results.sort(key=lambda x: x[1])
    #     if results[-1][1] > 14.5:
    #         print(f"{test_name} {results[-1][1]}: {results[-1][0]}")

    #  compare all solvers to z3
