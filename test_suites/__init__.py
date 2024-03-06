from test_suites.macroNAS import MacroNASBenchmark
from test_suites.nb201 import OurNASBench201Benchmark
from test_suites.nb101 import OurNASBench101Benchmark
from test_suites.zc_nb201 import OurZCNASBench201Benchmark
from test_suites.zc_nb101 import OurZCNASBench101Benchmark

def mop(problem_id):
    if problem_id == 1:
        return MacroNASBenchmark(
            objs='err&params', dataset='cifar10', normalized_objectives=False)
    elif problem_id == 2:
        return OurNASBench101Benchmark(
            objs='err&params', dataset='cifar10', normalized_objectives=False)
    elif problem_id == 3:
        return OurNASBench201Benchmark(
            objs='err&params', dataset='cifar10', normalized_objectives=False)
    elif problem_id == 4:
        return OurNASBench201Benchmark(
            objs='err&params', dataset='ImageNet16-120', normalized_objectives=False)
    elif problem_id == 5:
        return OurNASBench201Benchmark(
            objs='err&params&latency', dataset='cifar10', normalized_objectives=False)
    elif problem_id == 6:
        return OurNASBench201Benchmark(
            objs='err&params&latency', dataset='ImageNet16-120', normalized_objectives=False)
    elif problem_id == 7:
        return OurZCNASBench201Benchmark(
            objs='err&params', dataset='cifar10', normalized_objectives=False)
    elif problem_id == 8:
        return OurZCNASBench101Benchmark(
            objs='err&params', dataset='cifar10', normalized_objectives=False)
    else:
        raise ValueError("the requested problem id does not exist")