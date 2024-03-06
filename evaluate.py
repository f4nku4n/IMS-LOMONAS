import argparse
import pickle as p
from utilities import get_project_root

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def greedy_selection(front, n_survive, **kwargs):
    F = front.copy()
    scaler = MinMaxScaler()
    F = scaler.fit_transform(F)
    front = [{'fitness': [f], 'index': i} for i, f in enumerate(F)]
    selected_solutions = []

    front = list(front)
    front = sorted(front, key=lambda x: -x['fitness'][-1][0])

    cur_selected_indices = set([])

    selected_solutions.append(front[0])
    cur_selected_indices.add(0)
    while len(selected_solutions) < n_survive:
        points1 = np.array([x['fitness'][-1] for x in front])
        points2 = np.array([x['fitness'][-1] for x in selected_solutions])

        distances = euclidean_distances(points1, points2)
        cur_min_distances = np.min(distances, axis=1)

        ind_with_max_dist = -1
        max_dist = -float("inf")
        for j in range(len(front)):
            if j not in cur_selected_indices and cur_min_distances[j] > max_dist:
                max_dist = cur_min_distances[j]
                ind_with_max_dist = j
        selected_solutions.append(front[ind_with_max_dist])
        cur_selected_indices.add(ind_with_max_dist)
    I = np.unique([s['index'] for s in selected_solutions])
    return I

def evaluate(res_path, problem_id='1', algo_name='lomonas', dataset='cifar10'):
    root = get_project_root()
    if problem_id in [2, 8]:
        api = p.load(open(f'{root}/database/NASBench101/data.p', 'rb'))
    else:
        if dataset == 'cifar10':
            api = p.load(open(f'{root}/database/NASBench201/[CIFAR-10]_data.p', 'rb'))
        elif dataset == 'cifar100':
            api = p.load(open(f'{root}/database/NASBench201/[CIFAR-100]_data.p', 'rb'))
        else:
            api = p.load(open(f'{root}/database/NASBench201/[ImageNet16-120]_data.p', 'rb'))
    evaluation_cost = []
    best_err = []
    # fronts = []
    for i in range(31):
        archive = p.load(open(f'{res_path}/gecco-mop{problem_id}_run{i}_{algo_name}_archive.p', 'rb'))
        if problem_id in [2, 8]:
            X, idx = np.unique(archive['hashKey'], return_index=True)
        else:
            X, idx = np.unique(archive['X'], axis=0, return_index=True)
        F = archive['F'][idx]

        idx = greedy_selection(F, 20)
        X = np.array(X)[idx]

        _evaluation_cost = 0.0
        evaluation_front = []
        for arch in X:
            if problem_id in [2, 8]:
                test_error_rate = (1 - api['108'][arch]['test_acc']) * 100
                params = api['108'][arch]['n_params']
                evaluation_front.append([test_error_rate, params])
                _evaluation_cost += api['108'][arch]['train_time']
            else:
                hashKey = ''.join(map(str, arch))
                test_error_rate = (1 - api['200'][hashKey]['test_acc'][-1]) * 100
                params = api['200'][hashKey]['params']
                evaluation_front.append([test_error_rate, params])
                train_time = api['200'][hashKey]['train_time'] * 200
                if dataset == 'cifar10':
                    train_time /= 2
                _evaluation_cost += train_time
        evaluation_front = np.round(evaluation_front, 6)
        evaluation_front = np.array(evaluation_front)

        best_err.append(np.min(evaluation_front[:, 0]))
        evaluation_cost.append(_evaluation_cost)
    best_acc = 100 - np.array(best_err)
    print(f'Best Accuracy: {np.round(np.mean(best_acc), 2)} ({np.round(np.std(best_acc), 2)})')
    print('Evaluation Cost:', int(np.mean(evaluation_cost)))

def main(kwargs):
    evaluate(kwargs.res_path, kwargs.problem_id, kwargs.algo_name, kwargs.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--res_path', type=str, default='./res')
    parser.add_argument('--problem_id', type=int, default=1)

    parser.add_argument('--algo_name', type=str, default='lomonas', choices=['lomonas', 'ims-lomonas'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'])
    args = parser.parse_args()

    main(args)
