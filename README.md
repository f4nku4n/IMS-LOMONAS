# IMS-LOMONAS: Parameter-less Pareto Local Search for Multi-objective Neural Architecture Search with the Interleaved Multi-start Scheme
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong

## Setup
- Clone this repo
- Install necessary packages and databases.
```
$ cd IMS-LOMONAS
$ bash install.sh
```
- Download [data](https://drive.google.com/drive/folders/15Ux-FSRjfu8dPxFZ2B1yCOPEBb-QHkj4?usp=drive_link) and [database](https://drive.google.com/drive/folders/191iUhps8j1I3xg6AwNH1EIeYHtCmNBE3?usp=drive_link) for the set of problems in the [CEC'2023 Competition](https://www.emigroup.tech/index.php/news/ieee-cec2023-competition-on-multiobjective-neural-architecture-search/).
- Run the below cell to load the data
```
from evoxbench.database.init import config
config(<database_path>, <data_path>)
```
For example:
```
from evoxbench.database.init import config
config('/content/drive/database', '/content/drive/data')
```
## Reproducing the results
This repo have already implemented following NAS algorithms:
- **(IMS-)NSGA-II**
- **(IMS-)NSGA-III**
- **(IMS-)LOMONAS** (ours)

### CEC'2023 problems
To experiment on CEC'2023 problems, run the below script:
```shell
$ python main.py --optimizer <algo_name>[lomonas, ims-lomonas, nsga2, nsga3, ims-nsga2, ims-nsga3]
                 --test_suite <problem_name>[cec-c10, cec-in1k]
                 --pid <problem_id>[from 1 to 9]
                 --max_eval 10000 --n_run 31
                 --database_path <CEC_database_path> --data_path <CEC_data_path>
                 --using_archive --check_limited_neighbors --neighborhood_check_on_potential_sols --log_results
```

### Other problems
To experiment on other problems, run the below script:
```shell
$ python main.py --optimizer <algo_name>[lomonas, ims-lomonas]
                 --test_suite <problem_name>[gecco]
                 --pid <problem_id>[from 1 to 8]
                 --max_eval 3000 --n_run 31
                 --database_path <CEC_database_path> --data_path <CEC_data_path>
                 --using_archive --check_limited_neighbors --neighborhood_check_on_potential_sols --log_results
```
where:
| Problem ID                   | Search Space | Dataset | Target Objectives            |  Search Objectives   |            
|:--------------------------:|:----------------------:|:----------------------:|:--:|:---------------------------------------:|
|1          | MacroNAS | CIFAR-10 | test_err & params | val_err & params |
|2        | NAS-Bench-101 | CIFAR-10 | test_err & params | val_err_12 & params |
|3          | NAS-Bench-201 | CIFAR-10 | test_err & params | val_err_12 & params |
|4         | NAS-Bench-201 | ImageNet16-120 | test_err & params | val_err_12 & params |
|5         | NAS-Bench-201 | CIFAR-10 | test_err & params | val_err_12 & params |
|6         | NAS-Bench-201 | ImageNet16-120 | test_err & params | val_err_12 & params |
|7         | NAS-Bench-201 | CIFAR-10 | test_err & params | synflow & jacov & params|
|8         | NAS-Bench-101 | CIFAR-10 | test_err & params | synflow & jacov & params|

Set `pid` to 7 and 8 to experiment TF-(IMS-)LOMONAS.

## Evaluation (only for NAS-Bench-101 and NAS-Bench-201 (GECCO))
```shell
$ python evaluate.py --res_path <result_path>
                     --problem_id <problem_id>
                     --dataset [cifar10, cifar100, ImageNet16-120]
                     --algo_name [lomonas, ims-lomonas]
```
## Acknowledgement
We want to give our thanks to the authors of [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), and [NAS-Bench-ASR](https://openreview.net/forum?id=CU0APx9LMaL) for their search spaces; to the authors of [Zero-cost Lightweight NAS](https://openreview.net/pdf?id=0cmMMy8J5q) and [NAS-Bench-Zero-Suite](https://openreview.net/pdf?id=yWhuIjIjH8k) for their zero-cost metric databases.