# ILonVariousFeatrueSpace

The environments:

    pytorch 1.10.1
    numpy 1.21.2
    pandas 1.3.4
    python 3.7.10

To run the main experiments, run the run.py file with the method name, the correspondance with the parameter is listed below.

``` console
    python run.py -method=METHOD_NAME
```

| Method      | Joint | Ord-Joint | LwF | EWC |
|-------------|-------|-----------|-----|-----|
| METHOD_NAME | joint | ord_joint | lwf | ewc |

| Method      | ACL | PNN | MUC-LwF | MUC-EWC | ILEAHE-LwF | ILEAHE-EWC |
|-------------|-----|-----|---------|---------|------------|------------|
| METHOD_NAME | acl | pnn | muc_lwf | muc_ewc | ours_lwf   | ours_ewc   |

To test individual method, run main.py with the following parameters.

``` console
    python main.py
```

Related parameters:

    -method: method to use. 'shared_only', 'shared_only_ewc', 'specific_only', 'ours', 'ours_ewc',
    'muc', 'muc_ewc', 'pnn', 'acl', 'joint', 'ord_joint'

    -data_name: dataset to train on, choices are: 'bank', 'blastchar', 'income', 
    'shoppers', 'shrutime', 'volkert' 

    -distill_frac: correspond to beta_1 in paper.
    -alpha
    -beta: correspond to beta_2 in paper
    -gamma
    -lr
    -no_distill: unable distillation loss or ewc loss
    -no_discrim: unable discriminative loss
    -class_inc: add this when training on volkert dataset


For the parameters used, the ones need specialization are listed in below table. For the parameters or models not mentioned, use the default parameters directly.

For parameters of ILEAHE-LwF and ILEAHE-EWC, the parameters are:

| 　       | distill_frac |           | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.2 | 1         | 0.2    | 1        | 0.2      | 0.1     |
| ILEAHE-EWC | 0.5 | 0.4       | 1      | 0.6      | 1        | 1       |

| 　       | alpha                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.1 | 0.2       | 0.1    | 0.2      | 0.2      | 0.2     |
| ILEAHE-EWC | 0.2 | 0.3       | 0.1    | 0.4      | 0.2      | 0.2     |

| 　       | beta                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.1                    | 0.1       | 0.1    | 0.1      | 2        | 0.1     |
| ILEAHE-EWC | 0.1                    | 0.5       | 0.5    | 2        | 1        | 1       |

| 　       | gamma                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 30 | 5         | 15     | 25       | 30       | 5       |
| ILEAHE-EWC | 15                     | 10        | 25     | 25       | 15       | 10      |

For parameters of LwF and EWC:

|     | distill_frac |           |        |          |          |         |
|-----|----------------------|-----------|--------|----------|----------|---------|
|     | bank                 | blastchar | income | shoppers | shrutime | volkert |
| LwF | 1                    | 0.2       | 1      | 1        | 0.2      | 1       |
| EWC | 1                    | 2         | 1      | 1        | 1        | 1       |

|     | T                    |           |        |          |          |         |
|----------|------------------------|-----------|--------|----------|----------|---------|
|     | bank                 | blastchar | income | shoppers | shrutime | volkert |
| LwF | 4                    | 2         | 2      | 4        | 2        | 2       |

For parameters of Joint:
|       | lr     |           |        |          |          |         |
|-------|--------|-----------|--------|----------|----------|---------|
|       | bank   | blastchar | income | shoppers | shrutime | volkert |
| joint | 0.0001 | 0.0001    | 0.0001 | 0.0001   | 0.0001   | 0.0005  |

For parameters of MUC-LwF:

|         | distill_frac |           |        |          |          |         |
|---------|--------------------|-----------|--------|----------|----------|---------|
|         | bank | blastchar | income | shoppers | shrutime | volkert |
| MUC-LwF | 1                  | 1         | 0.005  | 0.1      | 0.5      | 1       |

|         | T                  |           |        |          |          |         |
|---------|--------------------|-----------|--------|----------|----------|---------|
|         | bank               | blastchar | income | shoppers | shrutime | volkert |
| MUC-LwF | 2                  | 4         | 4      | 4        | 2        | 2       |


