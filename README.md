# ILonVariousFeatrueSpace

The environments:

    pytorch 1.10.1
    numpy 1.21.2
    pandas 1.3.4
    python 3.7.10

related parameters:

    -method: method to use. 'shared_only', 'shared_only_ewc', 'specific_only', 'ours', 'ours_ewc',
    'muc', 'muc_ewc', 'pnn', 'acl', 'joint', 'ord_joint'

    -data_name: dataset to train on, choices are: 'bank', 'blastchar', 'income', 
    'shoppers', 'shrutime', 'volkert' 

    -distill_frac: correspond to beta_1 in paper.
    -alpha
    -beta_2
    -gamma
    -lr
    -no_distill: unable distillation loss or ewc loss
    -no_discrim: unable discriminative loss
    -class_inc: add this when training on volkert dataset

To run the code, call the main.py file with the above parameters.

``` console
    python main.py
```

For the parameters used, the ones need specialization are listed in below table. For the parameters or models not mentioned, use the default parameters directly.

For parameters of Ours-LwF and Ours-EWC, the parameters are:

| 　       | distill_frac   / beta1 |           | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| Ours-LwF | 0.2                    | 1         | 0.2    | 1        | 0.2      | 0.1     |
| Ours-EWC | 0.5                    | 0.4       | 1      | 0.6      | 1        | 1       |

| 　       | alpha                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| Ours-LwF | 0.1                    | 0.2       | 0.1    | 0.2      | 0.2      | 0.2     |
| Ours-EWC | 0.2                    | 0.3       | 0.1    | 0.4      | 0.2      | 0.2     |

| 　       | beta2                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| Ours-LwF | 0.1                    | 0.1       | 0.1    | 0.1      | 2        | 0.1     |
| Ours-EWC | 0.1                    | 0.5       | 0.5    | 2        | 1        | 1       |

| 　       | gamma                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| Ours-LwF | 30                     | 5         | 15     | 25       | 30       | 5       |
| Ours-EWC | 15                     | 10        | 25     | 25       | 15       | 10      |

For parameters of LwF and EWC:

|     | distill_frac / beta1 |           |        |          |          |         |
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

|         | distill_frac/beta1 |           |        |          |          |         |
|---------|--------------------|-----------|--------|----------|----------|---------|
|         | bank               | blastchar | income | shoppers | shrutime | volkert |
| MUC-LwF | 1                  | 1         | 0.005  | 0.1      | 0.5      | 1       |

|         | T                  |           |        |          |          |         |
|---------|--------------------|-----------|--------|----------|----------|---------|
|         | bank               | blastchar | income | shoppers | shrutime | volkert |
| MUC-LwF | 2                  | 4         | 4      | 4        | 2        | 2       |


