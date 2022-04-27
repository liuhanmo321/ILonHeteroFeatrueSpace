# ILonVariousFeatrueSpace

The environments:

    pytorch 1.10.1
    numpy 1.21.2
    pandas 1.3.4
    python 3.7.10

To run the main experiments:

    run data_prep.py to prepare data from the data folder.
    run the run.py file with the method name, the corresponding model name is listed below. GPU_ID is the gpu id used to run the code, for example "cuda:1" means -gpu=1.

``` console
    python data_prep.py
    python run.py -method=METHOD_NAME -gpu=GPU_ID
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

    -method: method to use. 'lwf', 'ewc', 'ours_lwf', 'ours_ewc',
    'muc_lwf', 'muc_ewc', 'pnn', 'acl', 'joint', 'ord_joint'

    -data_name: dataset to train on, choices are: 'bank', 'blastchar', 'income', 
    'shoppers', 'shrutime', 'volkert' 

    -T: hyperparameter T of Distillation Loss from LwF
    -distill_frac: correspond to beta_1 in paper.
    -alpha
    -beta: correspond to beta_2 in paper
    -gamma
    -lr: learning rate
    -class_inc: add this when training on volkert dataset


For the parameters used, the ones need specialization are listed in below tables. While for the parameters or models not mentioned, use the default parameters directly.

For parameters of ILEAHE-LwF and ILEAHE-EWC, the parameters are:

| 　       | distill_frac |           | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.2 | 1         | 0.2    | 1        | 0.2      | 0.1     |
| ILEAHE-EWC | 0.5 | 0.5       | 1      | 0.5      | 1        | 1       |

| 　       | alpha                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.1 | 0.4       | 0.1    | 0.2      | 0.2      | 0.2     |
| ILEAHE-EWC | 0.2 | 0.3       | 0.1    | 0.2      | 0.2      | 0.2     |

| 　       | beta                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank                   | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 0.1                    | 2       | 0.1    | 0.5      | 2        | 0.1     |
| ILEAHE-EWC | 0.1                    | 0.5       | 0.5    | 2        | 1        | 1       |

| 　       | gamma                  | 　        | 　     | 　       | 　       | 　      |
|----------|------------------------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 30 | 5         | 15     | 5       | 30       | 5       |
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

**Parameter Search Space**

Because we decrease learning rate during training, it is not searched for methods other than Joint and Ord-Joint, which are sensitive to learning rates.

ILEAHE-LwF and ILEAHE-EWC

    T: {0.5, 1, 2, 4} (For ILEAHE-LwF)
    distill_frac: {0.005, 0.1, 0.2, 0.5, 1, 2}
    alpha: {0.1, 0.2, 0.3, 0.4, 0.5}    
    beta: {0.1, 0.5, 1, 2, 5}
    gamma: {5, 10, 15, 20, 25, 30}

Ord-Joint and Joint

    lr: {0.0001, 0.0005}

LwF

    T: {0.5, 1, 2, 4}
    distill_frac: {0.005, 0.1, 0.2, 0.5, 1, 2}

EWC

    distill_frac: {0.005, 0.1, 0.2, 0.5, 1, 2}

MUC-LwF and MUC-EWC

    T: {0.5, 1, 2, 4} (For MUC-LwF)
    distill_frac: {0.005, 0.1, 0.2, 0.5, 1, 2}

Hyperparameters for ACL are adopted from original implementation, PNN doesn't include hyperparameters other than learning rate.


**Data Set Info**


|     Name             |     Data Amount    |     Cate Feat    |     Num Feat    |     Classes    |     Pos Rate (%)    |     Link                                                                                    |
|----------------------|--------------------|------------------|-----------------|----------------|---------------------|---------------------------------------------------------------------------------------------|
|     Bank    |     45211          |     9            |     7           |     2          |     11.70           |     https://www.openml.org/search?type=data&sort=runs&id=1461&status=active                 |
|     Shoppers         |     12330          |     2            |     15          |     2          |     15.47           |     https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset    |
|     Income           |     32561          |     8            |     6           |     2          |     24.08           |     https://www.kaggle.com/lodetomasi1995/income-classification                             |
|     BlastChar        |     7043           |     17           |     3           |     2          |     26.54           |     https://www.kaggle.com/datasets/blastchar/telco-customer-churn                          |
|     Shrutime         |     10000          |     4            |     6           |     2          |     20.37           |     https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling                         |
|     Volkert          |     58310          |     0            |     147         |     10         |     NA              |     https://www.openml.org/search?type=data&sort=runs&id=41166                              |

Use bank-full.csv for Bank data set.