# ILonVariousFeatrueSpace


## **Running the code**
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

| Method      | Joint | Ord-Joint | LwF | EWC | PCL | AFEC | DMC |
|-------------|-------|-----------|-----|-----|-----|------|-----|
| METHOD_NAME | joint | ord_joint | lwf | ewc | pcl | afec | dmc |

| Method      | ACL | PNN | MUC-LwF | MUC-EWC | ILEAHE-LwF | ILEAHE-EWC |
|-------------|-----|-----|---------|---------|------------|------------|
| METHOD_NAME | acl | pnn | muc_lwf | muc_ewc | ours_lwf   | ours_ewc   |

To test individual method, run main.py with the following parameters.

``` console
    python main.py
```

Related parameters:

    -method: method to use. 'lwf', 'ewc', 'ours_lwf', 'ours_ewc',
    'muc_lwf', 'muc_ewc', 'pnn', 'acl', 'joint', 'ord_joint', 'pcl', 'dmc, 'afec'

    -data_name: dataset to train on, choices are: 'bank', 'blastchar', 'income', 
    'shoppers', 'shrutime', 'volkert' 

    -T: hyperparameter T of Distillation Loss from LwF
    -distill_frac: correspond to beta_1 in paper.
    -alpha
    -beta: correspond to beta_2 in paper
    -gamma
    -lr: learning rate
    -class_inc: add this when training on volkert dataset

    -order: choose the order for learning the data sets, values are 1, 2, 3
    -rand: choose the random seed for selecting attribtues, values from 1 to 10
    
    -extractor_type: choose the structure type of extractors, transformer for Transformer structure, rnn for RNN structure and mlp for MLP structure.

## **Full Results of Figures in Paper**

The full results for effectiveness-efficiency study, corresponds to Sec. 4.4 and Fig. 6 in the paper.
![Effectiveness-Efficiency Study](/figures/eff_eff.png "Effectiveness-Efficiency Study")

The full results for ILEAHE, Joint and strong baselines under 3 different orders, corresponds to Sec. 4.3.1 and Fig. 5(a).
![Results under 3 order](/figures/order.png "Results under 3 order")

The full results for ILEAHE, Joint and strong baselines under 10 sets of randomly selected attribtues, corresponds to Sec. 4.3.2 and Fig. 5(b).
![Results under 10 sets of attribtues](/figures/rand.png "Results under 10 sets of attribtues")

## **Hyperparameters**

### **Selected hyperparameters**

For the parameters used, the ones need specialization are listed in below tables. While for the parameters or models not mentioned, use the default parameters directly.
We selected hyper-parameters using **hyperopt** package.

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
|----------|------|-----------|--------|----------|----------|---------|
| 　       | bank | blastchar | income | shoppers | shrutime | volkert |
| ILEAHE-LwF | 30 | 5         | 15     | 5       | 30       | 5       |
| ILEAHE-EWC | 15 | 10        | 25     | 25       | 15       | 10      |

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

### **Parameter Search Space**

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

PCL

    alpha: {0.01, 0.1, 1, 10, 100}
    beta: {0.01, 0.1, 1, 10, 100}

AFEC

    alpha: {0.01, 0.1, 1, 10, 100}
    distill_frac: {0.1, 0.5, 1, 2}

Hyperparameters for ACL are adopted from original implementation, PNN doesn't include hyperparameters other than learning rate.

### **Parameter Selection Explanation**

**T** is for distillation loss and to make the difference between soft prediction possibilities less evident, and larger value leads to smaller difference. In practice, 2 or 4 are preferred.

**Alpha** controls the weight of shared prediction in final prediction. Because of the forgetting issue, shared prediction is generally less effective than specific predictions, so the selected alphas are smaller than 0.5. On the other hand, if alpha is too small like 0.1, which makes final prediction equivalent to specific prediction, the AAUC begins to drop. This is due to the lost of benefit from assemble prediction.

**Gamma** lifts the impact of discriminability score on the discriminative loss. Higher gamma makes larger punishment on smaller discriminability scores. The value is subjective to data sets.

**Beta1 and Beta2** controls the effects of regularization loss and discriminative loss. Especially, Beta1 is preferred to be smaller. Otherwise the model would bias towards old datasets.


### **Hyperparameter Sensitivity Analysis**

The analysis is for ILEAHE-EWC, the model is robust to different hyper-parameters. Parameters related to the discriminative loss will affect more about the performance, as the effectiveness of specific features are subject to it.

The performances (AAUC and std.) of changing alpha/beta1/beta2/gamma for ILEAHE-EWC

| alpha | 0.1    | 0.2    | 0.3    | 0.4    | 0.5    |        |
|-------|--------|--------|--------|--------|--------|--------|
| AAUC  | 0.8161 | 0.8164 | 0.8166 | 0.8166 | 0.8166 |        |
| std   | 0.0029 | 0.0030 | 0.0032 | 0.0034 | 0.0036 |        |

| Beta1 | 0.0005 | 0.1    | 0.2    | 0.5    | 1      | 2      |
|-------|--------|--------|--------|--------|--------|--------|
| AAUC  | 0.8157 | 0.8161 | 0.8165 | 0.8175 | 0.8164 | 0.8182 |
| std   | 0.0026 | 0.0031 | 0.0028 | 0.0016 | 0.0031 | 0.0024 |

| Beta2 | 0.1    | 0.5    | 1      | 2      | 5      |        |
|-------|--------|--------|--------|--------|--------|--------|
| AAUC  | 0.8141 | 0.8174 | 0.8164 | 0.8154 | 0.8154 |        |
| std   | 0.0035 | 0.0021 | 0.0031 | 0.0027 | 0.0037 |        |

| Gamma | 5      | 10     | 15     | 20     | 25     | 30     |
|-------|--------|--------|--------|--------|--------|--------|
| AAUC  | 0.8164 | 0.8152 | 0.8164 | 0.817  | 0.8174 | 0.817  |
| std   | 0.0027 | 0.0029 | 0.0031 | 0.0023 | 0.0043 | 0.0023 |



## **Data Set Info**


|     Name             |     Data Amount    |     Cate Feat    |     Num Feat    |     Classes    |     Pos Rate (%)    |     Link                                                                                    |
|----------------------|--------------------|------------------|-----------------|----------------|---------------------|---------------------------------------------------------------------------------------------|
|     Bank    |     45211          |     9            |     7           |     2          |     11.70           |     https://www.openml.org/search?type=data&sort=runs&id=1461&status=active                 |
|     Shoppers         |     12330          |     2            |     15          |     2          |     15.47           |     https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset    |
|     Income           |     32561          |     8            |     6           |     2          |     24.08           |     https://www.kaggle.com/lodetomasi1995/income-classification                             |
|     BlastChar        |     7043           |     17           |     3           |     2          |     26.54           |     https://www.kaggle.com/datasets/blastchar/telco-customer-churn                          |
|     Shrutime         |     10000          |     4            |     6           |     2          |     20.37           |     https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling                         |
|     Volkert          |     58310          |     0            |     147         |     10         |     NA              |     https://www.openml.org/search?type=data&sort=runs&id=41166                              |

Use bank-full.csv for Bank data set.

