# Adversarial Personalized Ranking for Recommendation in Pytorch Lightning

This is my implementation for the paper:
Xiangnan He, Zhankui He, Xiaoyu Du & Tat-Seng Chua. 2018. **Adversarial Personalized Ranking for Recommendation**  

## Demo
The results in Yelp dataset, I train BPR on 50 epochs and robust APR on 25 epochs.

![figure](figure\validation.png)

#### Runnig
I created a demo using Yelp dataset, embedding size 64:
1. Training with Bayessian Personalized Ranking:
```shell
python train.py --config config/BPR.yaml
```
2. Robusting with Adversarial Personalized Ranking:
```shell
python train.py --config config/APR.yaml --pretrained exp/best_BPR.ckpt
```

## Dataset
I obtained datasets:  Yelp(yelp), MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20) from the original Git repository: https://github.com/hexiangnan/adversarial_personalized_ranking

**train.rating:**

- Train file.


- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

**test.rating:**

- Test file (positive instances).
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

**test.negative**

- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

## Notebook version:
adversarial-personalization-ranking-apr-pytorch.ipynb
https://www.kaggle.com/code/minhtutx/adversarial-personalization-ranking-apr-pytorch