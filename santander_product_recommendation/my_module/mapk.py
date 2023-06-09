# MAP@7の評価基準を求めるコード
import numpy as np

# average Precision k
def apk(actual, predicted, k=7, default=0.0):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    
    # 正答が空白の場合0点
    if not actual:
        return default
    
    return score / min(len(actual), k)

def mapk(actual, predicted, k=7, default=0.0):
    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicted)])

if __name__ == '__main__':
    pass