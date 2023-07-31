import numpy as np

# カーネル関数
def kernel(x1, x2, theta):
    return theta[0] * np.exp(-theta[1] * np.linalg.norm(x1 - x2)**2)

# グラム行列（カーネル行列）を計算
def compute_gram_matrix(X, theta):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], theta)
    return K

# ガウス過程からサンプルを生成
def sample_gp(X, theta, num_samples):
    # グラム行列を計算
    K = compute_gram_matrix(X, theta)

    # ガウス過程からサンプルを生成
    samples = np.random.multivariate_normal(np.zeros(len(X)), K, num_samples)
    
    return samples