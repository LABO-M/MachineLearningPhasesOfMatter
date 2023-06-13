import numpy as np
import matplotlib.pyplot as plt

# 磁化率の計算
def compute_magnetization(lattice):
    N  = lattice.shape[0] * lattice.shape[1]
    magnetization = np.sum(lattice) / N
    return magnetization

'''
隣接サイトの情報を取得し総和を取る，row:行，col:列
正方格子に関してはこれで大丈夫だが，三角格子などの場合は
さらに工夫が必要
'''
def compute_neighbor_sum(lattice, row, col):
    n = lattice.shape[0]  # 正方格子の一辺の長さ
    neighbor_sum = 0
    
    # 上方向の隣接要素の総和
    if row > 0:
        neighbor_sum += lattice[row-1, col]
    
    # 下方向の隣接要素の総和
    if row < n-1:
        neighbor_sum += lattice[row+1, col]
    
    # 左方向の隣接要素の総和
    if col > 0:
        neighbor_sum += lattice[row, col-1]
    
    # 右方向の隣接要素の総和
    if col < n-1:
        neighbor_sum += lattice[row, col+1]
    
    return neighbor_sum

# 2次元正方格子の作成をする関数，引数nは格子のサイズ
def create_lattice(n):
    lattice = np.random.choice([-1, 1], size=(n, n))
    return lattice

# 近傍の情報からボルツマン因子を計算する関数
def compute_boltzumann_factor(lattice, row, col, beta , J , h):
    neighbor_sum = compute_neighbor_sum(lattice, row, col)
    p = np.exp( beta *( J * neighbor_sum + h))
    m = np.exp( - beta *( J * neighbor_sum + h))
    return p , m

# 条件付き確率の計算し，その条件から次のステップの状態を選択する関数
def compute_conditional_probability(lattice, row, col, beta , J , h):
    p , m = compute_boltzumann_factor(lattice, row, col, beta , J , h)
    p , m = p / (p + m ) , m / (p + m )
    s = np.random.choice([1, -1], p=[p, m])
    return s , p , m

# 得られた未来の情報から配列に反映させる関数
def update_lattice(lattice, s, row, col):
    lattice[row, col] = s
    return lattice

# 一つのステップを行う関数
def one_step(lattice, beta , J , h):
    n1 = lattice.shape[0]
    n2 = lattice.shape[1]
    for row in range(n1):
        for col in range(n2):
            s , p , m = compute_conditional_probability(lattice, row, col, beta , J , h)
            lattice = update_lattice(lattice, s, row, col)
    return lattice

# 繰り返し格子の更新を行い磁化率の推移を見る関数
def compute_magnetization_transition(lattice, beta , J , h, n_steps):
    magnetization_transition = []
    for i in range(n_steps):
        lattice = one_step(lattice, beta , J , h)
        m = compute_magnetization(lattice)
        magnetization_transition.append(m)
    return magnetization_transition




