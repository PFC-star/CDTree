import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import entropy, ks_2samp
from torchvision import models
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

def gaussian_kernel(x, y, sigma=1.0):
    """计算高斯核函数的高效实现"""
    # 使用广播和矩阵运算来避免大矩阵
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # 计算平方差
    xx = np.sum(x**2, axis=1, keepdims=True)
    yy = np.sum(y**2, axis=1, keepdims=True)
    xy = np.dot(x, y.T)
    
    # 计算高斯核
    dists = xx + yy.T - 2*xy
    return np.exp(-dists / (2 * sigma**2))

def compute_mmd(x, y, sigma=1.0):
    """计算MMD（Maximum Mean Discrepancy）距离的高效实现"""
    # 随机采样以减少计算量
    max_samples = 1000
    if len(x) > max_samples:
        x_indices = np.random.choice(len(x), max_samples, replace=False)
        y_indices = np.random.choice(len(y), max_samples, replace=False)
        x = x[x_indices]
        y = y[y_indices]
    
    # 计算核矩阵
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    
    # 计算MMD
    mmd = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
    return np.sqrt(mmd)

def compute_cdf_distance(x, y, num_points=100):
    """计算CDF（累积分布函数）距离"""
    # 计算经验CDF
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # 生成均匀分布的点
    points = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), num_points)
    
    # 计算CDF
    x_cdf = np.searchsorted(x_sorted, points) / len(x)
    y_cdf = np.searchsorted(y_sorted, points) / len(y)
    
    # 计算CDF距离
    cdf_distance = np.mean(np.abs(x_cdf - y_cdf))
    return cdf_distance

def compute_domain_statistics(data_loader, device='cuda'):
    """计算数据域的基本统计特征
    
    Args:
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        dict: 包含均值、标准差等统计信息的字典
    """
    mean = torch.zeros(1).to(device)
    std = torch.zeros(1).to(device)
    total_samples = 0
    
    for data in data_loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(device)
        batch_size = data.size(0)
        data = data.float()
        
        mean += data.mean().item() * batch_size
        std += data.std().item() * batch_size
        total_samples += batch_size
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': float(mean.cpu().numpy()[0]),  # 转换为标量
        'std': float(std.cpu().numpy()[0])     # 转换为标量
    }

def compute_domain_difference(old_stats, new_stats):
    """计算两个域之间的统计差异
    
    Args:
        old_stats: 旧域的统计信息
        new_stats: 新域的统计信息
    
    Returns:
        dict: 包含各种差异度量的字典
    """
    mean_diff = np.abs(old_stats['mean'] - new_stats['mean'])
    std_diff = np.abs(old_stats['std'] - new_stats['std'])
    
    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'total_difference': np.mean(mean_diff) + np.mean(std_diff)
    }

def visualize_domain_distribution(old_data, new_data, save_path=None):
    """可视化两个域的数据分布
    
    Args:
        old_data: 旧域数据
        new_data: 新域数据
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制直方图
    plt.subplot(2, 2, 1)
    plt.hist(old_data, bins=100, alpha=0.5, label='Old Domain', density=True)
    plt.hist(new_data, bins=100, alpha=0.5, label='New Domain', density=True)
    plt.legend()
    plt.title('原始数据分布')
    
    # 绘制箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot([old_data, new_data], labels=['Old Domain', 'New Domain'])
    plt.title('数据分布箱线图')
    
    # 绘制CDF对比图
    plt.subplot(2, 2, 3)
    x_sorted = np.sort(old_data)
    y_sorted = np.sort(new_data)
    plt.plot(x_sorted, np.arange(len(x_sorted))/len(x_sorted), label='Old Domain')
    plt.plot(y_sorted, np.arange(len(y_sorted))/len(y_sorted), label='New Domain')
    plt.legend()
    plt.title('累积分布函数对比')
    
    # 绘制Q-Q图
    plt.subplot(2, 2, 4)
    quantiles = np.linspace(0, 1, 100)
    old_quantiles = np.percentile(old_data, quantiles * 100)
    new_quantiles = np.percentile(new_data, quantiles * 100)
    plt.plot(old_quantiles, new_quantiles, 'b.')
    plt.plot([old_quantiles[0], old_quantiles[-1]], [old_quantiles[0], old_quantiles[-1]], 'r--')
    plt.title('Q-Q图')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_domain_shift(old_data, new_data):
    """计算域偏移程度
    
    Args:
        old_data: 旧域数据
        new_data: 新域数据
    
    Returns:
        dict: 包含各种域偏移度量的字典
    """
    # 确保数据类型是浮点数
    old_data = old_data.astype(np.float32)
    new_data = new_data.astype(np.float32)
    
    # 数据降维和采样
    # 1. 将3D数据展平为2D
    old_data_flat = old_data.reshape(old_data.shape[0], -1)
    new_data_flat = new_data.reshape(new_data.shape[0], -1)
    
    # 2. 使用PCA降维
    n_components = min(50, old_data_flat.shape[1])  # 最多保留50个主成分
    pca = PCA(n_components=n_components)
    old_data_pca = pca.fit_transform(old_data_flat)
    new_data_pca = pca.transform(new_data_flat)
    
    # 3. 随机采样以减少数据量
    max_samples = 2000  # 减少采样数量
    if old_data_pca.shape[0] > max_samples:
        old_indices = np.random.choice(old_data_pca.shape[0], max_samples, replace=False)
        new_indices = np.random.choice(new_data_pca.shape[0], max_samples, replace=False)
        old_data_pca = old_data_pca[old_indices]
        new_data_pca = new_data_pca[new_indices]
    
    # 打印原始数据信息
    print("\n=== 计算域偏移前的数据信息 ===")
    print(f"旧域均值: {np.mean(old_data_pca):.4f}, 标准差: {np.std(old_data_pca):.4f}")
    print(f"新域均值: {np.mean(new_data_pca):.4f}, 标准差: {np.std(new_data_pca):.4f}")
    
    # 计算相对变化
    mean_change = np.abs(np.mean(new_data_pca) - np.mean(old_data_pca)) / (np.mean(old_data_pca) + 1e-8)
    std_change = np.abs(np.std(new_data_pca) - np.std(old_data_pca)) / (np.std(old_data_pca) + 1e-8)
    
    # 计算更多的分位数点
    quantiles = np.linspace(0.1, 0.9, 9)  # 10%到90%的分位数
    old_quantiles = np.percentile(old_data_pca, quantiles * 100)
    new_quantiles = np.percentile(new_data_pca, quantiles * 100)
    quantile_diff = np.mean(np.abs(old_quantiles - new_quantiles))
    
    # 计算相对分位数差异
    relative_quantile_diff = np.mean(np.abs(old_quantiles - new_quantiles) / (old_quantiles + 1e-8))
    
    # 计算Wasserstein距离
    w_dist = wasserstein_distance(old_data_pca.flatten(), new_data_pca.flatten())
    
    # 计算多个sigma值的MMD距离
    sigmas = [0.1, 1.0, 10.0]
    mmd_distances = [compute_mmd(old_data_pca.flatten(), new_data_pca.flatten(), sigma) for sigma in sigmas]
    
    # 计算CDF距离
    cdf_dist = compute_cdf_distance(old_data_pca.flatten(), new_data_pca.flatten())
    
    # 计算分布重叠度
    hist_old, _ = np.histogram(old_data_pca.flatten(), bins=100, density=True)
    hist_new, _ = np.histogram(new_data_pca.flatten(), bins=100, density=True)
    overlap = np.sum(np.minimum(hist_old, hist_new)) / np.sum(np.maximum(hist_old, hist_new))
    
    # 计算KS检验统计量
    ks_stat, p_value = ks_2samp(old_data_pca.flatten(), new_data_pca.flatten())
    
    # 计算KL散度
    old_hist = np.histogram(old_data_pca.flatten(), bins=100, density=True)[0]
    new_hist = np.histogram(new_data_pca.flatten(), bins=100, density=True)[0]
    old_hist = old_hist / (np.sum(old_hist) + 1e-8)
    new_hist = new_hist / (np.sum(new_hist) + 1e-8)
    kl_div = entropy(old_hist, new_hist)
    
    return {
        'mean_change': mean_change,
        'std_change': std_change,
        'quantile_difference': quantile_diff,
        'relative_quantile_difference': relative_quantile_diff,
        'wasserstein_distance': w_dist,
        'mmd_distances': mmd_distances,
        'cdf_distance': cdf_dist,
        'distribution_overlap': overlap,
        'ks_statistic': ks_stat,
        'ks_pvalue': p_value,
        'kl_divergence': kl_div
    }

def compute_domain_gap(old_data, new_data):
    """计算域差异的细粒度方法
    
    Args:
        old_data: 旧域数据, shape为(N, D)的张量
        new_data: 新域数据, shape为(M, D)的张量
    
    Returns:
        float: 域差异分数(0-1之间,越大表示差异越大)
    """
    # 确保数据是2D张量
    if len(old_data.shape) == 1:
        old_data = old_data.reshape(-1, 1)
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(-1, 1)
    
    # 转换数据类型为浮点型
    old_data = old_data.float()
    new_data = new_data.float()
    
    # 数据归一化
    old_data = (old_data - old_data.mean()) / (old_data.std() + 1e-8)
    new_data = (new_data - new_data.mean()) / (new_data.std() + 1e-8)
    
    # 1. 计算基本统计特征差异
    old_mean = torch.mean(old_data, dim=0)
    old_std = torch.std(old_data, dim=0)
    new_mean = torch.mean(new_data, dim=0)
    new_std = torch.std(new_data, dim=0)
    
    mean_diff = torch.mean(torch.abs(old_mean - new_mean))
    std_diff = torch.mean(torch.abs(old_std - new_std))
    
    # 2. 计算更细致的分布特征差异
    # 使用更多的分位数点
    quantiles = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=old_data.device)
    old_q = torch.quantile(old_data, quantiles, dim=0)
    new_q = torch.quantile(new_data, quantiles, dim=0)
    q_diff = torch.mean(torch.abs(old_q - new_q))
    
    # 3. 计算特征空间距离
    # 使用中心点距离和协方差矩阵差异
    old_center = torch.mean(old_data, dim=0)
    new_center = torch.mean(new_data, dim=0)
    center_dist = torch.norm(old_center - new_center)
    
    # 计算协方差矩阵差异
    old_cov = torch.cov(old_data.T)
    new_cov = torch.cov(new_data.T)
    cov_diff = torch.mean(torch.abs(old_cov - new_cov))
    
    # 4. 计算数据分布重叠度
    # 使用更多的直方图bin
    old_hist = torch.histc(old_data, bins=200, min=float(old_data.min()), max=float(old_data.max()))
    new_hist = torch.histc(new_data, bins=200, min=float(new_data.min()), max=float(new_data.max()))
    overlap = torch.sum(torch.min(old_hist, new_hist)) / torch.sum(torch.max(old_hist, new_hist))
    
    # 5. 计算高阶统计特征差异
    # 偏度
    old_skew = torch.mean(((old_data - old_mean) / (old_std + 1e-8)) ** 3)
    new_skew = torch.mean(((new_data - new_mean) / (new_std + 1e-8)) ** 3)
    skew_diff = torch.abs(old_skew - new_skew)
    
    # 峰度
    old_kurt = torch.mean(((old_data - old_mean) / (old_std + 1e-8)) ** 4)
    new_kurt = torch.mean(((new_data - new_mean) / (new_std + 1e-8)) ** 4)
    kurt_diff = torch.abs(old_kurt - new_kurt)
    
    # 6. 计算局部特征差异
    # 计算局部密度差异
    old_density = torch.histc(old_data, bins=50, min=float(old_data.min()), max=float(old_data.max()))
    new_density = torch.histc(new_data, bins=50, min=float(new_data.min()), max=float(new_data.max()))
    density_diff = torch.mean(torch.abs(old_density - new_density))
    
    # 计算局部方差差异
    old_local_var = torch.var(old_data, dim=0)
    new_local_var = torch.var(new_data, dim=0)
    local_var_diff = torch.mean(torch.abs(old_local_var - new_local_var))
    
    # 7. 计算自适应权重
    # 基于各个指标的变化程度动态调整权重
    base_weights = torch.tensor([0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], device=old_data.device)
    scores = torch.stack([
        mean_diff, std_diff, q_diff, center_dist, cov_diff,
        skew_diff, kurt_diff, density_diff, local_var_diff
    ])
    
    # 归一化各个指标
    scores = scores / (torch.mean(scores) + 1e-8)
    
    # 根据变化程度调整权重
    adaptive_weights = base_weights * (1 + scores)
    adaptive_weights = adaptive_weights / torch.sum(adaptive_weights)
    
    # 8. 计算最终域差异分数
    domain_gap = torch.sum(adaptive_weights * scores)
    
    # 9. 打印详细的差异分析
    print("\n=== 域差异详细分析 ===")
    print(f"均值差异: {mean_diff:.4f}")
    print(f"标准差差异: {std_diff:.4f}")
    print(f"分位数差异: {q_diff:.4f}")
    print(f"中心距离: {center_dist:.4f}")
    print(f"协方差差异: {cov_diff:.4f}")
    print(f"偏度差异: {skew_diff:.4f}")
    print(f"峰度差异: {kurt_diff:.4f}")
    print(f"局部密度差异: {density_diff:.4f}")
    print(f"局部方差差异: {local_var_diff:.4f}")
    print(f"分布重叠度: {overlap:.4f}")
    print(f"最终域差异分数: {domain_gap:.4f}")
    
    return domain_gap.item()

def analyze_domain_difference(old_data_loader, new_data_loader, device='cuda'):
    """综合分析域间差异
    
    Args:
        old_data_loader: 旧域数据加载器
        new_data_loader: 新域数据加载器
        device: 计算设备
    
    Returns:
        dict: 包含各种分析结果的字典
    """
    try:
        # 收集数据
        old_data = []
        new_data = []
        
        print("\n=== 数据收集过程 ===")
        print("正在收集旧域数据...")
        for data in old_data_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            old_data.append(data.to(device))
            
        print("\n正在收集新域数据...")
        for data in new_data_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            new_data.append(data.to(device))
        
        old_data = torch.cat(old_data)
        new_data = torch.cat(new_data)
        
        # 计算域差异分数
        domain_gap = compute_domain_gap(old_data, new_data)
        
        # 计算基本统计特征
        old_stats = compute_domain_statistics(old_data_loader, device)
        new_stats = compute_domain_statistics(new_data_loader, device)
        
        # 计算域间差异
        diff_stats = compute_domain_difference(old_stats, new_stats)
        
        print("\n=== 域差异分析结果 ===")
        print(f"域差异分数: {domain_gap:.4f}")
        print(f"均值差异: {diff_stats['mean_difference']:.4f}")
        print(f"标准差差异: {diff_stats['std_difference']:.4f}")
        
        return {
            'domain_gap': domain_gap,
            'statistics': diff_stats,
            'old_stats': old_stats,
            'new_stats': new_stats
        }
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        print("错误类型:", type(e).__name__)
        import traceback
        print("错误堆栈:", traceback.format_exc())
        raise