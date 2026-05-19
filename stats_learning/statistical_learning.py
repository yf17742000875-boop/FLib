"""
Statistical Learning Theory Module
==================================

【Purpose】
Provides interactive demonstrations for statistical learning concepts including
bias-variance tradeoff, cross-validation, regularization, and model selection.

【Mathematical Foundation】
- Bias-Variance Decomposition: E[(y - ŷ)²] = Bias² + Variance + Noise
- Cross-Validation: k-fold, leave-one-out
- Regularization: L1 (Lasso), L2 (Ridge)
- Model Complexity vs Generalization

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/weight-decay.html
- Related Concepts: Overfitting, Underfitting, Generalization Error
"""

import numpy as np
import matplotlib.pyplot as plt

def bias_variance_decomposition():
    """
    Interactive demonstration of bias-variance tradeoff.
    """
    print(f"{'='*60}")
    print(f"偏差-方差权衡演示")
    print(f"{'='*60}")
    
    print(f"\n偏差-方差分解公式:")
    print(f"  E[(y - ŷ)²] = Bias²(ŷ) + Variance(ŷ) + Noise")
    
    print(f"\n其中:")
    print(f"  - Bias²(ŷ) = [E[ŷ] - y]²: 模型的系统性误差")
    print(f"  - Variance(ŷ) = E[(ŷ - E[ŷ])²]: 模型的随机性")
    print(f"  - Noise = E[(y - E[y])²]: 数据本身的噪声")
    
    np.random.seed(42)
    
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    def generate_data(n_samples=20, noise_std=0.1):
        X = np.random.uniform(0, 1, n_samples)
        y = true_function(X) + noise_std * np.random.randn(n_samples)
        return X, y
    
    def fit_polynomial(X, y, degree):
        coeffs = np.polyfit(X, y, degree)
        return coeffs
    
    def predict_polynomial(coeffs, X):
        return np.polyval(coeffs, X)
    
    n_trials = 50
    n_samples = 20
    degrees = [1, 3, 9, 15]
    
    plt.figure(figsize=(12, 8))
    
    for i, degree in enumerate(degrees):
        predictions = []
        
        for _ in range(n_trials):
            X, y = generate_data(n_samples)
            coeffs = fit_polynomial(X, y, degree)
            X_test = np.linspace(0, 1, 100)
            y_pred = predict_polynomial(coeffs, X_test)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        ax = plt.subplot(2, 2, i + 1)
        X_test = np.linspace(0, 1, 100)
        ax.plot(X_test, true_function(X_test), 'r--', label='True Function', linewidth=2)
        ax.plot(X_test, mean_pred, 'b-', label='Average Prediction', linewidth=2)
        
        for j in range(min(5, n_trials)):
            ax.plot(X_test, predictions[j], 'gray', alpha=0.3)
        
        ax.fill_between(X_test, mean_pred - np.sqrt(var_pred), 
                       mean_pred + np.sqrt(var_pred), 
                       color='blue', alpha=0.2, label='±1 Std')
        
        ax.set_title(f'Degree {degree} Polynomial')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"观察结果分析:")
    print(f"{'='*60}")
    print(f"\n1. 低复杂度模型（如1次多项式）:")
    print(f"   - 偏差高: 模型过于简单，无法捕捉真实函数的复杂模式")
    print(f"   - 方差低: 多次训练结果稳定，变化不大")
    print(f"   - 表现: 欠拟合")
    
    print(f"\n2. 中等复杂度模型（如3次多项式）:")
    print(f"   - 偏差适中: 能够较好地近似真实函数")
    print(f"   - 方差适中: 训练结果有一定变化但可控")
    print(f"   - 表现: 最佳泛化能力")
    
    print(f"\n3. 高复杂度模型（如15次多项式）:")
    print(f"   - 偏差低: 几乎完美拟合训练数据")
    print(f"   - 方差高: 多次训练结果差异很大")
    print(f"   - 表现: 过拟合")

def k_fold_cross_validation(X, y, model_class, k=5, **model_kwargs):
    """
    Perform k-fold cross-validation.

    Parameters:
        X: Feature matrix
        y: Target vector
        model_class: Model class with fit and predict methods
        k: Number of folds (default: 5)
        **model_kwargs: Keyword arguments for model initialization

    Returns:
        dict: Results including train and validation scores
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    
    n_samples = len(y)
    fold_size = n_samples // k
    indices = np.random.permutation(n_samples)
    
    train_scores = []
    val_scores = []
    
    for i in range(k):
        val_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_score = 1 - np.mean((y_train - y_train_pred) ** 2) / np.var(y_train)
        val_score = 1 - np.mean((y_val - y_val_pred) ** 2) / np.var(y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    return {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'mean_train_score': np.mean(train_scores),
        'mean_val_score': np.mean(val_scores),
        'std_train_score': np.std(train_scores),
        'std_val_score': np.std(val_scores)
    }

def cross_validation_demo():
    """
    Interactive demonstration of cross-validation.
    """
    print(f"{'='*60}")
    print(f"交叉验证演示")
    print(f"{'='*60}")
    
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.flatten() + 3 + 0.5 * np.random.randn(100)
    
    from .regression import LinearRegression
    
    k_values = [3, 5, 10, 20]
    
    print(f"\n不同k值的交叉验证结果:")
    print(f"{'='*60}")
    
    for k in k_values:
        results = k_fold_cross_validation(X, y, LinearRegression, k=k)
        print(f"\nk={k}:")
        print(f"  训练R²: {results['mean_train_score']:.4f} ± {results['std_train_score']:.4f}")
        print(f"  验证R²: {results['mean_val_score']:.4f} ± {results['std_val_score']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"交叉验证原理说明:")
    print(f"{'='*60}")
    print(f"\n1. 数据划分: 将数据随机划分为k个相等的子集（folds）")
    print(f"2. 训练与验证:")
    print(f"   - 使用k-1个子集训练模型")
    print(f"   - 使用剩余1个子集验证模型")
    print(f"3. 重复: 重复k次，每次使用不同的子集作为验证集")
    print(f"4. 评估: 平均k次验证分数作为最终模型性能估计")
    
    print(f"\n选择k值的考量:")
    print(f"  - k太小: 训练数据少，模型不稳定，验证方差大")
    print(f"  - k太大: 训练数据多，验证集小，验证偏差大")
    print(f"  - 常用值: k=5 或 k=10")

class RegularizedLinearRegression:
    """
    Linear Regression with L1 (Lasso) and L2 (Ridge) regularization.

    Attributes:
        coef_: Coefficient vector
        intercept_: Intercept term
        alpha: Regularization strength
        penalty: 'l1', 'l2', or 'elasticnet'
        l1_ratio: Mixing ratio for elastic net (0 = L2, 1 = L1)
    """
    
    def __init__(self, alpha=1.0, penalty='l2', l1_ratio=0.5, fit_intercept=True):
        """
        Initialize RegularizedLinearRegression.

        Parameters:
            alpha: Regularization strength (default: 1.0)
            penalty: Regularization type: 'l1', 'l2', or 'elasticnet'
            l1_ratio: Mixing ratio for elastic net (0 = L2, 1 = L1)
            fit_intercept: Whether to fit an intercept term
        """
        self.alpha = alpha
        self.penalty = penalty.lower()
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit regularized linear regression model.

        Parameters:
            X: Feature matrix (n_samples x n_features)
            y: Target vector (n_samples,)

        Returns:
            self: Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        XTX = X.T @ X
        XTy = X.T @ y
        
        if self.penalty == 'l2':
            penalty_matrix = self.alpha * np.eye(n_features)
            if self.fit_intercept:
                penalty_matrix[0, 0] = 0
            beta = np.linalg.solve(XTX + penalty_matrix, XTy)
        
        elif self.penalty == 'l1':
            beta = self._lasso_coordinate_descent(X, y, n_features)
        
        elif self.penalty == 'elasticnet':
            penalty_matrix = self.alpha * (1 - self.l1_ratio) * np.eye(n_features)
            if self.fit_intercept:
                penalty_matrix[0, 0] = 0
            XTX_reg = XTX + penalty_matrix
            beta = self._elasticnet_coordinate_descent(X, y, XTX_reg, n_features)
        
        else:
            raise ValueError("penalty must be 'l1', 'l2', or 'elasticnet'")
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        
        return self
    
    def _lasso_coordinate_descent(self, X, y, n_features, max_iter=1000, tol=1e-6):
        """Coordinate descent for L1 regularization."""
        beta = np.zeros(n_features)
        XTX_diag = np.diag(X.T @ X)
        
        for _ in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                if self.fit_intercept and j == 0:
                    continue
                
                residual = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j].T @ residual
                
                if rho_j < -self.alpha:
                    beta[j] = (rho_j + self.alpha) / XTX_diag[j]
                elif rho_j > self.alpha:
                    beta[j] = (rho_j - self.alpha) / XTX_diag[j]
                else:
                    beta[j] = 0
            
            if np.linalg.norm(beta - beta_old) < tol:
                break
        
        return beta
    
    def _elasticnet_coordinate_descent(self, X, y, XTX_reg, n_features, max_iter=1000, tol=1e-6):
        """Coordinate descent for elastic net regularization."""
        beta = np.zeros(n_features)
        XTX_diag = np.diag(XTX_reg)
        
        for _ in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                if self.fit_intercept and j == 0:
                    continue
                
                residual = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j].T @ residual
                lambda1 = self.alpha * self.l1_ratio
                
                if rho_j < -lambda1:
                    beta[j] = (rho_j + lambda1) / XTX_diag[j]
                elif rho_j > lambda1:
                    beta[j] = (rho_j - lambda1) / XTX_diag[j]
                else:
                    beta[j] = 0
            
            if np.linalg.norm(beta - beta_old) < tol:
                break
        
        return beta
    
    def predict(self, X):
        """
        Predict target values.

        Parameters:
            X: Feature matrix (n_samples x n_features)

        Returns:
            np.ndarray: Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        
        return X @ np.concatenate([[self.intercept_], self.coef_])

def regularization_demo():
    """
    Interactive demonstration of regularization.
    """
    print(f"{'='*60}")
    print(f"正则化演示")
    print(f"{'='*60}")
    
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    true_coeffs = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_coeffs + 0.5 * np.random.randn(n_samples)
    
    alphas = [0, 0.01, 0.1, 1, 10, 100]
    
    print(f"\n正则化强度(alpha)对系数的影响:")
    print(f"{'='*60}")
    
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        ridge_model = RegularizedLinearRegression(alpha=alpha, penalty='l2')
        ridge_model.fit(X, y)
        
        lasso_model = RegularizedLinearRegression(alpha=alpha, penalty='l1')
        lasso_model.fit(X, y)
        
        plt.plot(range(n_features), ridge_model.coef_, label=f'Ridge (α={alpha})', 
                 linestyle='-', marker='o')
        plt.plot(range(n_features), lasso_model.coef_, label=f'Lasso (α={alpha})', 
                 linestyle='--', marker='x')
    
    plt.plot(range(n_features), true_coeffs, label='True Coefficients', 
             linestyle=':', marker='s', color='black', linewidth=2)
    
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Effect of Regularization on Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.75)
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"L2正则化(Ridge Regression):")
    print(f"{'='*60}")
    print(f"- 惩罚项: λ * ||β||²")
    print(f"- 效果: 所有系数都被缩小，但不会变为0")
    print(f"- 用途: 处理多重共线性，稳定模型")
    
    print(f"\n{'='*60}")
    print(f"L1正则化(Lasso Regression):")
    print(f"{'='*60}")
    print(f"- 惩罚项: λ * ||β||₁")
    print(f"- 效果: 某些系数被压缩为0，实现特征选择")
    print(f"- 用途: 特征选择，处理稀疏数据")
    
    print(f"\n{'='*60}")
    print(f"选择正则化强度(alpha):")
    print(f"{'='*60}")
    print(f"- alpha=0: 无正则化，普通线性回归")
    print(f"- alpha很小: 正则化效果弱，接近普通回归")
    print(f"- alpha适中: 平衡拟合与正则化")
    print(f"- alpha很大: 正则化效果强，系数趋近于0")

def model_selection_demo():
    """
    Interactive demonstration of model selection.
    """
    print(f"{'='*60}")
    print(f"模型选择演示")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * x
    
    n_samples = 30
    X = np.random.uniform(0, 1, n_samples)
    y = true_function(X) + 0.3 * np.random.randn(n_samples)
    
    degrees = range(1, 11)
    train_errors = []
    val_errors = []
    
    X_val = np.linspace(0, 1, 100)
    y_val_true = true_function(X_val)
    
    for degree in degrees:
        coeffs = np.polyfit(X, y, degree)
        y_train_pred = np.polyval(coeffs, X)
        y_val_pred = np.polyval(coeffs, X_val)
        
        train_mse = np.mean((y - y_train_pred) ** 2)
        val_mse = np.mean((y_val_true - y_val_pred) ** 2)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
    
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, label='Training Error', marker='o', color='blue')
    plt.plot(degrees, val_errors, label='Validation Error', marker='x', color='red')
    plt.axvline(x=3, color='green', linestyle='--', label='Optimal Complexity')
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Selection: Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.75)
    plt.show()
    
    print(f"\n模型选择流程:")
    print(f"{'='*60}")
    print(f"\n1. 数据划分:")
    print(f"   - 训练集: 用于模型训练")
    print(f"   - 验证集: 用于模型选择")
    print(f"   - 测试集: 用于最终评估")
    
    print(f"\n2. 训练多个模型:")
    print(f"   - 尝试不同复杂度的模型")
    print(f"   - 记录训练误差和验证误差")
    
    print(f"\n3. 选择最佳模型:")
    print(f"   - 验证误差最小的模型通常是最佳选择")
    print(f"   - 当验证误差开始上升时，表示过拟合")
    
    print(f"\n4. 最终评估:")
    print(f"   - 使用测试集评估选中模型")
    print(f"   - 测试误差是模型泛化能力的无偏估计")

def learning_curves_demo():
    """
    Interactive demonstration of learning curves.
    """
    print(f"{'='*60}")
    print(f"学习曲线演示")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    def true_function(x):
        return 2 * x + 3 + 0.5 * np.random.randn(len(x))
    
    n_samples_list = [5, 10, 20, 50, 100, 200]
    train_errors = []
    val_errors = []
    
    for n_samples in n_samples_list:
        X_train = np.random.randn(n_samples, 1)
        y_train = true_function(X_train.flatten())
        
        X_val = np.random.randn(100, 1)
        y_val = true_function(X_val.flatten())
        
        from .regression import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        val_mse = np.mean((y_val - y_val_pred) ** 2)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_list, train_errors, label='Training Error', marker='o', color='blue')
    plt.plot(n_samples_list, val_errors, label='Validation Error', marker='x', color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.75)
    plt.show()
    
    print(f"\n学习曲线分析:")
    print(f"{'='*60}")
    print(f"\n1. 训练误差:")
    print(f"   - 数据量小时: 误差低，模型容易拟合少量数据")
    print(f"   - 数据量大时: 误差可能上升，但趋于稳定")
    
    print(f"\n2. 验证误差:")
    print(f"   - 数据量小时: 误差高，模型泛化能力差")
    print(f"   - 数据量大时: 误差下降，模型泛化能力提升")
    
    print(f"\n3. 曲线间距:")
    print(f"   - 间距大: 过拟合，模型复杂度太高")
    print(f"   - 间距小: 模型拟合充分")
    
    print(f"\n4. 收敛趋势:")
    print(f"   - 两条曲线收敛: 增加数据不会显著改善")
    print(f"   - 两条曲线不收敛: 增加数据可能有帮助")
