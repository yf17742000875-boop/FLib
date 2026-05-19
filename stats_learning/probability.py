"""
Probability Module
==================

【Purpose】
Provides functions for probability theory concepts including distributions,
conditional probability, Bayesian inference, and Markov chains.

【Mathematical Foundation】
- Normal Distribution: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
- Binomial Distribution: P(k) = C(n,k) * p^k * (1-p)^(n-k)
- Poisson Distribution: P(k) = (e^(-λ) * λ^k) / k!
- Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_preliminaries/probability.html
- Related Concepts: Probability Theory, Bayesian Inference, Markov Processes
"""

import numpy as np
import math
from scipy.special import comb, factorial
from .visualization import probability_distribution, multiple_distributions

def normal_pdf(x, mean=0, std=1):
    """
    Calculate probability density function for normal distribution.

    Parameters:
        x: Value or array of values
        mean: Mean of the distribution (default: 0)
        std: Standard deviation (default: 1)

    Returns:
        float or np.ndarray: Probability density
    """
    x = np.asarray(x, dtype=np.float64)
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def normal_cdf(x, mean=0, std=1):
    """
    Calculate cumulative distribution function for normal distribution.

    Parameters:
        x: Value or array of values
        mean: Mean of the distribution (default: 0)
        std: Standard deviation (default: 1)

    Returns:
        float or np.ndarray: Cumulative probability
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1 + math.erf((x - mean) / (std * np.sqrt(2))))

def binomial_pmf(k, n, p):
    """
    Calculate probability mass function for binomial distribution.

    Parameters:
        k: Number of successes
        n: Number of trials
        p: Probability of success

    Returns:
        float: Probability
    """
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binomial_cdf(k, n, p):
    """
    Calculate cumulative distribution function for binomial distribution.

    Parameters:
        k: Maximum number of successes
        n: Number of trials
        p: Probability of success

    Returns:
        float: Cumulative probability
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    
    total = 0.0
    for i in range(int(k) + 1):
        total += binomial_pmf(i, n, p)
    return total

def poisson_pmf(k, lam):
    """
    Calculate probability mass function for Poisson distribution.

    Parameters:
        k: Number of events
        lam: Average rate (lambda)

    Returns:
        float: Probability
    """
    if k < 0:
        return 0.0
    return (np.exp(-lam) * (lam ** k)) / factorial(k)

def poisson_cdf(k, lam):
    """
    Calculate cumulative distribution function for Poisson distribution.

    Parameters:
        k: Maximum number of events
        lam: Average rate (lambda)

    Returns:
        float: Cumulative probability
    """
    if k < 0:
        return 0.0
    
    total = 0.0
    for i in range(int(k) + 1):
        total += poisson_pmf(i, lam)
    return total

def uniform_pdf(x, a=0, b=1):
    """
    Calculate probability density function for uniform distribution.

    Parameters:
        x: Value or array of values
        a: Lower bound (default: 0)
        b: Upper bound (default: 1)

    Returns:
        float or np.ndarray: Probability density
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    result[(x >= a) & (x <= b)] = 1 / (b - a)
    return result

def uniform_cdf(x, a=0, b=1):
    """
    Calculate cumulative distribution function for uniform distribution.

    Parameters:
        x: Value or array of values
        a: Lower bound (default: 0)
        b: Upper bound (default: 1)

    Returns:
        float or np.ndarray: Cumulative probability
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))
    return result

def conditional_probability(p_ab, p_b):
    """
    Calculate conditional probability P(A|B) = P(A∩B) / P(B).

    Parameters:
        p_ab: Probability of A and B (P(A∩B))
        p_b: Probability of B (P(B))

    Returns:
        float: Conditional probability P(A|B)

    Raises:
        ValueError: If p_b is zero
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return p_ab / p_b

def bayes_theorem(p_b_given_a, p_a, p_b):
    """
    Apply Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B).

    Parameters:
        p_b_given_a: Probability of B given A (P(B|A))
        p_a: Prior probability of A (P(A))
        p_b: Total probability of B (P(B))

    Returns:
        float: Posterior probability P(A|B)

    Raises:
        ValueError: If p_b is zero
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return (p_b_given_a * p_a) / p_b

def total_probability(priors, likelihoods):
    """
    Calculate total probability using law of total probability.

    Parameters:
        priors: List of prior probabilities P(A_i)
        likelihoods: List of likelihoods P(B|A_i)

    Returns:
        float: Total probability P(B)
    """
    if len(priors) != len(likelihoods):
        raise ValueError("priors and likelihoods must have same length")
    
    total = 0.0
    for p_a, p_b_given_a in zip(priors, likelihoods):
        total += p_a * p_b_given_a
    return total

def expected_value(values, probabilities):
    """
    Calculate expected value E[X].

    Parameters:
        values: List of possible values
        probabilities: List of corresponding probabilities

    Returns:
        float: Expected value
    """
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have same length")
    
    return sum(v * p for v, p in zip(values, probabilities))

def variance_from_dist(values, probabilities):
    """
    Calculate variance from probability distribution.

    Parameters:
        values: List of possible values
        probabilities: List of corresponding probabilities

    Returns:
        float: Variance
    """
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have same length")
    
    ev = expected_value(values, probabilities)
    return sum((v - ev) ** 2 * p for v, p in zip(values, probabilities))

class MarkovChain:
    """
    Simple Markov Chain implementation.

    Attributes:
        transition_matrix: Transition probability matrix
        states: List of state names
    """
    
    def __init__(self, transition_matrix, states=None):
        """
        Initialize Markov Chain.

        Parameters:
            transition_matrix: Square matrix of transition probabilities
            states: Optional list of state names
        """
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        self.n_states = self.transition_matrix.shape[0]
        
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
        
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Rows of transition matrix must sum to 1")
        
        if states is None:
            self.states = [f'State {i}' for i in range(self.n_states)]
        else:
            self.states = states
    
    def step(self, current_state):
        """
        Perform one step in the Markov chain.

        Parameters:
            current_state: Current state index

        Returns:
            int: Next state index
        """
        if current_state < 0 or current_state >= self.n_states:
            raise ValueError(f"Invalid state: {current_state}")
        
        probabilities = self.transition_matrix[current_state]
        return np.random.choice(self.n_states, p=probabilities)
    
    def simulate(self, start_state, n_steps):
        """
        Simulate the Markov chain for n steps.

        Parameters:
            start_state: Starting state index
            n_steps: Number of steps to simulate

        Returns:
            list: Sequence of states visited
        """
        states = [start_state]
        current = start_state
        
        for _ in range(n_steps):
            current = self.step(current)
            states.append(current)
        
        return states
    
    def get_state_distribution(self, initial_distribution, n_steps):
        """
        Compute state distribution after n steps.

        Parameters:
            initial_distribution: Initial state probabilities
            n_steps: Number of steps

        Returns:
            np.ndarray: State distribution after n steps
        """
        dist = np.asarray(initial_distribution, dtype=np.float64)
        
        for _ in range(n_steps):
            dist = dist @ self.transition_matrix
        
        return dist
    
    def get_stationary_distribution(self, tolerance=1e-10, max_iter=1000):
        """
        Compute stationary distribution using power iteration.

        Parameters:
            tolerance: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            np.ndarray: Stationary distribution
        """
        dist = np.ones(self.n_states) / self.n_states
        
        for _ in range(max_iter):
            new_dist = dist @ self.transition_matrix
            
            if np.linalg.norm(new_dist - dist) < tolerance:
                return new_dist
            
            dist = new_dist
        
        return dist

def plot_normal_distribution(mean=0, std=1, show=True):
    """
    Plot normal distribution PDF.

    Parameters:
        mean: Mean of the distribution
        std: Standard deviation
        show: Whether to display the plot

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    y = normal_pdf(x, mean, std)
    
    return probability_distribution(x, y, f'Normal (μ={mean}, σ={std})', show=show)

def plot_binomial_distribution(n=10, p=0.5, show=True):
    """
    Plot binomial distribution PMF.

    Parameters:
        n: Number of trials
        p: Probability of success
        show: Whether to display the plot

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    x = np.arange(0, n + 1)
    y = [binomial_pmf(k, n, p) for k in x]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color='skyblue', edgecolor='black')
    ax.set_title(f'Binomial Distribution (n={n}, p={p})', fontsize=14, pad=20)
    ax.set_xlabel('Number of Successes', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def plot_poisson_distribution(lam=5, show=True):
    """
    Plot Poisson distribution PMF.

    Parameters:
        lam: Lambda parameter (mean)
        show: Whether to display the plot

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    x = np.arange(0, int(lam * 3) + 1)
    y = [poisson_pmf(k, lam) for k in x]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color='green', edgecolor='black')
    ax.set_title(f'Poisson Distribution (λ={lam})', fontsize=14, pad=20)
    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
    
    return fig, ax

def compare_normal_distributions(parameters, show=True):
    """
    Compare multiple normal distributions.

    Parameters:
        parameters: List of (mean, std) tuples
        show: Whether to display the plot

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    x_min = min(p[0] - 4 * p[1] for p in parameters)
    x_max = max(p[0] + 4 * p[1] for p in parameters)
    x = np.linspace(x_min, x_max, 1000)
    
    y_list = []
    labels = []
    
    for mean, std in parameters:
        y = normal_pdf(x, mean, std)
        y_list.append(y)
        labels.append(f'μ={mean}, σ={std}')
    
    return multiple_distributions([x] * len(parameters), y_list, labels,
                                 title='Normal Distributions Comparison', show=show)

def bayesian_inference_example():
    """
    Interactive example demonstrating Bayesian inference.
    
    Example scenario: Medical test for a disease
    """
    print(f"{'='*60}")
    print(f"贝叶斯定理示例: 医学检测")
    print(f"{'='*60}")
    print(f"\n场景描述:")
    print(f"假设某种疾病的发病率为 1% (P(Disease) = 0.01)")
    print(f"检测的准确率为 95% (P(Positive|Disease) = 0.95)")
    print(f"检测的假阳性率为 5% (P(Positive|No Disease) = 0.05)")
    print(f"\n问题: 如果检测结果为阳性，真正患病的概率是多少？")
    
    p_disease = 0.01
    p_no_disease = 1 - p_disease
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.05
    
    p_positive = total_probability(
        [p_disease, p_no_disease],
        [p_positive_given_disease, p_positive_given_no_disease]
    )
    
    p_disease_given_positive = bayes_theorem(
        p_positive_given_disease,
        p_disease,
        p_positive
    )
    
    print(f"\n计算过程:")
    print(f"1. P(Disease) = {p_disease:.4f}")
    print(f"2. P(No Disease) = {p_no_disease:.4f}")
    print(f"3. P(Positive|Disease) = {p_positive_given_disease:.4f}")
    print(f"4. P(Positive|No Disease) = {p_positive_given_no_disease:.4f}")
    print(f"5. P(Positive) = P(Disease)*P(Positive|Disease) + P(No Disease)*P(Positive|No Disease)")
    print(f"   = {p_disease} * {p_positive_given_disease} + {p_no_disease} * {p_positive_given_no_disease}")
    print(f"   = {p_positive:.4f}")
    print(f"\n6. 应用贝叶斯定理:")
    print(f"   P(Disease|Positive) = P(Positive|Disease) * P(Disease) / P(Positive)")
    print(f"   = {p_positive_given_disease} * {p_disease} / {p_positive:.4f}")
    print(f"   = {p_disease_given_positive:.4f}")
    
    print(f"\n{'='*60}")
    print(f"结论: 检测结果为阳性时，真正患病的概率约为 {p_disease_given_positive*100:.1f}%")
    print(f"{'='*60}")
    print(f"\n这个结果可能令人惊讶！即使检测准确率很高，由于疾病本身")
    print(f"发病率很低，阳性结果中大部分是假阳性。这就是贝叶斯思维的重要性。")
    
    return {
        'P(Disease)': p_disease,
        'P(Positive|Disease)': p_positive_given_disease,
        'P(Positive|No Disease)': p_positive_given_no_disease,
        'P(Disease|Positive)': p_disease_given_positive
    }

def markov_chain_example():
    """
    Interactive example demonstrating Markov Chain.
    
    Example scenario: Weather forecasting
    """
    print(f"{'='*60}")
    print(f"马尔科夫链示例: 天气预报")
    print(f"{'='*60}")
    
    transition_matrix = [
        [0.7, 0.2, 0.1],  # Sunny -> Sunny, Cloudy, Rainy
        [0.3, 0.5, 0.2],  # Cloudy -> Sunny, Cloudy, Rainy
        [0.2, 0.4, 0.4]   # Rainy -> Sunny, Cloudy, Rainy
    ]
    
    states = ['晴天', '多云', '雨天']
    mc = MarkovChain(transition_matrix, states)
    
    print(f"\n转移概率矩阵:")
    print(f"              {' | '.join(states)}")
    print(f"{'='*40}")
    for i, state in enumerate(states):
        probs = ' | '.join(f'{p:.2f}' for p in transition_matrix[i])
        print(f"从 {state:4s} 出发: {probs}")
    
    print(f"\n模拟未来7天的天气（从晴天开始）:")
    simulation = mc.simulate(0, 7)
    for day, state_idx in enumerate(simulation):
        print(f"  第{day}天: {states[state_idx]}")
    
    print(f"\n初始分布: [晴天=1.0, 多云=0.0, 雨天=0.0]")
    print(f"经过不同天数后的分布:")
    initial_dist = [1.0, 0.0, 0.0]
    for days in [1, 3, 7, 30]:
        dist = mc.get_state_distribution(initial_dist, days)
        print(f"  {days}天后: 晴天={dist[0]:.3f}, 多云={dist[1]:.3f}, 雨天={dist[2]:.3f}")
    
    stationary = mc.get_stationary_distribution()
    print(f"\n平稳分布（长期趋势）:")
    print(f"  晴天: {stationary[0]:.3f}")
    print(f"  多云: {stationary[1]:.3f}")
    print(f"  雨天: {stationary[2]:.3f}")
    
    return {
        'transition_matrix': transition_matrix,
        'states': states,
        'simulation': [states[i] for i in simulation],
        'stationary_distribution': stationary
    }
