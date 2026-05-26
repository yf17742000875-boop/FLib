# AGENTS.md

## Language Policy

This English file is the canonical agent instruction file for the repository.

- Agents should read and follow `AGENTS.md`.
- A separate Chinese reference may exist for the user's convenience.
- If both files exist, `AGENTS.md` is the source of truth and the file agents should consult when reading project instructions.

## Purpose

This repository is used for hands-on study of deep learning with two parallel goals:

1. Learn the fundamentals from Dive into Deep Learning (D2L): https://zh.d2l.ai/chapter_introduction/index.html
2. Connect those fundamentals to real projects, including usage, reading, debugging, adaptation, and modification of practical libraries such as FoundationPose, DINOv3, and related vision / multimodal repos

Agents working in this repository should optimize for learning clarity first and engineering rigor second. The code and notes here are not only for getting results, but for understanding why the results happen.

## Primary Working Style

- Prefer explicit, readable code over clever or compressed code.
- When implementing anything nontrivial, explain the idea in comments or nearby Markdown if the reason is not obvious.
- Keep theory and practice connected:
  - when adding a model, loss, optimization trick, or training routine, relate it to the underlying concept from D2L when possible
  - when modifying an external repo workflow, explain what changed and why
- Favor small, testable steps over broad refactors.
- Do not introduce unnecessary abstractions early in exploratory code.

## Repository Priorities

Agents should support the following types of work:

1. D2L study notes and minimal reproductions
2. PyTorch-based experiments and training scripts
3. Reading and modifying real-world repositories
4. Debugging environments, dependencies, datasets, and training / inference pipelines
5. Building bridges from textbook concepts to practical systems

## Expected Structure

If creating new content, prefer organizing it under folders like these:

- `notes/`
  - chapter-based D2L notes
  - concept summaries
  - debugging notes
- `experiments/`
  - small self-contained reproductions
  - ablation or comparison scripts
- `projects/`
  - practical project integrations or adapted upstream repos
- `datasets/`
  - lightweight metadata, scripts, or instructions only
  - avoid checking in large raw datasets unless explicitly intended
- `tools/`
  - utility scripts for setup, conversion, evaluation, visualization, and profiling
- `reports/`
  - experiment conclusions, failure analysis, and reading summaries

Do not force this structure retroactively unless requested, but prefer it for all new additions.

## How To Handle D2L Learning Content

When working on D2L-related material:

- Keep examples minimal and educational.
- Prefer implementing core ideas from scratch before wrapping them in larger frameworks.
- If reproducing a D2L section, keep the mapping clear:
  - chapter / section name
  - key mathematical idea
  - implementation file
  - observed result
- If a concept appears again in a real project, call out the connection explicitly.

Examples of useful connections:

- linear algebra and tensor ops -> shape debugging in PyTorch repos
- convolution / pooling -> backbone feature extraction behavior
- attention -> transformer-based encoders and multimodal backbones
- optimization -> learning rate schedules, warmup, weight decay, gradient clipping
- computer vision chapters -> detection, segmentation, pose estimation, representation learning

## D2L Knowledge Base Workflow

When creating or updating D2L study material, use a section-first knowledge base structure. A D2L section usually corresponds to one D2L web page, one focused note, and optionally one minimal runnable script.

Default locations:

- Section notes: `notes/d2l/chXX/{section-id}-{slug}.md`
- Classroom or direct-follow code: `d2l/code/`
- Extended experiments and ablations: `experiments/d2l/chXX/`
- Chapter reviews: `reports/d2l/chXX-review.md`

Use ASCII filenames for new notes, experiments, and reports. Chinese prose is fine inside files. This avoids shell and encoding problems on Windows while keeping the learning content readable.

Do not move or rename existing D2L files only to enforce this structure. Apply it to new material and to files the user explicitly asks to reorganize.

Each D2L section note should include these blocks unless the section clearly makes one irrelevant:

1. `一句话总结`
2. `本节解决的问题`
3. `核心概念 / 公式 / API`
4. `输入输出形状`
5. `最小可运行代码`
6. `课后练习`
7. `易错点`
8. `和前后章节或真实项目的连接`
9. `复习卡片`

Choose the note emphasis by article type:

| D2L article type | Examples | Main knowledge artifact | Review focus |
| --- | --- | --- | --- |
| Math / tensor basics | data manipulation, linear algebra, calculus, probability | concept note + API cheatsheet | shape, dtype, broadcasting, gradient meaning |
| Model theory | linear regression, softmax, MLP, convolution | formula derivation + from-scratch explanation | inputs, outputs, losses, gradients, assumptions |
| Data pipeline | Fashion-MNIST, sequence data | dataset card + loader code | batch, shuffle, transform, `num_workers` |
| From-scratch implementation | softmax from scratch, MLP from scratch | runnable script + manual component notes | initialization, training loop, evaluation |
| Concise framework implementation | PyTorch concise implementations | manual-to-framework mapping | `nn.Module`, loss, optimizer, `DataLoader` |
| Optimization / regularization | weight decay, dropout, optimization algorithms | experiment card + comparison table | hyperparameter effects, symptoms, failure modes |
| Network architecture | LeNet, AlexNet, ResNet, RNN, attention | module card + shape trace | layer structure, receptive field, residuals, attention |
| Engineering computation | GPU, multi-GPU, model construction | debugging runbook | device, memory, speed, reproducibility |
| Project connection | DINOv3, FoundationPose, other practical repos | bridge note | where the D2L concept appears in real systems |

For review support, prefer short active-recall questions over long summaries. A good review card has a concrete prompt, a compact answer, and when useful, a tiny code or shape example.

## AI Component Development Template Framework

This section defines a standardized template framework for developing various AI components, including neural networks, optimization algorithms, probabilistic models, automatic differentiation systems, and more.

### Framework Structure

The framework consists of five core sections that all components should implement:

| Section | Purpose | Key Elements |
|---------|---------|--------------|
| **Module Definition** | Describe the component's purpose and scope | Name, description, mathematical foundation |
| **Interface Specification** | Define inputs/outputs and parameters | Type hints, parameter validation, error handling |
| **Core Implementation** | Implement the main functionality | Algorithms, data structures, computational logic |
| **Testing Strategy** | Validate correctness and performance | Unit tests, edge cases, numerical stability |
| **Usage Examples** | Demonstrate practical applications | Code examples, expected outputs, use cases |

### Template Specification

#### 1. Module Definition Section
```python
"""
{Component Name} Module
=======================

【Purpose】
Brief description of what this component does and its applications.

【Mathematical Foundation】
Key mathematical concepts and formulas underlying this component.

【References】
- D2L Chapter: {link}
- Related Concepts: {concept1}, {concept2}, {concept3}
"""
```

#### 2. Interface Specification Section
```python
def function_name(param1: Type, param2: Type, ...) -> ReturnType:
    """
    Brief description of the function's purpose.

    Parameters:
        param1: Description with type and valid range
        param2: Description with type and valid range
        ...

    Returns:
        Description of return value and format

    Raises:
        TypeError: If parameters have incorrect types
        ValueError: If parameters are out of valid range
    """
```

#### 3. Core Implementation Section
```python
class ComponentClass:
    """
    Main class implementing the component.

    Attributes:
        attr1: Description
        attr2: Description
        ...

    Methods:
        method1(): Brief description
        method2(): Brief description
        ...
    """
    
    def __init__(self, param1, param2, ...):
        """Initialize the component with given parameters."""
        # Parameter validation
        # Initialization logic
    
    def core_method(self, inputs):
        """
        Main computational method.
        
        Mathematical Principle:
        Description of the algorithm and its mathematical basis.
        """
        # Implementation with comments
```

#### 4. Testing Strategy Section
```python
def test_component_functionality():
    """Test core functionality of the component."""
    # Test case 1: Basic functionality
    # Test case 2: Edge cases
    # Test case 3: Numerical stability
    # Test case 4: Parameter validation
```

#### 5. Usage Examples Section
```python
def demo_component():
    """Demonstrate typical usage of the component."""
    # Example 1: Basic usage
    # Example 2: Advanced usage
    # Example 3: Integration with other components
```

### Example 1: Linear Neural Network Component

```python
"""
Linear Neural Network Module
============================

【Purpose】
Implements a linear layer (fully connected layer) for neural networks.
Used as building blocks in deep learning models.

【Mathematical Foundation】
Output: y = x @ W + b
Where:
- x: Input tensor of shape (batch_size, input_dim)
- W: Weight matrix of shape (input_dim, output_dim)
- b: Bias vector of shape (output_dim,)
- @: Matrix multiplication

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_linear-networks/index.html
- Related Concepts: Linear Regression, Feedforward Networks
"""

import numpy as np

class LinearLayer:
    """
    Linear layer implementation.

    Attributes:
        weight: Weight matrix (input_dim x output_dim)
        bias: Bias vector (output_dim,)
        input_dim: Input feature dimension
        output_dim: Output feature dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize linear layer.

        Parameters:
            input_dim: Number of input features
            output_dim: Number of output features

        Raises:
            ValueError: If dimensions are non-positive
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass computation.

        Parameters:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)

        Raises:
            TypeError: If input is not numpy array
            ValueError: If input shape is incorrect
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input must be numpy array, got {type(x)}")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Input shape must be (batch_size, {self.input_dim}), got {x.shape}")
        
        return x @ self.weight + self.bias
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> dict:
        """
        Backward pass computation (gradient calculation).

        Parameters:
            x: Input tensor from forward pass
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Dictionary with gradients: {'weight', 'bias', 'input'}
        """
        batch_size = x.shape[0]
        
        grad_weight = x.T @ grad_output / batch_size
        grad_bias = np.mean(grad_output, axis=0)
        grad_input = grad_output @ self.weight.T
        
        return {
            'weight': grad_weight,
            'bias': grad_bias,
            'input': grad_input
        }

# Testing
def test_linear_layer():
    """Test LinearLayer functionality."""
    layer = LinearLayer(input_dim=3, output_dim=2)
    
    # Test forward pass
    x = np.random.randn(5, 3)
    output = layer.forward(x)
    assert output.shape == (5, 2), f"Expected (5, 2), got {output.shape}"
    
    # Test backward pass
    grad_output = np.random.randn(5, 2)
    grads = layer.backward(x, grad_output)
    assert grads['weight'].shape == (3, 2)
    assert grads['bias'].shape == (2,)
    assert grads['input'].shape == (5, 3)
    
    # Test parameter validation
    try:
        LinearLayer(input_dim=0, output_dim=5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("LinearLayer tests passed!")

# Usage Demo
def demo_linear_layer():
    """Demonstrate LinearLayer usage."""
    # Create a linear layer
    layer = LinearLayer(input_dim=10, output_dim=5)
    
    # Generate random input
    x = np.random.randn(32, 10)  # Batch of 32 samples, 10 features each
    
    # Forward pass
    output = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compute gradients (simulating backpropagation)
    grad_output = np.random.randn(32, 5)
    grads = layer.backward(x, grad_output)
    print(f"Gradient shapes:")
    print(f"  weight: {grads['weight'].shape}")
    print(f"  bias: {grads['bias'].shape}")
    print(f"  input: {grads['input'].shape}")
```

### Example 2: Optimization Algorithm Component

```python
"""
Gradient Descent Optimizer Module
=================================

【Purpose】
Implements gradient descent optimization algorithm for training ML models.
Minimizes loss function by iteratively updating parameters in gradient direction.

【Mathematical Foundation】
Parameter update rule: θ = θ - lr * ∇L(θ)
Where:
- θ: Model parameters
- lr: Learning rate (step size)
- ∇L(θ): Gradient of loss function w.r.t. parameters

【References】
- D2L Chapter: https://zh.d2l.ai/chapter_optimization/index.html
- Related Concepts: Stochastic Gradient Descent, Momentum, Adam
"""

import numpy as np
from typing import Dict, Callable

class GradientDescent:
    """
    Gradient descent optimizer with momentum support.

    Attributes:
        lr: Learning rate
        momentum: Momentum factor (0 for standard GD)
        velocity: Velocity buffer for momentum
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        """
        Initialize optimizer.

        Parameters:
            lr: Learning rate (default: 0.01)
            momentum: Momentum factor [0, 1) (default: 0.0)

        Raises:
            ValueError: If lr is non-positive or momentum out of range
        """
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0 <= momentum < 1):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, params: Dict[str, np.ndarray], 
             grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one optimization step.

        Parameters:
            params: Dictionary of parameter arrays
            grads: Dictionary of gradient arrays (same keys as params)

        Returns:
            Updated parameters dictionary

        Raises:
            KeyError: If params and grads have mismatched keys
        """
        if params.keys() != grads.keys():
            raise KeyError("params and grads must have the same keys")
        
        updated_params = {}
        
        for key in params:
            grad = grads[key]
            
            # Initialize velocity if not exists
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Apply momentum
            self.velocity[key] = self.momentum * self.velocity[key] + grad
            
            # Update parameters
            updated_params[key] = params[key] - self.lr * self.velocity[key]
        
        return updated_params
    
    def zero_grad(self):
        """Reset velocity buffers."""
        self.velocity = {}

# Testing
def test_gradient_descent():
    """Test GradientDescent optimizer."""
    optimizer = GradientDescent(lr=0.1, momentum=0.9)
    
    # Test basic step
    params = {'w': np.array([1.0, 2.0]), 'b': np.array([0.5])}
    grads = {'w': np.array([0.1, 0.2]), 'b': np.array([0.05])}
    
    updated = optimizer.step(params, grads)
    assert updated['w'].shape == (2,)
    assert updated['b'].shape == (1,)
    
    # Test momentum accumulation
    updated2 = optimizer.step(params, grads)
    # With momentum, velocity should accumulate
    assert not np.allclose(updated['w'], updated2['w'])
    
    # Test parameter validation
    try:
        GradientDescent(lr=-0.1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    try:
        GradientDescent(momentum=1.0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("GradientDescent tests passed!")

# Usage Demo
def demo_gradient_descent():
    """Demonstrate GradientDescent usage."""
    # Define a simple quadratic loss function
    def loss(w):
        return (w - 3) ** 2  # Minimum at w = 3
    
    def grad_loss(w):
        return 2 * (w - 3)  # Derivative
    
    # Initialize optimizer
    optimizer = GradientDescent(lr=0.1, momentum=0.9)
    
    # Initialize parameter
    params = {'w': np.array([0.0])}
    
    # Training loop
    print("Optimization steps:")
    for i in range(10):
        grads = {'w': grad_loss(params['w'])}
        params = optimizer.step(params, grads)
        current_loss = loss(params['w'])
        print(f"Step {i+1}: w = {params['w'][0]:.4f}, loss = {current_loss:.4f}")
    
    print(f"\nFinal w = {params['w'][0]:.4f} (expected ~3.0)")
```

### Best Practices for Component Development

1. **Modularity**: Each component should have a single, well-defined purpose
2. **Type Hints**: Use Python type hints for better documentation and IDE support
3. **Parameter Validation**: Always validate inputs with clear error messages
4. **Documentation**: Provide docstrings with mathematical explanations
5. **Testing**: Include comprehensive unit tests covering edge cases
6. **Numerical Stability**: Handle edge cases like division by zero
7. **Consistency**: Follow naming conventions and code style consistently

### Framework Extensibility

The template framework can be extended to support:
- **Neural Network Layers**: Convolutional layers, recurrent layers, attention mechanisms
- **Optimizers**: Adam, RMSprop, Adagrad, learning rate scheduling
- **Probabilistic Models**: Bayesian networks, hidden Markov models, variational inference
- **Loss Functions**: Cross-entropy, MSE, custom loss functions
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score

## How To Handle Real Project Repos

When using or modifying practical repos such as FoundationPose, DINOv3, or similar codebases:

- Preserve upstream behavior unless there is a clear reason to change it.
- Separate local adaptations from upstream-like code as much as possible.
- Document every important modification:
  - what file or module changed
  - whether the change is for compatibility, debugging, experimentation, or performance
  - expected impact on outputs, speed, memory, or reproducibility
- Prefer writing thin wrappers, adapters, config overrides, or experiment scripts before deeply editing third-party internals.
- If deep edits are necessary, keep them small and well explained.

## Modification Rules For External Libraries

- Never silently change model semantics.
- Do not remove assertions, safety checks, or preprocessing steps unless the reason is documented.
- Before changing a training or inference pipeline, identify:
  - inputs and expected shapes
  - devices and dtype assumptions
  - checkpoint loading path
  - config dependencies
  - output contract
- If fixing a bug, record:
  - symptom
  - root cause
  - validation method

## Code Quality Expectations

- Use Python and PyTorch idioms unless the target repo clearly uses another stack.
- Prefer scripts that can be run independently with clear arguments.
- Add concise docstrings or comments for:
  - tensor shape expectations
  - non-obvious coordinate systems
  - preprocessing and normalization assumptions
  - loss composition
  - evaluation logic
- Avoid hardcoding machine-specific paths. Prefer config files, environment variables, or clearly marked local placeholders.
- Keep notebooks optional. Important logic should live in normal source files when feasible.

## Experiment Discipline

For experiments, aim to make runs understandable and reproducible:

- Record config, dataset version, checkpoint source, and main metrics.
- Save short conclusions, not just raw logs.
- Note failures and surprising outcomes, not only successful runs.
- When comparing methods, change one major variable at a time when possible.

Useful experiment metadata includes:

- objective
- hypothesis
- environment
- command
- metrics
- qualitative observations
- next action

## Debugging Expectations

When debugging:

- Start by narrowing scope: environment, import, data, model, loss, optimization, evaluation, or visualization.
- Verify shapes, dtype, device placement, and coordinate conventions early.
- For vision projects, inspect representative inputs and outputs whenever possible.
- Prefer targeted logging and minimal repro scripts over broad speculative edits.
- If a failure is likely caused by version mismatch, identify the exact package, API, or checkpoint assumption.

## Communication Style For Agents

When making changes in this repository, agents should:

- state assumptions clearly
- explain tradeoffs briefly and concretely
- summarize what was learned, not only what was changed
- keep responses concise but technically meaningful

If asked to review code, prioritize:

1. correctness
2. training / inference behavior
3. reproducibility
4. maintainability
5. performance

## Safe Defaults

- Prefer non-destructive edits.
- Do not delete user experiments, notes, checkpoints, or datasets unless explicitly asked.
- Treat large generated artifacts as disposable unless the user indicates they should be versioned.
- Avoid network downloads or heavyweight setup changes unless necessary for the task.

## Good Output From Agents

Strong contributions in this repo usually do at least one of these:

- make a D2L concept easier to understand
- turn a theoretical concept into runnable code
- make a practical repo easier to run or debug
- explain a real bug with concrete technical reasoning
- improve reproducibility of an experiment

## When Unsure

If multiple valid approaches exist, prefer the one that:

1. teaches the underlying concept more clearly
2. keeps the implementation easier to inspect
3. minimizes irreversible changes to external code
4. makes future experiments easier to compare
