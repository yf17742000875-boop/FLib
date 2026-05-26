# D2L Article Taxonomy

Use this table to choose the right note emphasis before writing.

| Article type | Typical sections | Required emphasis | Common artifacts |
| --- | --- | --- | --- |
| Math / tensor basics | data manipulation, linear algebra, calculus, probability | shape, dtype, broadcasting, gradient meaning | concept note, API cheatsheet, small tensor demos |
| Model theory | linear regression, softmax, MLP, convolution | formulas, assumptions, gradients, loss interpretation | derivation note, graph of computation, from-scratch pseudocode |
| Data pipeline | Fashion-MNIST, sequence data, image augmentation | dataset contract, transforms, batches, workers | dataset card, loader function, sample visualization checklist |
| From-scratch implementation | softmax from scratch, MLP from scratch, training loop | parameter initialization, forward pass, loss, backward, update | runnable script, shape trace, failure checklist |
| Concise framework implementation | PyTorch concise implementations | mapping from manual implementation to framework APIs | manual-to-API table, `nn.Module` notes, optimizer/loss mapping |
| Optimization / regularization | weight decay, dropout, momentum, Adam | hyperparameter effects, stability, symptoms | experiment card, comparison table, run metadata |
| Network architecture | LeNet, AlexNet, VGG, ResNet, RNN, attention | layer-by-layer structure, shape changes, inductive bias | module card, shape trace, parameter count notes |
| Engineering computation | model construction, GPU, multi-GPU, serialization | device, memory, speed, reproducibility | debugging runbook, environment checklist |
| Project bridge | DINOv3, FoundationPose, adapted repos | where D2L concepts appear in real code | bridge note, file/module map, debugging observations |

## Type-Specific Prompts

For math / tensor basics, ask:

- What operation is being introduced?
- What shape transformations happen?
- What PyTorch API expresses it?
- What breaks when dtype or broadcasting is wrong?

For model theory, ask:

- What is the model's input/output contract?
- What loss is optimized?
- What gradient or update rule matters?
- Which assumptions make the model appropriate?

For data pipeline sections, ask:

- What is one sample?
- What is one batch?
- What transformations happen before the model sees data?
- Which loader parameters affect correctness or throughput?

For architecture sections, ask:

- How does the shape change after each layer or block?
- What inductive bias does the architecture add?
- Which part later appears in practical vision or multimodal projects?

For project bridge notes, ask:

- Which D2L concept appears in the real repo?
- What file or module contains it?
- What input/output contract must be checked?
- What debugging insight does the textbook concept provide?
