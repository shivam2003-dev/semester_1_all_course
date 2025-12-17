---
layout: course
collection: dnn
course: dnn
order: 3
module_title: "Linear Neural Networks"
short_description: "Regression and classification with linear nets"
---

## Regression

- Single-neuron linear models
- Squared loss and mini-batch SGD

### Loss

- MSE: L = (1/n) Σ (y − ŷ)²

## Classification

- Binary: sigmoid + cross-entropy
- Multi-class: softmax + cross-entropy
- From-scratch implementations

### Equations

- σ(z) = 1 / (1 + e^{−z})
- softmax(zᵢ) = e^{zᵢ} / Σ e^{zⱼ}

## Notes

- Feature scaling is essential for stable training

## Exam Tips

- Derive gradient of cross-entropy with softmax

## Industry Tips

- Start with linear baselines for sanity checks before deep models
