---
layout: course
collection: dnn
course: dnn
order: 2
module_title: "Artificial Neuron and Perceptron"
short_description: "Perceptron model, learning, logic gates, XOR limitation"
---

## Overview

- Biological vs artificial neuron
- Perceptron model and learning algorithm
- Logic gates (AND, OR, NOT)
- Limitations: XOR problem and linear separability

## Equations

- Perceptron: y = sign(wᵀx + b)
- Update rule: w ← w + η (y_true − y_pred) x

## Notes

- Linear separability defines what perceptrons can learn
- XOR is not linearly separable → need multi-layer networks

## Exam Tips

- Be able to prove why XOR needs at least one hidden layer

## Industry Tips

- Perceptron insights persist in margin-based classifiers like SVMs
