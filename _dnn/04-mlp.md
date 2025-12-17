---
layout: course
collection: dnn
course: dnn
order: 4
module_title: "Deep Feedforward Neural Networks (MLP)"
short_description: "Hidden layers, forward/backprop, depth/width effects"
---

## Overview

- Hidden layers and non-linearity
- Forward propagation
- Backpropagation and gradient computation
- Effect of depth and width on performance

## Equations

- Layer: z = W x + b, a = σ(z)
- Backprop: δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ)

## Notes

- Initialization matters: Xavier/He for stable gradients
- Non-linear activations enable complex decision boundaries

## Exam Tips

- Explain vanishing/exploding gradients and remedies

## Industry Tips

- Use batch norm before activation; dropout to regularize
