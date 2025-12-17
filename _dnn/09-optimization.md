---
layout: course
collection: dnn
course: dnn
order: 9
module_title: "Optimization of Deep Models"
short_description: "GD variants: Momentum, Adagrad, RMSProp, Adam"
---

## Overview

- Optimization challenges in deep learning
- Gradient Descent variants
- Momentum, Adagrad, RMSProp
- Adam and related algorithms
- From-scratch implementation and comparison

## Equations

- SGD: θ ← θ − α ∇L
- Momentum: v ← β v + (1−β) ∇L; θ ← θ − α v
- Adam: m,v moments; bias correction; adaptive step sizes

## Notes

- Large batches can hurt generalization; tune batch size and learning rate schedule

## Exam Tips

- Compare adaptive vs non-adaptive methods; when each shines

## Industry Tips

- Cosine decay and warmup schedules stabilize training
