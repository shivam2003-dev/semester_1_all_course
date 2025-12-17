---
layout: course
collection: dnn
course: dnn
order: 6
module_title: "Recurrent Neural Networks (RNNs)"
short_description: "BPTT, LSTM/GRU, encoder–decoder, sequence tasks"
---

## Overview

- Sequence modeling and hidden states
- Backpropagation Through Time (BPTT)
- Encoder–decoder architecture
- LSTM, GRU, BiLSTM
- Applications in NLP and time series

## Equations

- LSTM gates: f, i, o; cell update Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

## Notes

- Mask padding tokens in loss to avoid noise learning

## Exam Tips

- Compare LSTM vs GRU trade-offs

## Industry Tips

- Prefer transformers for parallelism; use RNNs for streaming
