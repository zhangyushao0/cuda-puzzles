# Issues and Gotchas

## Numerical Stability
- Softmax MUST use max-subtraction trick: softmax(z-max(z)) to prevent overflow
- Cross-entropy needs epsilon protection: log(p + 1e-10)
- Test with large logits [1000, 1001, 1002] to verify stability

## Byte Swapping
- MNIST IDX format is big-endian
- Windows is little-endian → need swap_endian() function
- Magic numbers: 0x00000803 (images), 0x00000801 (labels)

## Backward Pass Complexity
- Conv backward is hardest: weight gradients are correlation, input gradients need 180° filter rotation
- Must save forward pass intermediates: layer inputs, ReLU masks, max pool indices
- Numerical gradient checking relaxed tolerance for deep chains: 1e-2

