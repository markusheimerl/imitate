# transformer
A minimal decoder transformer implementation

SCALE! Let transformers do what transformers do best

### Strategies for Improving Finite Differences Training

1. **Efficient Gradient Estimation**:
   - Consider using **central differences** instead of forward differences for better accuracy. Central differences use the formula: \((f(x+h) - f(x-h)) / (2h)\), often providing a more precise gradient approximation.
   
2. **Adaptive \(\epsilon\) Selection**:
   - Dynamically adjust \(\epsilon\) based on the scale of each parameter or the loss function's sensitivity. For example, scaling \(\epsilon\) relative to the magnitude of the parameter or using a small base \(\epsilon\) with multiplicative factors.

3. **Parallelization**:
   - Maximize computation efficiency by parallelizing the finite difference calculations. Given that each weight update calculation is independent, use multi-threading or GPUs to evaluate perturbations concurrently.

4. **Subsampling Parameters**:
   - Instead of updating all parameters at each step, randomly sample a subset to perturb and update. This reduces computational load per step and introduces stochasticity similar to SGD.

5. **Gradient Caching**:
   - Leverage previously computed gradients where possible, especially in configurations of parameters that change less frequently.
   
6. **Use Efficient Numerical Libraries**:
   - Optimize the numerical computation using highly efficient libraries or SIMD instructions if updating parameters in batch.

7. **Hybrid Approaches**:
   - Consider a hybrid approach where finite differences are used initially to handle complex landscapes or unknown derivatives, and then switch to a more traditional method as the landscape becomes well-understood.

8. **Line Search**:
   - Implement line search techniques with finite differences to determine optimal step sizes during weight updates, aiming to minimize loss function more efficiently rather than fixed learning rates.

9. **Sensitivity Analysis**:
   - Perform a sensitivity analysis to identify parameters that significantly affect the loss and prioritize these for more frequent updates.

Here are the key concepts and improvements for finite difference-based training, explained without code examples:

1. Adaptive Epsilon:
Instead of using a fixed epsilon value, implement an adaptive epsilon that scales with the magnitude of each weight. This improves gradient estimation accuracy, especially for weights of varying scales.

2. Central Difference Method:
Replace the forward difference method with central difference. This involves evaluating the loss function at both positive and negative perturbations, providing a more accurate gradient estimate by reducing the impact of higher-order terms in the Taylor expansion.

3. Gradient Accumulation:
To reduce noise in gradient estimates, accumulate gradients over multiple samples before applying updates. This technique averages out some of the noise inherent in finite difference approximations.

4. Gradient Smoothing:
Implement an exponential moving average (EMA) for gradients. This smooths out noisy gradient estimates over time, potentially leading to more stable training.

5. Random Perturbation Direction:
Instead of perturbing each weight individually, use random direction perturbation. This involves generating a random vector, normalizing it, and using it to perturb all weights simultaneously. This can be more efficient and sometimes provides better exploration of the loss landscape.

6. Careful Learning Rate Tuning:
With finite differences, appropriate learning rate selection is crucial. Consider implementing a learning rate schedule that decreases over time, or an adaptive learning rate method.

7. Regularization:
Incorporate regularization techniques like L2 regularization (weight decay) to prevent overfitting and improve generalization.

8. Gradient Clipping:
Implement gradient clipping to prevent extremely large updates that can destabilize training, especially important with the potential noise in finite difference estimates.

9. Batch Size Considerations:
Experiment with different batch sizes. Larger batch sizes can help reduce noise in gradient estimates but may slow down training.

10. Monitor Gradient Statistics:
Keep track of gradient magnitudes and other statistics to gain insights into the training process and potential issues.