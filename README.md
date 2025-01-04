# transformer
A minimal decoder transformer implementation

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