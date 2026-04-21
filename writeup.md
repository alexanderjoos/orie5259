# PBV Strategy Train of Thought

## Iterations 1-2

`pbv.ipynb` began with raw level-by-level d1/d2 signals, and the follow-up composite scoring idea tried to combine levels 1 to 5 into one pressure score, but neither was strong enough as a final rule: d1 was only mildly useful, d2 was noisy, and the composite version often fired too late or leaned on fallback execution. The bucket-vector state was the first piece that felt genuinely useful, since summarizing the book into near/far pressure produced a cleaner representation, and the later logistic regression model made better use of those features than hand-tuned triggers; Order Book Imbalance was tested too, but it added little beyond the existing PBV state.

## Iteration 5

In the causal `v5` stage, the main lesson was that the model seemed to generalize more to within-minute drift and timing structure than to a clean PBV-specific edge. So the honest conclusion was to keep the useful feature-design lessons from PBV, but move forward with other models rather than retain PBV itself as the final strategy.
