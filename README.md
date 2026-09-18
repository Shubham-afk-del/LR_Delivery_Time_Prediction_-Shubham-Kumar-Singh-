# Delivery time prediction with linear regression

Predicting food delivery time from the order and the state of the courier fleet at the moment
it was placed. 175,777 orders, 15 columns, from the IIIT Bangalore Data Science programme.

## Results

All figures on the held-out test set (35,156 rows).

| Model | Features | MAE (min) | RMSE | R² |
|---|---|---|---|---|
| All features | 12 | 2.33 | 3.24 | 0.8800 |
| RFE, unscaled | 8 | 2.95 | 3.98 | 0.8191 |
| RFE, scaled | 8 | 2.45 | 3.24 | 0.8737 |

**Recursive Feature Elimination made the model worse.** Dropping four features cost six points
of R². The brief framed RFE as a refinement step; on this data it was a downgrade.

The gap between the two 8-feature rows is not a scaling effect. Ordinary least squares is
scale-invariant, so standardising the inputs cannot change R². The rows differ because RFE ran
separately on each matrix and **selected different features**: the scaled run took `subtotal`
and `total_busy_dashers` where the unscaled run took `num_distinct_items` and `isWeekend`.
Swapping a weekend flag for a second courier-load count recovered five of the six lost points,
which says more about RFE's instability on correlated features than about scaling.

`distance` correlates most strongly with delivery time at r = 0.4645, with an unscaled
coefficient of 0.4803, so each additional unit adds about half a minute. Three of the eight
selected features are courier-load counts: delivery time is mostly about how busy the fleet is.

## What I would do differently

Fit a gradient-boosted baseline alongside it, to find out what the linear assumption cost.
Without that number, R² 0.88 has nothing to be good or bad relative to. The brief fixing the
model is a reason I did not do it during the assignment, not a reason it should stay undone.

And sweep RFE across feature counts with the scores plotted, rather than treating it as a step
you apply once. That would have surfaced the drop immediately.

## Corrections, September 2026

Three of the written answers in the submitted notebook described **a different dataset**,
naming columns (`Discount_offered`, `Weight_in_gms`, `Warehouse_block`, `store_type`,
`region_code`, `coupon_category`) that do not exist in this data. One still carried a "replace
with the actual column name you observe" placeholder.

The code and every metric were correct. The prose written over them had not been checked
against the output. Questions 1, 3 and 5 are rewritten in `LR_Delivery_Time_Estimation.ipynb`
from the correlation table the notebook itself prints.

The version submitted for assessment is unchanged, in the zip archive in this repository.

## Files

- `LR_Delivery_Time_Estimation.ipynb` - the corrected notebook, renders on GitHub
- `LR_Delivery_Time_Prediction_<Shubham Kumar Singh>.zip` - the submitted version and report

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
