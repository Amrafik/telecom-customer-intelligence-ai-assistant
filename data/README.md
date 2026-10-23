# Data provenance and feature dictionary

`raw/customer_churn.csv` is the unmodified UCI Iranian Churn CSV, downloaded by `python -m telecom.prepare_data`. Attribution: **Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z**. The repository publishes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset. Its checksum is recorded in `provenance.json` and the benchmark. The data contains 3,150 anonymised rows. Features aggregate the first nine months; the churn label is measured at month twelve, leaving a three-month planning gap.

Training transformations are in `telecom/ml.py`; the original CSV is retained. All app customer records in `customers.json` were independently authored and are **fictional**, with no mapping to UCI subjects. `faqs.json` contains eight **fictional DemoTel policies**, including invented plan prices. Neither file contains real Vodafone information or offers. No affiliation is implied.

`python -m telecom.init_db` validates that seed JSON and inserts it into `customers.sqlite3`. The running application reads SQLite, not the seed JSON. Customer profiles and usage metrics are stored in two related tables; lookup uses a parameterized INNER JOIN. Seeding is idempotent and preserves existing rows. To edit a demo record after seeding, update SQLite explicitly or back up and remove only the generated database before seeding again. The database is a rebuildable local artifact and is excluded from Git.

| App feature | Source field | Meaning |
|---|---|---|
| call_failures | Call Failure | Failed calls in observation period |
| complaints | Complains | 0 → no, 1 → yes |
| subscription_months | Subscription Length | Months of subscription |
| charge_band | Charge Amount | Ordinal band 0–9; **not currency** |
| seconds_of_use | Seconds of Use | Total call duration in seconds |
| calls | Frequency of use | Call count |
| sms | Frequency of SMS | Text message count |
| distinct_called_numbers | Distinct Called Numbers | Number of distinct called numbers |
| tariff_plan | Tariff Plan | 1 → pay_as_you_go, 2 → contract |
| churn (target only) | Churn | 0 → retained, 1 → churned |

The two categorical features are one-hot encoded. Numeric features are median-imputed and standardised inside the fitted pipeline. The public file has no documented missing values, but the pipeline supports missing numeric values. Unknown source categories fail validation; OneHotEncoder also handles future unknown categories at inference.

Age and Age Group are excluded to keep the demo small and avoid direct age-based predictions; this is **not** proof of fairness. Status is excluded conservatively because inactivity may proxy churn. Customer Value is excluded because its derivation is unspecified. Customer IDs, display names and target labels never enter the model. No customer-specific causal reasons are inferred from a probability.

Identical retained feature rows, including conflicting labels, share a group. The first fold of a seeded five-fold stratified group split is the held-out test set. Three-fold stratified group cross-validation runs only on the remaining training set. The final saved pipeline is the best CV estimator fitted to the training partition; test rows are never refitted. Metrics are for this public Iranian sample, not for Egypt, Vodafone customers, or the fictional app population. Scores are not calibrated for deployment.
