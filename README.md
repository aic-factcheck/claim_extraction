# FEVERFact ACL Rolling Review Repository
To be refined on camera-ready once the anonymity is dropped and codes that attribute authors etc may be published.

## Datasets
See dir **feverfact**, data was obtained using attached **feverfact_dataset_generation.ipynb**.

## Models
Could not be uploaded due to size limits on GitHub and 4open.science -- will be linked via huggingface model hub for camera-ready.

## Notebooks
See dir **notebooks**, cover metrics, predictions etc.

## Results
See dir **results**, shows the claims generated via various methods, and their grading against the gold gold data using the metrics.

## Annotations
See dir **annotations**, both single- and multi-claim annotations are flattened into a string -- single claim metrics as a 4-digit number as "3311" -- first digit denotes Faithfulness, second Fluency, third Decontextualization and fourth Atomicity. For multi-claim metrics (focus, coverage), each digit (0 or 1) denotes whether the pred/gold claim with the same order as the digit has been found within the concatenation of its counterpart claims.