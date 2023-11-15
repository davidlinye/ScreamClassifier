README2

Small description on how to proceed

- Pre-pre-processing should result in something like annotation.csv
- Vocal-separation.ipynb works
- Vggish.ipynb should be run next
- The vggish file will create data_with_vggisch.npy and vocal_only_data_with_vggish.npy
  [RIGHT NOW: generate vocal_only_data_with_vggish.npy]
- vocal_only_data_with_vggish.npy is used in vocal-only-classification.ipynb. \* This notebook results in vocal_only_features.npy
- After this, we should be able to classify the vocal_only part.
