# Scream Detection and Classification for Heavy Metal vocals

The objective of this project is to create a dataset of heavy metal music with labelled vocal types, and then detect and classify the vocals into sign vs scream vs no vocal present

Work has been divided into the following stages:
- Data fetch and cleaning 
  - Dataset is built off youtube audio releases from the artist's official channels
- Data annotation (done using SonicVisualizer)
- Preprocessing:
  - Apply Spleeter source separation to extract vocal track
  - Extract VGGish features
- Feature Extraction:
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Contrast
  - Spectral Flatness
  - Spectral Roll-off
- kNN classifier using MFCC and delta MFCCs
- SVM and RF classifiers using the extracted features listed above features along with MFCCs and delta MFCCs

Interesting insights:
- Neither Spleeter nor Demucs consistently work well with heavy metal screams, often adding a large number of perceivable artifacts
