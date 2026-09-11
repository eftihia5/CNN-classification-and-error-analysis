# CNN Digit Classification and Error Analysis: Where Does the Network Make Its Mistakes?

## 1. Project Title and Short Description
**Title:** Handwritten Digit Classification & Error Analysis on MNIST  
This project builds and evaluates a Convolutional Neural Network (CNN) to recognise handwritten digits from the MNIST dataset. Beyond measuring raw classification accuracy, it investigates where the model fails, using confusion matrices to identify primary misclassification patterns, and evaluates whether a controlled architectural change (increasing network depth) meaningfully reduces these specific errors.

---

## 2. Problem Statement
The goal is to accurately classify greyscale images of handwritten digits into 10 classes (0–9) for real-world automated pipelines (e.g., postal code reading and banking transactions). In these settings, a confident misreading carries a substantial cost (e.g., misdirected letters or funds). The central question addressed is:
> *Where does the CNN make its mistakes, and does one controlled change to the model move them?*

---

## 3. Data set
The project uses the **MNIST** dataset retrieved from OpenML:
* **Instances:** 70,000 greyscale images of handwritten digits (28×28 pixels).
* **Inputs:** Each image is represented as a 784-dimensional vector of pixel intensities (0–255).
* **Target:** Integer labels from 0 to 9.
* **Preprocessing:** Pixel intensities were normalised to the range [0.0, 1.0] by dividing by 255.0 and reshaped to `(28, 28, 1)`. The dataset was split into an 80% training set (56,000 samples) and a 20% test set (14,000 samples) using stratified sampling to preserve class distribution.

---

## 4. Method
Two CNN architectures were trained and compared under identical optimisation conditions (`Adam`, `sparse_categorical_crossentropy`, batch size 128, up to 15 epochs, 10% validation split, and `EarlyStopping` monitoring `val_loss` with patience=3):

1. **Baseline Model:**
   * `Conv2D(32, (3, 3), relu)` -> `MaxPooling2D((2, 2))` -> `Dropout(0.3)` -> `Flatten()` -> `Dense(64, relu)` -> `Dense(10, softmax)`.
2. **Deeper Model (Controlled Modification - Option A):**
   * An additional convolutional block was inserted before flattening: `Conv2D(64, (3, 3), relu)` -> `MaxPooling2D((2, 2))`.

**Evaluation Method:** Overall test accuracy/loss and row-normalised 10x10 confusion matrices to observe inter-class error shifts.

---

## 5. Results
* **Performance Metrics:**
  * **Base Model:** Test Accuracy = **98.64%** | Test Loss = **0.0486**
  * **Deeper Model:** Test Accuracy = **98.92%** | Test Loss = **0.0350**
* **Confusion Analysis (Key Error Pairs):**
  * **True 4 predicted as 9:** Reduced from **24** instances in the baseline model to **13** in the deeper model.
  * **True 2 predicted as 8:** Reduced from **10** instances in the baseline model to **7** in the deeper model.
  * **True 9 predicted as 7:** Reduced from **9** instances in the baseline model to **5** in the deeper model.

---

## 6. Interpretation & Ethical Considerations
* **Interpretation:** The errors made by the baseline model were not randomly distributed; they concentrated around structural ambiguities where stroke loops or ascenders mimic adjacent digits (e.g., an open-topped 4 mimicking a 9, or a looped 2 mimicking an 8). Adding hierarchical depth allowed the network to learn finer geometric features, nearly halving the dominant 4->9 confusion without creating new error clusters elsewhere.
* **Ethical Considerations:** Automated digit readers deployed in banking or postal sorting face severe risks when exposed to non-Western handwriting conventions (such as European crossed sevens or ones with leading strokes) or tremors common in elderly populations, which are under-represented in MNIST. Because an erroneous high-confidence reading can misdirect payments or mail, automated pipelines must implement confidence thresholding to flag borderline cases for human review.

---

## 7. Reflection
* **What Worked Well:** The controlled comparison clearly isolated the impact of network depth, demonstrating a meaningful reduction in the most prominent confusion pairs. EarlyStopping effectively prevented overfitting.
* **Limitations & Future Improvements:** With more time, incorporating **data augmentation** would help the model generalise better to irregular handwriting styles. Adding **Batch Normalization** and implementing a **reject option** would be essential steps prior to real-world deployment.
