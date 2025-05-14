# 👶 New Born — Name Nationality Classifier

Predict the **nationality/language origin** of a given name using a Recurrent Neural Network (GRU). This project showcases preprocessing text data, one-hot encoding characters, and building a GRU-based classifier in Keras with AUC evaluation.

---

## 📁 Dataset Overview

### Training Data

- Text files stored in `data/train/`, each file corresponds to a language (e.g., `English.txt`, `French.txt`)
- Each file contains one name per line
- Constructed into a DataFrame with:
  - `Name`: the actual name
  - `Origin`: the corresponding nationality/language

### Test Data

- `test.csv` containing a list of names to predict their origin.

---

## 🛠️ Preprocessing Steps

- Extract all unique characters used across names
- One-hot encode each name character-by-character
- Pad all encoded names to the length of the longest name (20 characters)
- One-hot encode the labels (18 nationality classes)

---

## 🧠 Model Architecture

**Input Shape**: `(max_length, num_unique_chars)`

**Model**:
```python
model = Sequential()
model.add(Input(shape=(max_length, num_chars)))
model.add(GRU(256))
model.add(Dense(18, activation='softmax'))
```
## 🔧 Model Configuration

- **Loss Function**: `categorical_crossentropy`  
- **Metric**: `AUC`  
- **Optimizer**: `Adam`  
- **Epochs**: `120`  

---

## 📈 Training Results

- Achieved **Validation AUC ≈ 0.98**
- Gradual reduction in validation loss and increase in AUC confirms convergence

---

## 🧪 Evaluation

- **Metric 1**: ROC AUC  
- **Metric 2**: F1 Score (Weighted Average)

### ✅ Benchmark Goals

- ROC AUC ≥ **0.93**  
- F1 Score ≥ **0.82**

---

## 📤 Submission Format

The `submission.csv` includes:

```plaintext
```bash
submission.csv
y_pred.npy
new_born.ipynb
result.zip
prediction
1
10
14
```
## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

