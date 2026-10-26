# AIC380 Lab Exam Cheat Sheet

Part A: CIFAR-10 MLP and CNN (Assignment 2). Part B: Vectorized Logistic Regression (graded lab).
All code is copy-paste ready. Comments explain the concept.

---

## Part A: CIFAR-10 (MLP and CNN)

### A0. Formulas

| Quantity | Formula | Example |
|---|---|---|
| Dense params | inputs x units + units | 3072x128+128 = 393,344 |
| Conv2D params | (kh x kw x C_in + 1) x filters | (3x3x3+1)x32 = 896 |
| Conv output (valid) | (n - k)//stride + 1 | (32-3)//1+1 = 30 |
| Conv output (same, stride 1) | n | 32 stays 32 |
| MaxPooling2D(2) | n//2 (odd sizes floor) | 13 -> 6 |
| Flatten | H x W x C | 6x6x64 = 2304 |
| Steps per epoch | ceil(N_train / batch) | ceil(45000/128) = 352 |

- Pooling, Flatten, activations: 0 params.
- Softmax: e^zi / sum(e^zj). Loss = -log(p of true class).
- Gradient descent: w = w - lr * dL/dw.
- Logistic regression on CIFAR (Input(3072) -> Dense(10, softmax)) = 3072x10+10 = 30,730 params.

### A1. Task 1: Data preparation

```python
# 1a. Labels come as (N,1); loss and np.bincount want a 1D vector (N,)
y_train = y_train.flatten()
y_test  = y_test.flatten()

# 1b. uint8 0-255 -> float32 0-1 (small floats train better)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 1c. Images per class (np.bincount(y)[i] = how many labels equal i)
for name, c in zip(class_names, np.bincount(y_train)):
    print(f"{name:<12} {c}")          # 5000 each -> balanced

# 1d. Flat copies for the MLP. -1 lets NumPy infer 32*32*3 = 3072
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat  = x_test.reshape(x_test.shape[0], -1)
```

### A2. Task 2: Baseline MLP

```python
results = []   # every run_experiment call appends one row here

# 2.1 Functional API: layer = Layer(...)(previous_tensor)
inputs  = keras.Input(shape=(3072,))                      # one flat image
x       = layers.Dense(512, activation="relu")(inputs)    # 3072*512+512 params
x       = layers.Dense(256, activation="relu")(x)         # 512*256+256 params
outputs = layers.Dense(10, activation="softmax")(x)       # 10 class probabilities
mlp = keras.Model(inputs, outputs, name="mlp")
mlp.summary()

# 2.2 The MLP uses the FLAT data
mlp_hist = run_experiment("MLP baseline", mlp, x_train_flat, y_train, results,
                          epochs=10, batch_size=128)

# 2.3 Params by hand: Dense = inputs*units + units (weights + biases)
first_layer_by_hand = 3072 * 512 + 512                               # 1,573,376
total_by_hand = (3072*512 + 512) + (512*256 + 256) + (256*10 + 10)   # 1,707,274
print("first layer:", first_layer_by_hand, "| total:", total_by_hand,
      "| Keras says:", mlp.count_params())                           # must match

# 2.4 Gap between train and validation curves = overfitting
plot_history(mlp_hist, "accuracy")
plot_history(mlp_hist, "loss")
```

Why 2.5 (flattening is bad): flattening destroys 2D neighbourhoods, and a shifted object lands on different input neurons, so the network relearns the same pattern at every position.

### A3. Task 3: First CNN

```python
# 3.1 Input keeps the 2D shape (32,32,3), NOT flattened
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)   # 32 filters of 3x3: edges, textures
x = layers.MaxPooling2D(2)(x)                         # halves height and width
x = layers.Conv2D(64, 3, activation="relu")(x)        # combines edges into shapes
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)                               # feature maps -> 1D vector
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
cnn = keras.Model(inputs, outputs, name="cnn")

# 3.2 Predict shapes BEFORE running. valid conv: (n-k)//s+1. Pool: n//2
shape_predictions = [
    (30, 30, 32),   # Conv2D(32,3):  32-3+1
    (15, 15, 32),   # MaxPool:       30//2
    (13, 13, 64),   # Conv2D(64,3):  15-3+1
    (6, 6, 64),     # MaxPool:       13//2 (floors)
    (2304,),        # Flatten:       6*6*64
    (128,),         # Dense(128)
    (10,),          # Dense(10)
]
check_shape_predictions(cnn, shape_predictions)
cnn.summary()

# 3.3 The CNN uses the 4D images (x_train, not flat)
cnn_hist = run_experiment("CNN baseline", cnn, x_train, y_train, results,
                          epochs=10, batch_size=128)
plot_history(cnn_hist, "accuracy")
plot_history(cnn_hist, "loss")
pd.DataFrame(results)
```

Parameter table for this CNN:

| Layer | Output shape | Params |
|---|---|---|
| Conv2D(32,3) | (30,30,32) | 896 |
| MaxPool | (15,15,32) | 0 |
| Conv2D(64,3) | (13,13,64) | 18,496 |
| MaxPool | (6,6,64) | 0 |
| Flatten | (2304,) | 0 |
| Dense(128) | (128,) | 295,040 |
| Dense(10) | (10,) | 1,290 |
| Total | | 315,722 |

3.5 (why fewer params): weight sharing. One small filter slides over the whole image, so params depend on filter size and count, not image size, and the filters detect edges and textures wherever they appear.

### A4. Task 4: Two variants, one change each

```python
# Version 2
# Change: padding="same" in both Conv2D layers.
# Expectation: keeps edge pixels and bigger maps (32->16->8), slightly higher accuracy, more params.
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)   # 32x32x32
x = layers.MaxPooling2D(2)(x)                                         # 16x16x32
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)        # 16x16x64
x = layers.MaxPooling2D(2)(x)                                         # 8x8x64
x = layers.Flatten()(x)                                               # 4096
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
cnn_v2 = keras.Model(inputs, outputs, name="cnn_v2")
cnn_v2_hist = run_experiment("CNN v2: padding same", cnn_v2, x_train, y_train, results,
                             epochs=10, batch_size=128)
plot_history(cnn_v2_hist, "accuracy")
plot_history(cnn_v2_hist, "loss")

# Version 3
# Change: add a third Conv2D(128)+MaxPooling block.
# Expectation: learns more complex features; Flatten shrinks to 512, so fewer Dense params.
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)   # 30x30x32
x = layers.MaxPooling2D(2)(x)                         # 15x15x32
x = layers.Conv2D(64, 3, activation="relu")(x)        # 13x13x64
x = layers.MaxPooling2D(2)(x)                         # 6x6x64
x = layers.Conv2D(128, 3, activation="relu")(x)       # 4x4x128
x = layers.MaxPooling2D(2)(x)                         # 2x2x128
x = layers.Flatten()(x)                               # 512
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
cnn_v3 = keras.Model(inputs, outputs, name="cnn_v3")
cnn_v3_hist = run_experiment("CNN v3: third conv block", cnn_v3, x_train, y_train, results,
                             epochs=10, batch_size=128)
plot_history(cnn_v3_hist, "accuracy")
plot_history(cnn_v3_hist, "loss")

pd.DataFrame(results)

# 4.3 Choose by VALIDATION accuracy, never train or test accuracy
df = pd.DataFrame(results)
cnn_rows = df[df["name"].str.contains("CNN")]                       # the 3 CNNs only
best_name = cnn_rows.loc[cnn_rows["val_acc"].idxmax(), "name"]      # row label of max, then its name
models = {"CNN baseline": cnn,
          "CNN v2: padding same": cnn_v2,
          "CNN v3: third conv block": cnn_v3}
best_model = models[best_name]
print(cnn_rows[["name", "val_acc", "best_val_acc", "best_epoch", "train_acc"]])
```

4.4 (overfitting): training accuracy keeps rising while validation accuracy plateaus or drops and validation loss rises, so the model is memorising. Stopping at the best epoch keeps the weights that generalise best.

### A5. Task 5: Final evaluation

```python
# 5.1 The test set is used ONLY here
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)        # compare with val_acc in the table

# 5.2 predict -> (10,10) probabilities; argmax(axis=1) = most likely class per image
pred_classes = best_model.predict(x_test[:10], verbose=0).argmax(axis=1)
correct = 0
for p, t in zip(pred_classes, y_test[:10]):
    print(f"predicted: {class_names[p]:<12} true: {class_names[t]}")
    correct += (p == t)
print(f"{correct}/10 correct")

# 5.3 Mistakes among the first 1000 test images -> error rate = mistakes / 1000
n_mistakes = show_mistakes(best_model, x_test, y_test, class_names, n=8, search=1000)
print("Mistakes:", n_mistakes, "| error rate:", n_mistakes / 1000)
```

- 5.1: test and validation accuracy should be close (within 1-2 points). A big gap suggests overfitting to the validation set or a data mismatch.
- 5.3: typical mistakes are cat vs dog, automobile vs truck, deer vs horse. Name the two you actually see.

### A6. Quick written answers

- 1.2a: axes are (images, height, width, channels); 32x32x3 = 3072.
- 1.2b: labels were (N,1) but the loss and bincount need (N,); the classes are balanced at 5,000 each.
- 3.4: compare params, validation accuracy and training accuracy from your own table. Typically the CNN has about 5x fewer params and about 15 points higher validation accuracy.

### A7. Traps

- Flat data goes to the MLP, 4D images go to the CNN.
- Build a fresh untrained model before each run_experiment call.
- Do not forget the + units (bias) in param counts.
- Choose best_model by validation accuracy. The test set is only for Task 5.
- Restart and Run All before submitting. File name: `AIC380_A2_<roll number>.ipynb`.

---

## Part B: Vectorized Logistic Regression (graded lab, 20 marks, 25 minutes)

Rules: NumPy vectorization only, no loops, no sklearn model, no hard-coded answers. Print every result with a label.

### B1. Equations

| Step | Formula |
|---|---|
| Forward | z = X @ w + b ; p = sigmoid(z) |
| Decision | pred = 1 if p >= 0.5 else 0 |
| Cost | J = -mean[ y log(p) + (1-y) log(1-p) ] |
| Gradients | error = p - y ; dw = X.T @ error / m ; db = mean(error) |
| Update | w_new = w - alpha*dw ; b_new = b - alpha*db |

### B2. Full solution

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

X = np.array([[1., 0., 2.],
              [0., 1., 1.],
              [2., 1., 0.],
              [1., 2., 1.]])
y = np.array([1., 0., 1., 0.])
w = np.array([0.2, -0.3, 0.1])
b = 0.0
alpha = 0.4

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m = len(y)   # 4 examples

# Task 1: shapes
print("Shape of X   :", X.shape)         # (4, 3)
print("Shape of w   :", w.shape)         # (3,)
print("Shape of X@w :", (X @ w).shape)   # (4,)

# Task 2: forward pass
z = X @ w + b                       # b is a scalar, broadcast onto all 4 scores
p = sigmoid(z)
pred = (p >= 0.5).astype(int)
print("Shape of z:", z.shape, "| Shape of p:", p.shape)   # (4,) and (4,)
print("z    =", z)
print("p    =", p)
print("pred =", pred)

J_before = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print("J_before =", J_before)

correct = np.sum(pred == y)         # True counts as 1, no loop
print("Correct predictions:", correct, "out of", m)

# Task 3: one gradient-descent update
error = p - y                       # (4,)
dw = X.T @ error / m                # (3,4) @ (4,) -> (3,), same shape as w
db = np.mean(error)                 # scalar
print("error =", error)
print("dw    =", dw, "| shape:", dw.shape)
print("db    =", db)

# Compute BOTH gradients before changing w or b (simultaneous update)
w_new = w - alpha * dw
b_new = b - alpha * db
print("w_new =", w_new)
print("b_new =", b_new)

p_new = sigmoid(X @ w_new + b_new)
pred_new = (p_new >= 0.5).astype(int)
J_after = -np.mean(y * np.log(p_new) + (1 - y) * np.log(1 - p_new))
print("p_new    =", p_new)
print("pred_new =", pred_new)
print("J_after  =", J_after)
print("J_after < J_before:", J_after < J_before)

# Task 4: silent broadcasting bug
y_bad = y.reshape(-1, 1)            # (4,1) column
wrong_error = p - y_bad             # predicted shape: (4, 4)
print("wrong_error shape:", wrong_error.shape)

correct_error = p - y               # (4,) - (4,) -> (4,)
print("correct_error shape:", correct_error.shape)
```

### B3. Expected output (verified by running the code)

| Item | Value |
|---|---|
| z | [0.4, -0.2, 0.1, -0.3] |
| p | [0.5987, 0.4502, 0.5250, 0.4256] |
| pred | [1, 0, 1, 0] |
| J_before | 0.5775 |
| Correct | 4 out of 4 |
| error | [-0.4013, 0.4502, -0.4750, 0.4256] |
| dw | [-0.2314, 0.2066, 0.0183] |
| db | -0.00015 |
| w_new | [0.2926, -0.3826, 0.0927] |
| b_new | about 0.00006 |
| p_new | [0.6173, 0.4280, 0.5505, 0.4061] |
| pred_new | [1, 0, 1, 0] |
| J_after | 0.5398 |
| wrong_error shape | (4, 4) |

### B4. Written answers

- **Why can scalar b be added to X @ w?** Broadcasting stretches the scalar across all 4 elements, so the same bias is added to every example.
- **Why must dw match the shape of w?** Each weight needs exactly one gradient, and the update `w - alpha*dw` needs matching shapes. X.T @ error gives (3,4) @ (4,) -> (3,).
- **Cost fell but predictions did not change?** Predictions only depend on which side of 0.5 each probability is. The update moved every probability toward its correct label (alert ones up, normal ones down) without crossing 0.5, so the cross-entropy fell while the classes stayed the same.
- **Shape of wrong_error?** (4, 4). p is (4,), treated as (1,4); y_bad is (4,1); NumPy stretches both to (4,4), so entry [i,j] = p[j] - y[i], comparing every prediction with every label.
- **Corrected expression?** `error = p - y`, or `p - y_bad.ravel()`, which gives shape (4,).
- **Why dangerous?** No exception is raised. The code returns a plausible array of the wrong shape, and later steps can silently compute wrong gradients and wrong cost, so training goes wrong without any warning. Always check shapes after a reshape.

### B5. Exam tips

- Memorise: `error = p - y`, `dw = X.T @ error / m`, `db = np.mean(error)`.
- Compute dw and db before updating w or b.
- Use `np.sum(pred == y)` for the accuracy count.
- Print every item with a label. The output checklist is marked.
