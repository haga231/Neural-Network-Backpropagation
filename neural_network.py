import numpy as np

print("="*60)
print("ASSIGNMENT 1 - Feed Forward and Back Propagation")
print("="*60)
print("Name: Hagar Ashraf El_Bialy")
print("ID: 4231064")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)

# Network architecture parameters
input_size = 2
hidden_size = 2
output_size = 1

print("\n NETWORK ARCHITECTURE:")
print(f"   - Input layer: {input_size} neurons")
print(f"   - Hidden layer: {hidden_size} neurons")
print(f"   - Output layer: {output_size} neuron")
print("   - Activation function: tanh")

# Initialize weights randomly from [-0.5, 0.5]
W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))

# Set biases as given
b1 = 0.5  # bias for hidden layer
b2 = 0.7  # bias for output layer

print("\n INITIAL WEIGHTS:")
print(f"   W1 (Input → Hidden):\n{W1}")
print(f"\n   W2 (Hidden → Output):\n{W2}")
print(f"\n BIASES: b1 = {b1}, b2 = {b2}")

# ============================================
# PART 1: FEED FORWARD
# ============================================
print("\n" + "="*60)
print("PART 1: FEED FORWARD PROPAGATION")
print("="*60)

# Input data
X = np.array([[0.5, 0.3]])  # 1 sample with 2 features
print(f"\n INPUT DATA: {X[0]}")

# Step 1: Input to Hidden Layer
print("\n HIDDEN LAYER CALCULATION:")
Z1 = np.dot(X, W1) + b1
print(f"   Z1 = X·W1 + b1 = {X[0]} · W1 + {b1}")
print(f"   Z1 = {Z1[0]}")

A1 = np.tanh(Z1)
print(f"   A1 = tanh(Z1) = {A1[0]}")

# Step 2: Hidden to Output Layer
print("\n OUTPUT LAYER CALCULATION:")
Z2 = np.dot(A1, W2) + b2
print(f"   Z2 = A1·W2 + b2 = {A1[0]} · W2 + {b2}")
print(f"   Z2 = {Z2[0][0]:.6f}")

A2 = np.tanh(Z2)
print(f"   A2 = tanh(Z2) = {A2[0][0]:.6f}")

# Final Output
print("\n FINAL OUTPUT:")
print(f"   Network output = {A2[0][0]:.6f}")

# ============================================
# PART 2: BACK PROPAGATION
# ============================================
print("\n" + "="*60)
print("PART 2: BACK PROPAGATION")
print("="*60)

# Target output (assumed for demonstration)
y_true = np.array([[0.8]])  # desired output
print(f"\n TARGET OUTPUT: {y_true[0][0]}")

# Calculate error
error = y_true - A2
print("\n ERROR CALCULATION:")
print(f"   Error = Target - Predicted = {y_true[0][0]:.6f} - "
      f"{A2[0][0]:.6f} = {error[0][0]:.6f}")

# Derivative of tanh: tanh'(x) = 1 - tanh²(x)
print("\n ACTIVATION DERIVATIVES:")

# Output layer gradient
dA2 = 1 - np.tanh(Z2)**2
print(f"   tanh'(Z2) = 1 - tanh²(Z2) = {dA2[0][0]:.6f}")

# Hidden layer gradient
dA1 = 1 - np.tanh(Z1)**2
print(f"   tanh'(Z1) = 1 - tanh²(Z1) = {dA1[0]}")

# Calculate gradients
print("\n GRADIENT CALCULATION:")

# Output layer gradients
delta2 = error * dA2
dW2 = np.dot(A1.T, delta2)
db2 = np.sum(delta2, axis=0, keepdims=True)

print(f"   delta2 = error × tanh'(Z2) = {delta2[0][0]:.6f}")
print(f"   dW2 = A1ᵀ · delta2 = \n{dW2}")
print(f"   db2 = {db2[0][0]:.6f}")

# Hidden layer gradients
delta1 = np.dot(delta2, W2.T) * dA1
dW1 = np.dot(X.T, delta1)
db1 = np.sum(delta1, axis=0, keepdims=True)

print(f"\n   delta1 = (delta2 · W2ᵀ) × tanh'(Z1) = \n{delta1}")
print(f"   dW1 = Xᵀ · delta1 = \n{dW1}")
print(f"   db1 = {db1[0]}")

# ============================================
# PART 3: WEIGHT UPDATE
# ============================================
print("\n" + "="*60)
print("PART 3: WEIGHT UPDATE")
print("="*60)

learning_rate = 0.1
print(f"\n LEARNING RATE: {learning_rate}")

# Update weights and biases
W1_new = W1 + learning_rate * dW1
W2_new = W2 + learning_rate * dW2
b1_new = b1 + learning_rate * db1
b2_new = b2 + learning_rate * db2

print("\n UPDATED WEIGHTS:")
print(f"   W1 (new) = W1 + α·dW1 = \n{W1_new}")
print(f"\n   W2 (new) = W2 + α·dW2 = \n{W2_new}")
print(f"\n   b1 (new) = {b1_new[0][0]:.6f}")
print(f"   b2 (new) = {b2_new[0][0]:.6f}")

# New forward pass with updated weights
print("\n NEW FORWARD PASS WITH UPDATED WEIGHTS:")

Z1_new = np.dot(X, W1_new) + b1_new
A1_new = np.tanh(Z1_new)
Z2_new = np.dot(A1_new, W2_new) + b2_new
A2_new = np.tanh(Z2_new)

print(f"   New output = {A2_new[0][0]:.6f}")
print(f"   New error = {y_true[0][0] - A2_new[0][0]:.6f}")

# Compare errors
print("\n ERROR COMPARISON:")
print(f"   Error before update: {error[0][0]:.6f}")
print(f"   Error after update:  {y_true[0][0] - A2_new[0][0]:.6f}")

if abs(y_true[0][0] - A2_new[0][0]) < abs(error[0][0]):
    print("    Error decreased! Network is learning.")
else:
    print("    Error increased. Try adjusting learning rate.")

print("\n" + "="*60)
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("="*60)
