import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up the plot style
plt.style.use('seaborn-v0_8')
encoder = LabelEncoder()

# Load and prepare data
df = pd.read_csv('Iris.csv')
df = df.drop(columns='Id')

# Store original species names for plotting
species_names = df['Species'].unique()
df['Species'] = encoder.fit_transform(df['Species'])

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = 'Species'

x = df[features]
y = df[target]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f"Model Accuracy: {score:.4f}")


# Function to plot decision boundaries
def plot_decision_boundary(X, y, model, feature_indices, feature_names, ax, title):
    """Plot 2D decision boundary for given feature pair"""
    # Convert training data to numpy arrays to avoid feature name warnings
    X_train_np = x_train.values
    y_train_np = y_train.values

    # Use only the specified features
    X_subset = X_train_np[:, feature_indices]

    # Create a mesh to plot the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X_subset[:, 0].min() - 0.5, X_subset[:, 0].max() + 0.5
    y_min, y_max = X_subset[:, 1].min() - 0.5, X_subset[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # For prediction, we need all 4 features, so we'll use the mean of other features
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Create full feature vectors by adding mean values for missing features
    full_mesh_points = np.zeros((mesh_points.shape[0], 4))
    full_mesh_points[:, feature_indices] = mesh_points

    # Fill missing features with training data means
    for i in range(4):
        if i not in feature_indices:
            full_mesh_points[:, i] = X_train_np[:, i].mean()

    # Get predictions
    Z = model.predict(full_mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, colors='black', linestyles='--', linewidths=0.5)

    # Plot the data points
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        idx = np.where(y_train_np == i)
        ax.scatter(X_subset[idx, 0], X_subset[idx, 1],
                   c=color, marker='o', s=50, alpha=0.8,
                   label=f'{species_names[i]}', edgecolors='black', linewidth=0.5)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# Convert to numpy arrays for easier indexing
X_train_np = x_train.values
X_test_np = x_test.values
X_full_np = x.values
y_full_np = y.values

# Plot 1: Most informative feature pairs
feature_pairs = [
    ([2, 3], ['Petal Length', 'Petal Width']),  # Most separable
    ([0, 2], ['Sepal Length', 'Petal Length']),
    ([1, 3], ['Sepal Width', 'Petal Width']),
    ([0, 1], ['Sepal Length', 'Sepal Width'])
]

for i, (indices, names) in enumerate(feature_pairs):
    ax = plt.subplot(3, 3, i + 1)
    plot_decision_boundary(X_full_np, y_full_np, model, indices, names, ax,
                           f'SVM Decision Boundary\n{names[0]} vs {names[1]}')

# Plot 5: Support Vectors visualization (using most separable features)
ax5 = plt.subplot(3, 3, 5)
X_subset = X_train_np[:, [2, 3]]  # Petal length and width
support_vectors = model.support_vectors_[:, [2, 3]]

# Plot decision boundary
h = 0.02
x_min, x_max = X_subset[:, 0].min() - 0.5, X_subset[:, 0].max() + 0.5
y_min, y_max = X_subset[:, 1].min() - 0.5, X_subset[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

mesh_points = np.c_[xx.ravel(), yy.ravel()]
full_mesh_points = np.zeros((mesh_points.shape[0], 4))
full_mesh_points[:, [2, 3]] = mesh_points
full_mesh_points[:, 0] = X_train_np[:, 0].mean()
full_mesh_points[:, 1] = X_train_np[:, 1].mean()

# For multi-class SVM, we need to handle decision function differently
try:
    Z_decision = model.decision_function(full_mesh_points)
    if Z_decision.ndim > 1:
        # Multi-class: use the class with highest decision value
        Z = np.argmax(Z_decision, axis=1)
    else:
        # Binary classification
        Z = (Z_decision > 0).astype(int)
    Z = Z.reshape(xx.shape)
except:
    # Fallback to predict if decision_function fails
    Z = model.predict(full_mesh_points)
    Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
ax5.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=np.arange(-0.5, 3.5, 1))
ax5.contour(xx, yy, Z, colors='black', levels=[0.5, 1.5], linestyles='-', linewidths=1)

# Plot training data
colors = ['red', 'blue', 'green']
for i, color in enumerate(colors):
    idx = np.where(y_train == i)
    ax5.scatter(X_subset[idx, 0], X_subset[idx, 1],
                c=color, marker='o', s=50, alpha=0.6,
                label=f'{species_names[i]}', edgecolors='black', linewidth=0.5)

# Highlight support vectors
ax5.scatter(support_vectors[:, 0], support_vectors[:, 1],
            s=200, facecolors='none', edgecolors='red', linewidths=3, label='Support Vectors')

ax5.set_xlabel('Petal Length (cm)')
ax5.set_ylabel('Petal Width (cm)')
ax5.set_title('SVM Support Vectors & Margins\n(Training Data)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Model performance metrics
ax6 = plt.subplot(3, 3, 6)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=species_names, yticklabels=species_names)
ax6.set_title('Confusion Matrix\n(Test Data)')
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')

# Plot 7: Feature importance (coefficients)
ax7 = plt.subplot(3, 3, 7)
feature_importance = np.abs(model.coef_).mean(axis=0)
bars = ax7.bar(range(len(features)), feature_importance,
               color=['skyblue', 'lightgreen', 'salmon', 'gold'])
ax7.set_xticks(range(len(features)))
ax7.set_xticklabels(['Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth'])
ax7.set_title('Feature Importance\n(Average |Coefficients|)')
ax7.set_ylabel('Importance')
ax7.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, importance in zip(bars, feature_importance):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{importance:.3f}', ha='center', va='bottom')

# Plot 8: Training vs Test performance
ax8 = plt.subplot(3, 3, 8)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

scores = [train_score, test_score]
labels = ['Training', 'Test']
bars = ax8.bar(labels, scores, color=['lightblue', 'lightcoral'])
ax8.set_ylim([0, 1.1])
ax8.set_title('Model Performance')
ax8.set_ylabel('Accuracy')

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Decision function values distribution
ax9 = plt.subplot(3, 3, 9)
try:
    decision_values = model.decision_function(x_test)
    if decision_values.ndim > 1:  # Multi-class case
        # Plot histogram for each class's decision scores
        for i in range(decision_values.shape[1]):
            ax9.hist(decision_values[:, i], alpha=0.6,
                     label=f'Class {i} scores', bins=10)
        ax9.set_xlabel('Decision Function Scores')
        ax9.set_title('Decision Function Scores\n(Test Data)')
    else:
        # Binary case
        for i in range(len(species_names)):
            mask = y_test == i
            if np.sum(mask) > 0:
                ax9.hist(decision_values[mask], alpha=0.6,
                         label=f'{species_names[i]}', bins=10)
        ax9.set_xlabel('Decision Function Value')
        ax9.set_title('Decision Function Values\n(Test Data)')
except:
    # Fallback: show prediction confidence
    y_pred_proba = model.predict(x_test)
    for i in range(len(species_names)):
        mask = y_test == i
        if np.sum(mask) > 0:
            ax9.hist(y_pred_proba[mask], alpha=0.6,
                     label=f'{species_names[i]}', bins=3)
    ax9.set_xlabel('Predicted Class')
    ax9.set_title('Predictions Distribution\n(Test Data)')

ax9.set_ylabel('Frequency')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed results
print(f"\nDetailed Results:")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Number of Support Vectors: {len(model.support_)}")
print(f"Support Vectors per class: {model.n_support_}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=species_names))

print(f"\nModel Parameters:")
print(f"Kernel: {model.kernel}")
print(f"C (Regularization): {model.C}")
print(f"Gamma: {model.gamma}")