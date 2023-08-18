import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Load the trained model and vectorizer
model_filename = "text_identification_model.pkl"
vectorizer_filename = "text_identification_vectorizer.pkl"
loaded_classifier = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

# Create a sample text
sample_text = "קיץ בריא ונעים חברים!"

# Transform the sample text using the vectorizer
sample_text_tfidf = vectorizer.transform([sample_text])

# Make predictions using the loaded model
predicted_class = loaded_classifier.predict(sample_text_tfidf)

# Visualize the decision boundaries (example with a simple 2D dataset)
# Modify this part according to your data and model
# For complex data, consider using libraries like plotly
# to create more informative visualizations

# Generate data for visualization
X_visual = np.random.rand(300, 2) * 10
y_visual = np.random.randint(0, 3, size=300)

# Train an SVM on the generated data
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_visual, y_visual)

# Plot the data points
plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_visual, cmap=plt.cm.Paired)

# Plot the decision boundaries
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))

Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Highlight the support vectors
plt.scatter(svm_classifier.support_vectors_[:, 0],
            svm_classifier.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')

# Plot the predicted sample point
plt.scatter(sample_text_tfidf[0, 0], sample_text_tfidf[0, 1], marker='x', color='red', label=f'Predicted Class: {predicted_class[0]}')

plt.title('Support Vector Machine Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
