import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SimpleKMeans:
    def __init__(self, clusters=2, method='k-means++', attempts=5,
                 max_steps=300, threshold=1e-4, seed=None):
        self.clusters = clusters
        self.method = method
        self.attempts = attempts
        self.max_steps = max_steps
        self.threshold = threshold
        self.random_gen = np.random.RandomState(seed)
        self.centroids = None
        self.group_ids = None
        self.total_variance = None
        self.total_iterations = 0

    def _initialize_plus(self, data):
        n_points, n_dims = data.shape
        centers = np.empty((self.clusters, n_dims), dtype=data.dtype)
        centers[0] = data[self.random_gen.choice(n_points)]
        distances = np.full(n_points, np.inf)

        for c in range(1, self.clusters):
            distances = np.minimum(distances, np.sum((data - centers[c-1])**2, axis=1))
            probs = distances / distances.sum()
            centers[c] = data[self.random_gen.choice(n_points, p=probs)]

        return centers

    def _run_lloyd(self, data, centers):
        for step in range(self.max_steps):
            dists = np.sum((data[:, None] - centers) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            updated_centers = np.empty_like(centers)
            for j in range(self.clusters):
                points = data[labels == j]
                if points.any():
                    updated_centers[j] = points.mean(axis=0)
                else:
                    updated_centers[j] = data[self.random_gen.choice(data.shape[0])]

            if np.sum((updated_centers - centers) ** 2) <= self.threshold:
                break

            centers = updated_centers

        final_distances = np.sum((data[:, None] - centers) ** 2, axis=2)
        final_labels = np.argmin(final_distances, axis=1)
        inertia = np.sum(final_distances[np.arange(len(final_distances)), final_labels])

        return centers, final_labels, inertia, step + 1

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        lowest_inertia = np.inf

        for _ in range(self.attempts):
            centers = (self._initialize_plus(data)
                       if self.method == 'k-means++'
                       else data[self.random_gen.choice(data.shape[0], self.clusters, False)])

            centers, labels, inertia, iterations = self._run_lloyd(data, centers)

            if inertia < lowest_inertia:
                self.centroids = centers
                self.group_ids = labels
                self.total_variance = inertia
                self.total_iterations = iterations
                lowest_inertia = inertia

        return self

    def predict(self, data):
        data = np.asarray(data, dtype=np.float64)
        distances = np.sum((data[:, None] - self.centroids)**2, axis=2)
        return np.argmin(distances, axis=1)

def isolate_green_number(image_path, output_path):
    original = cv2.imread(image_path)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

    a_vals = lab[:, :, 1].astype(np.float64)
    a_vals = (a_vals - a_vals.mean()) / a_vals.std()
    flat_data = a_vals.reshape(-1, 1)

    model = SimpleKMeans(clusters=2, method='k-means++', attempts=10, seed=42)
    model.fit(flat_data)
    labels = model.predict(flat_data).reshape(lab.shape[:2])

    result_img = np.zeros_like(rgb)
    digit_cluster = np.argmax(model.centroids)
    result_img[labels == digit_cluster] = [0, 255, 0]

    # Save result
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    # Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1), plt.imshow(rgb), plt.title("Original")
    plt.subplot(1, 2, 2), plt.imshow(result_img), plt.title("Digits in Green")
    plt.tight_layout()
    plt.show()

    print(f"Saved: {output_path}")
    print("Centers:", model.centroids.flatten())
    print("Sizes:", [np.sum(labels == i) for i in range(2)])
    print("Inertia:", round(model.total_variance, 2))
    print("Iterations:", model.total_iterations)
    print()

# Create output directory
input_dir = 'Images'
output_dir = 'Results'
os.makedirs(output_dir, exist_ok=True)

# Process all images
for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, f'green_{fname}')
        isolate_green_number(input_path, output_path)
