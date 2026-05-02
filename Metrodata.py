import kagglehub

# Download latest version
path = kagglehub.dataset_download("pooriamst/metro-interstate-traffic-volume")

print("Path to dataset files:", path)