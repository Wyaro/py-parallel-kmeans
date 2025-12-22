# data/validation.py

def validate_dataset(dataset):
    meta = dataset.metadata[next(iter(dataset.metadata))]

    assert dataset.X.shape[0] == meta["N"]
    assert dataset.X.shape[1] == meta["D"]
    assert dataset.initial_centroids.shape == (meta["K"], meta["D"])
