# Paths to CIFAR-10 data files
class DataPaths:
    test_batch: str = './test_data/test_batch'
    batch_meta: str = './test_data/batches.meta'
    validation_batch: str = './test_data/data_batch_5'
    train_batches: list = [
        './test_data/data_batch_1',
        './test_data/data_batch_2',
        './test_data/data_batch_3',
        './test_data/data_batch_4'
    ]