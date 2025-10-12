from os.path import join

def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return train_dir


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return test_dir