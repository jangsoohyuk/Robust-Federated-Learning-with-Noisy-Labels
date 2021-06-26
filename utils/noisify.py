import random


def noisify_label(true_label, num_classes=10, noise_type="symmetric"):
    if noise_type == "symmetric":
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return (true_label - 1) % num_classes
