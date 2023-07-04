class BaseMetric:
    def __init__(self):
        pass

    def evaluate(self):
        raise NotImplementedError

    def evaluate_all(self):
        raise NotImplementedError

    def evaluate_batch(self, sample, output):
        raise NotImplementedError
