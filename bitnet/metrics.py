import enum

class Metrics(enum.Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"

    def __str__(self):
        return self.value
