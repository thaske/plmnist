from ray import tune

from plmnist.config import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DROPOUT_PROB,
    SEED,
)

# define the search space for the hyperparameter tuning
parameter_space = dict(
    batch_size=BATCH_SIZE,
    hidden_size=tune.qlograndint(16, 64, 16),
    learning_rate=tune.qloguniform(1e-8, 1e-1, 1e-8),
    dropout_prob=tune.quniform(0.0, 0.7, 0.1),
    seed=SEED,
)

# define the default configuration for the training process
#   note that this must be within the search space defined above
default_config = dict(
    batch_size=BATCH_SIZE,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_prob=DROPOUT_PROB,
    seed=SEED,
)
