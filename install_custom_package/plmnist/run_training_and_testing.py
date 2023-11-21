import argparse

import pytorch_lightning as pl

from plmnist import train, test, write

from fgsm import fgsm_from_path, plot_fgsm

from config import (
    NUM_EPOCHS,
    LOG_PATH,
    RESULT_PATH,
    DATA_PATH,
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DROPOUT_PROB,
    SEED,
    FGSM_EPSILON,
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--log_path", type=str, default=LOG_PATH)
    parser.add_argument("--result_path", type=str, default=RESULT_PATH)
    parser.add_argument("--data_dir", type=str, default=DATA_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout_prob", type=float, default=DROPOUT_PROB)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fgsm_epsilon", type=float, default=FGSM_EPSILON)

    parser.add_argument("--no_dhash", dest="do_dhash", action="store_false")
    parser.add_argument("--no_fgsm", dest="do_fgsm", action="store_false")
    args = parser.parse_args()

    # first part -- train/test

    pl.seed_everything(args.seed)

    trainer, model = train(
        max_epochs=args.num_epochs,
        log_path=args.log_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
    )

    results = test(trainer, args.seed)

    dhash = write(results, trainer, directory=args.result_path, do_dhash=args.do_dhash)

    if args.do_fgsm:
        # separate part -- fgsm
        # only needs args.result_path and args.seed (and dhash if used) from above
        # you can get seed/dhash from the results json
        # if dhash is not used, it should be replaced with an empty string
        pl.seed_everything(args.seed)

        fgsm_results = fgsm_from_path(args.result_path, args.fgsm_epsilon, dhash)

        fig_path = f"{args.result_path}/fgsm{dhash}.png"
        first_fgsm = fgsm_results[1][0]
        plot_fgsm(*first_fgsm, args.fgsm_epsilon, fig_path)
