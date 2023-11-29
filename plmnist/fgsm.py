import os, json, argparse

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from plmnist.plmnist import LitMNIST
from plmnist.config import FGSM_EPSILON, RESULT_PATH, SEED


# based on https://pytorch.org/tutorials/beginner/fgsm_tutorial.html#fgsm-attack
def fgsm(model: pl.LightningModule, epsilon: float = FGSM_EPSILON):
    total, correct, adv_examples = 0, 0, []
    for batch in model.test_dataloader():
        data, target = batch
        for data_i, target_i in zip(data, target):
            total += 1

            # predict
            data_i.requires_grad = True
            logits = model(data_i.unsqueeze(0))
            init_pred = torch.argmax(logits, dim=1)[0]

            # skip if prediction is wrong
            if init_pred.item() != target_i.item():
                total -= 1
                continue

            # calculate loss and gradient
            loss = F.nll_loss(logits, target_i.unsqueeze(0))
            model.zero_grad()
            loss.backward()
            data_grad = data_i.grad.data

            # attack
            perturbed_data = torch.clamp(data_i + epsilon * data_grad.sign(), 0, 1)

            # predict on attacked image
            output = model(perturbed_data.unsqueeze(0))
            final_pred = torch.argmax(output, dim=1)[0]

            if final_pred.item() == target_i.item():
                # attack did not fool the model
                correct += 1
            else:
                adv_ex = perturbed_data.squeeze().detach().cpu()
                if len(adv_examples) < 5:
                    # save only first 5
                    adv_examples.append(
                        (
                            init_pred.item(),
                            final_pred.item(),
                            adv_ex.squeeze().numpy().tolist(),
                            data_i.squeeze().detach().numpy().tolist(),
                        )
                    )

    final_acc = correct / total
    print(
        "Epsilon: {}\tFGSM Accuracy = {} / {} = {}".format(
            epsilon, correct, total, final_acc
        )
    )

    return final_acc, adv_examples, epsilon


def add_fgsm_to_results(fgsm: tuple[float, list], results_path: str):
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

        results["fgsm"] = dict()
        results["fgsm"]["accuracy"] = fgsm[0]
        results["fgsm"]["examples"] = fgsm[1]
        results["fgsm"]["epsilon"] = fgsm[2]

        with open(results_path, "w") as f:
            json.dump(results, f)
    else:
        raise FileNotFoundError(f"File not found: {results_path}")


def plot_fgsm(init_pred, final_pred, data_i, adv_ex, epsilon, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(data_i, cmap="gray")
    axs[0].set_title(f"Original - {init_pred}")

    axs[1].imshow(adv_ex, cmap="gray")
    axs[1].set_title(f"FGSM - {final_pred}")

    fig.tight_layout()

    fig.suptitle(f"FGSM with Epsilon = {epsilon}")

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()


def fgsm_from_path(result_path: str, epsilon: float = FGSM_EPSILON, dhash: str = ""):
    ckpt_path = f"{result_path}/model{dhash}.ckpt"
    json_path = f"{result_path}/results{dhash}.json"

    model = LitMNIST.load_from_checkpoint(ckpt_path)
    model.setup()
    model.eval()

    fgsm_results = fgsm(model, epsilon)
    add_fgsm_to_results(fgsm_results, json_path)

    return fgsm_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--result_path", type=str, default=RESULT_PATH)
    parser.add_argument("--fgsm_epsilon", type=float, default=FGSM_EPSILON)
    parser.add_argument("--dhash", type=str, default="")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    fgsm_results = fgsm_from_path(args.result_path, args.fgsm_epsilon, args.dhash)

    fig_path = f"{args.result_path}/fgsm{args.dhash}.png"
    first_fgsm = fgsm_results[1][0]
    plot_fgsm(*first_fgsm, args.fgsm_epsilon, fig_path)
