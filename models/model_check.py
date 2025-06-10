import argparse
import re
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM

def collect_layer_weights(model):
    """
    Collects all weights from each layer of the model into flat numpy arrays.
    Layers are detected by regex matching 'layer.<index>' in parameter names.
    """
    layer_weights = {}
    pattern = re.compile(r'layer\.(\d+)')
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy().flatten()
        m = pattern.search(name)
        if m:
            layer_idx = int(m.group(1))
            layer_weights.setdefault(layer_idx, []).append(arr)
        else:
            layer_weights.setdefault('others', []).append(arr)
    # concatenate lists of arrays into single arrays
    for key in list(layer_weights.keys()):
        layer_weights[key] = np.concatenate(layer_weights[key], axis=0)
    return layer_weights

def plot_distributions(layer_weights, model_name, save_path=None):
    """
    Plots weight distributions per layer using seaborn histograms with KDE overlay.
    Arranges subplots in a square grid.
    """
    keys = sorted(layer_weights.keys(), key=lambda x: (float('inf') if x == 'others' else x))
    num_layers = len(keys)
    cols = int(np.ceil(np.sqrt(num_layers)))
    rows = int(np.ceil(num_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        data = layer_weights[key]
        ax = axes[idx]
        sns.histplot(data, bins=100, kde=True, ax=ax)
        ax.set_title(f'Layer {key}')
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Frequency')

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Weight Distributions for {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Load a masked-language-model and inspect weight distributions per layer."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pretrained MLM model"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Optional path to save the resulting plot image"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="If set, disables CUDA even if available"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Loading model {args.model_name} on device {device}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print("Collecting weights per layer...")
    layer_weights = collect_layer_weights(model)

    print("Checking for NaNs in weights...")
    for layer, weights in layer_weights.items():
        if np.isnan(weights).any():
            print(f"Warning: NaNs detected in layer {layer}")

    print("Plotting distributions...")
    plot_distributions(layer_weights, args.model_name, args.save_plot)

if __name__ == "__main__":
    main()