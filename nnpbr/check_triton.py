"""Verify NN-PBR forward and backward compilation on the selected GPU."""

import argparse

import torch

from .model import NeuralMaterialCompressionModel, make_partition_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.exit(1, "A PyTorch CUDA or ROCm GPU is required.\n")
    torch.cuda.set_device(device.index if device.index is not None else 0)
    from torch.utils._triton import has_triton

    if not has_triton():
        parser.exit(
            1,
            "Triton is unavailable. Install the Triton build paired with your GPU's PyTorch distribution.\n",
        )
    import triton

    torch.manual_seed(7)
    model = NeuralMaterialCompressionModel(
        [8, 4, 4, 4],
        [2, 1, 1, 1],
        8,
        16,
        make_partition_bank(torch.device("cpu")),
        6,
        3,
        16,
    ).to(device)
    uv = torch.rand(128, 2, device=device)
    lod = torch.full((128,), 1.25, device=device)
    for forward in (model.forward_warmup, model.forward_bc):
        model.zero_grad(set_to_none=True)
        expected = forward(uv, lod)
        expected.square().mean().backward()
        expected_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        actual = torch.compile(forward, backend="inductor")(uv, lod)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
        for name, param in model.named_parameters():
            if name in expected_grads:
                torch.testing.assert_close(
                    param.grad, expected_grads[name], rtol=1e-3, atol=1e-5
                )
        print(f"{forward.__name__}: output and gradients match eager execution")
    print(
        f"Triton {triton.__version__} / PyTorch {torch.__version__} / {torch.cuda.get_device_name(device)}"
    )


if __name__ == "__main__":
    main()
