import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
from nnpbr.model import NeuralMaterialCompressionModel, make_partition_bank
from nnpbr.bc_math import decode_blocks
from nnpbr.codec import _pack_bc6h_mode10_block, _decode_bc6h_mode10_block, read_dds
from nnpbr.artifacts import export_trained_artifacts, ExportedMaterial
from nnpbr.sampling import sample_mips_trilinear, build_mip_chain
from nnpbr.training import train, TrainConfig

torch.set_num_threads(2)


def model():
    torch.manual_seed(7)
    return NeuralMaterialCompressionModel(
        [8, 4, 4, 4],
        [2, 1, 1, 1],
        8,
        16,
        make_partition_bank(torch.device("cpu")),
        6,
        3,
        16,
    )


class Contract(unittest.TestCase):
    def test_all_partitions_and_extremes(self):
        rng = np.random.default_rng(7)
        bank = make_partition_bank(torch.device("cpu"))
        for pid in range(32):
            for k in range(8):
                ep = rng.integers(0, 64, (4, 3))
                ep[0, 0] = 0
                ep[3, 2] = 63
                idx = rng.integers(0, 8, 16)
                blob = _pack_bc6h_mode10_block(ep, idx, pid)
                exact = _decode_bc6h_mode10_block(blob, None, False)
                tensor = decode_blocks(
                    torch.tensor(ep)[None], torch.tensor(idx)[None], bank[pid][None]
                )[0].reshape(4, 4, 3)
                np.testing.assert_array_equal(tensor.numpy(), exact)

    def test_sampling_edges_and_last_mip(self):
        uv = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.5, 0.5]])
        mip = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        one = sample_mips_trilinear(
            [mip], uv, torch.tensor([-9.0, 0.0, 100.0]), "bilinear"
        )
        torch.testing.assert_close(one[:, 0], torch.tensor([1.0, 4.0, 2.5]))
        last = sample_mips_trilinear(
            [mip, torch.ones(1, 1, 1) * 9], uv, torch.ones(3) * 99, "bilinear"
        )
        self.assertTrue(torch.equal(last, torch.ones(3, 1) * 9))

    def test_frozen_export_and_rejection(self):
        m = model()
        m.quantize_and_freeze_bc_features()
        m.decoder.quantized_forward = True
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            export_trained_artifacts(m, p)
            # Debug checkpoint must never be needed for inference.
            (p / "metadata/decoder_state.pt").unlink()
            runtime = ExportedMaterial(p)
            uv = torch.rand(100, 2) * 1.2 - 0.1
            lod = torch.rand(100) * 8
            torch.testing.assert_close(
                m.forward_bc(uv, lod), runtime(uv, lod), rtol=0, atol=0
            )
            original = (p / "latent_00.bc6.dds").read_bytes()
            for corrupt in (original[:-1], original + b"x", b"bad"):
                (p / "latent_00.bc6.dds").write_bytes(corrupt)
                with self.assertRaises(ValueError):
                    ExportedMaterial(p)
                with self.assertRaises(ValueError):
                    read_dds(p / "latent_00.bc6.dds")
            (p / "latent_00.bc6.dds").write_bytes(original)
            (p / "decoder_fp16.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                ExportedMaterial(p)

    def test_training_and_microbatch(self):
        reference = build_mip_chain(torch.rand(8, 16, 16), 5)
        a = model()
        b = model()
        cfg = TrainConfig(
            batch_size=16,
            microbatch_size=16,
            phase1_iters=2,
            phase2_iters=2,
            phase3_iters=2,
            log_every=2,
        )
        train(a, reference, cfg)
        cfg.microbatch_size = 8
        train(b, reference, cfg)
        uv = torch.rand(30, 2)
        lod = torch.rand(30) * 4
        torch.testing.assert_close(
            a.forward_bc(uv, lod), b.forward_bc(uv, lod), atol=2e-4, rtol=2e-4
        )
        self.assertTrue(all(not p.requires_grad for p in a.bc_feature_parameters()))

    def test_resume_mid_constrained_stage(self):
        from unittest.mock import patch

        reference = build_mip_chain(torch.rand(8, 16, 16), 5)
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "checkpoint.pt"
            middle = Path(td) / "middle.pt"
            cfg = TrainConfig(
                batch_size=16,
                microbatch_size=8,
                phase1_iters=2,
                phase2_iters=3,
                phase3_iters=2,
                log_every=3,
                checkpoint_every=1,
                checkpoint_path=str(checkpoint),
            )
            save = torch.save

            def capture(obj, path):
                save(obj, path)
                if obj["phase"] == 2 and obj["completed"] == 1:
                    save(obj, middle)

            a = model()
            with patch("nnpbr.training.torch.save", side_effect=capture):
                train(a, reference, cfg)
            b = model()
            cfg.resume = str(middle)
            train(b, reference, cfg)
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                train(model(), [r + 0.01 for r in reference], cfg)

    def test_independent_oracle(self):
        import ctypes, os

        library = os.environ.get("BCDEC_ORACLE")
        if not library:
            self.skipTest("Set BCDEC_ORACLE to the built independent decoder library")
        oracle = ctypes.CDLL(library)
        oracle.decode_block.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        rng = np.random.default_rng(20261106)
        for signed in (False, True):
            for pid in range(32):
                for trial in range(32):
                    ep = rng.integers(
                        -31 if signed else 0, 32 if signed else 64, (4, 3)
                    )
                    indices = rng.integers(0, 8, 16)
                    packed = _pack_bc6h_mode10_block(
                        ep, indices, pid, signed_mode=signed
                    )
                    reference = np.empty((4, 4, 3), np.float32)
                    oracle.decode_block(
                        ctypes.create_string_buffer(packed),
                        reference.ctypes.data,
                        int(signed),
                    )
                    np.testing.assert_array_equal(
                        _decode_bc6h_mode10_block(packed, None, signed), reference
                    )


if __name__ == "__main__":
    unittest.main()
