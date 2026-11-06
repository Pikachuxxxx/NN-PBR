"""Compatibility imports; implementation lives in nnpbr/."""
from nnpbr.api import *
from nnpbr.codec import decode_quantized_params_to_mip, _write_bc6h_dds, _pack_mip_blocks_to_bc6h_mode10_bytes, _decode_mode10_params_to_mip, _decode_mode10_bytes_to_mip, _bc6h_mode10_partition_bank_np, _bc6h_sign_extend, _bc6h_unquantize, _bc6h_finish_unquantize, _bc6h_int_to_half_float, _bc6h_mode10_prepare_block, _bc6h_mode10_decode_from_quantized, _decode_bc6h_mode10_block, decode_bc6h_dds_mip0, _pack_bc6h_mode10_block
from nnpbr.model import _decode_mode10_quantized_blocks_torch, _search_mode10_initial_block_params
NeuralMaterialModel = NeuralMaterialCompressionModel
DecoderMLP = MaterialDecoderMLP
BCBlockMip = BC6SurrogateBlockLevel
BCBlockPyramid = BC6SurrogatePyramid
UnconstrainedLatentPyramid = WarmupLatentPyramid
if __name__ == "__main__":
    from nnpbr.cli import main
    main()
