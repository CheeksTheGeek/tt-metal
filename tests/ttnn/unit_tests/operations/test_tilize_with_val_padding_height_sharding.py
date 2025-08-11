# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_multicore", [True, False])
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        # HEIGHT_SHARDED: only pad height, keep width constant (width must match physical device width)
        ([1, 1, 64, 128], [1, 1, 128, 128]),
        ([2, 1, 64, 128], [2, 1, 128, 128]),
        ([1, 2, 128, 256], [1, 2, 256, 256]),
    ],
)
@pytest.mark.parametrize("pad_value", [0.0, 1.5, -3.2])
@pytest.mark.parametrize(
    "num_cores, core_grid",
    [
        (2, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})),
        (4, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})),
    ],
)
def test_tilize_with_val_padding_height_sharded(
    device, dtype, use_multicore, input_shape, output_shape, pad_value, num_cores, core_grid
):
    """Test TilizeWithValPadding with HEIGHT_SHARDED memory layout."""
    
    # Calculate height shard dimensions
    total_height = 1
    for i in range(len(input_shape) - 1):
        total_height *= input_shape[i]
    width = input_shape[-1]
    
    shard_height = total_height // num_cores
    shard_shape = (shard_height, width)
    
    TILE = 32
    # Skip if per-core shard height is not a multiple of tile height
    if total_height % (num_cores * TILE) != 0:
        pytest.skip(
            f"Per-core shard height must be a multiple of {TILE}: "
            f"N*C*H={total_height}, cores={num_cores}"
        )
    
    # Create input tensor
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Setup height sharded memory config for input
    input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    
    # Setup height sharded memory config for output  
    output_total_height = 1
    for i in range(len(output_shape) - 1):
        output_total_height *= output_shape[i]
    output_width = output_shape[-1]
    output_shard_height = output_total_height // num_cores
    output_shard_shape = (output_shard_height, output_width)
    
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    
    # Move input to device with height sharding
    input_ttnn = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config)
    
    # Execute tilize with val padding
    output_ttnn = ttnn.tilize_with_val_padding(
        input_ttnn,
        output_tensor_shape=output_shape,
        pad_value=pad_value,
        memory_config=output_memory_config,
        use_multicore=use_multicore,
    )
    
    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(output_ttnn)
    
    # Create expected output using torch operations
    expected_output = torch.full(output_shape, pad_value, dtype=torch.bfloat16)
    
    # Copy input data to the appropriate location in expected output
    slices = []
    for i in range(len(input_shape)):
        slices.append(slice(0, input_shape[i]))
    expected_output[tuple(slices)] = input_torch
    
    # Assert outputs match within tolerance
    assert_with_pcc(expected_output, output_torch, 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_multicore", [True, False])
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        # Make both H and W tile-aligned per-core for the tested core grids
        ([1, 1, 64, 64], [1, 1, 128, 128]),
        ([2, 2, 128, 128], [2, 2, 256, 256]),
    ],
)
@pytest.mark.parametrize("pad_value", [0.0, 2.5])
@pytest.mark.parametrize(
    "num_cores, core_grid",
    [
        (4, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})),
        (16, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})),
    ],
)
def test_tilize_with_val_padding_block_sharded(
    device, dtype, use_multicore, input_shape, output_shape, pad_value, num_cores, core_grid
):
    """Test TilizeWithValPadding with BLOCK_SHARDED memory layout."""
    
    # Calculate block shard dimensions
    total_height = 1
    for i in range(len(input_shape) - 1):
        total_height *= input_shape[i]
    width = input_shape[-1]
    
    cores_per_dim = int(math.sqrt(num_cores))
    shard_height = total_height // cores_per_dim
    shard_width = width // cores_per_dim
    shard_shape = (shard_height, shard_width)
    
    TILE = 32
    # Skip if per-core shard dims are not multiples of tile dims
    if (total_height % (cores_per_dim * TILE) != 0) or (width % (cores_per_dim * TILE) != 0):
        pytest.skip(
            f"Per-core shard must be tile {TILE}x{TILE}-sized: "
            f"N*C*H={total_height}, W={width}, cores_per_dim={cores_per_dim}"
        )
    
    # Create input tensor
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Setup block sharded memory config for input
    input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    
    # Setup block sharded memory config for output
    output_total_height = 1
    for i in range(len(output_shape) - 1):
        output_total_height *= output_shape[i]
    output_width = output_shape[-1]
    output_shard_height = output_total_height // cores_per_dim
    output_shard_width = output_width // cores_per_dim
    output_shard_shape = (output_shard_height, output_shard_width)
    
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    
    # Move input to device with block sharding
    input_ttnn = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config)
    
    # Execute tilize with val padding
    output_ttnn = ttnn.tilize_with_val_padding(
        input_ttnn,
        output_tensor_shape=output_shape,
        pad_value=pad_value,
        memory_config=output_memory_config,
        use_multicore=use_multicore,
    )
    
    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(output_ttnn)
    
    # Create expected output using torch operations
    expected_output = torch.full(output_shape, pad_value, dtype=torch.bfloat16)
    
    # Copy input data to the appropriate location in expected output
    slices = []
    for i in range(len(input_shape)):
        slices.append(slice(0, input_shape[i]))
    expected_output[tuple(slices)] = input_torch
    
    # Assert outputs match within tolerance
    assert_with_pcc(expected_output, output_torch, 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "input_layout, output_layout",
    [
        # Only test supported combinations - HEIGHT_SHARDED with constant width
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
def test_tilize_with_val_padding_memory_layout_combinations(
    device, dtype, input_layout, output_layout
):
    """Test TilizeWithValPadding with different input/output memory layout combinations."""
    
    # Use constant width for HEIGHT_SHARDED to avoid width constraint violation
    input_shape = [1, 1, 64, 128]
    output_shape = [1, 1, 128, 128]  # Only pad height, keep width constant
    pad_value = 1.0
    
    num_cores = 2
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    
    # Calculate shard dimensions for height sharding
    total_height = 64
    width = 128
    shard_height = total_height // num_cores
    shard_shape = (shard_height, width)
    
    output_total_height = 128
    output_width = 128  # Keep width constant
    output_shard_height = output_total_height // num_cores
    output_shard_shape = (output_shard_height, output_width)
    
    # Create input tensor
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    # Setup input memory config
    if input_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        input_memory_config = ttnn.MemoryConfig(input_layout, ttnn.BufferType.L1, input_shard_spec)
    else:
        input_memory_config = ttnn.MemoryConfig(input_layout, ttnn.BufferType.L1)
    
    # Setup output memory config
    if output_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_memory_config = ttnn.MemoryConfig(output_layout, ttnn.BufferType.L1, output_shard_spec)
    else:
        output_memory_config = ttnn.MemoryConfig(output_layout, ttnn.BufferType.L1)
    
    # Move input to device
    input_ttnn = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config)
    
    # Execute tilize with val padding
    output_ttnn = ttnn.tilize_with_val_padding(
        input_ttnn,
        output_tensor_shape=output_shape,
        pad_value=pad_value,
        memory_config=output_memory_config,
        use_multicore=True,
    )
    
    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(output_ttnn)
    
    # Create expected output
    expected_output = torch.full(output_shape, pad_value, dtype=torch.bfloat16)
    expected_output[:, :, :64, :] = input_torch  # Width is constant, only height is padded
    
    # Assert outputs match within tolerance
    assert_with_pcc(expected_output, output_torch, 0.9999)


def test_tilize_with_val_padding_mixed_layout_failures(device):
    """Test that mixed layout combinations fail with appropriate error messages."""
    
    input_shape = [1, 1, 32, 32]
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    shard_shape = (16, 32)
    
    # Test 1: HEIGHT_SHARDED input -> INTERLEAVED output (mixed layouts not supported)
    input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    
    input_ttnn_sharded = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config)
    
    with pytest.raises(RuntimeError, match="Mixed interleaved/sharded memory layout combinations are not yet supported"):
        ttnn.tilize_with_val_padding(
            input_ttnn_sharded,
            output_tensor_shape=[1, 1, 64, 64],
            pad_value=0.0,
            memory_config=output_memory_config,
        )

    # Test 2: INTERLEAVED input -> HEIGHT_SHARDED output (mixed layouts not supported)
    input_memory_config_interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_shard_spec = ttnn.ShardSpec(core_grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    
    input_ttnn_interleaved = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config_interleaved)
    
    with pytest.raises(RuntimeError, match="Mixed interleaved/sharded memory layout combinations are not yet supported"):
        ttnn.tilize_with_val_padding(
            input_ttnn_interleaved,
            output_tensor_shape=[1, 1, 64, 128],  # Increasing width would also fail HEIGHT_SHARDED constraint
            pad_value=0.0,
            memory_config=output_memory_config_sharded,
        )


def test_tilize_with_val_padding_height_sharding_validation_errors(device):
    """Test that proper validation errors are raised for invalid height sharding configurations."""
    
    input_shape = [1, 1, 32, 32]
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    shard_shape = (16, 32)
    
    # Test 1: HEIGHT_SHARDED width change should fail with specific error message
    input_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    output_shard_spec = ttnn.ShardSpec(core_grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    
    input_ttnn = ttnn.to_device(input_ttnn, device, memory_config=input_memory_config)
    
    with pytest.raises(RuntimeError, match="HEIGHT_SHARDED tensors cannot change width"):
        ttnn.tilize_with_val_padding(
            input_ttnn,
            output_tensor_shape=[1, 1, 64, 64],  # Width changes from 32 to 64
            pad_value=0.0,
            memory_config=output_memory_config,
        )
    
    # Test 2: Different sharding types should fail
    different_shard_output_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    
    with pytest.raises(RuntimeError, match="input and output must have the same memory layout"):
        ttnn.tilize_with_val_padding(
            input_ttnn,
            output_tensor_shape=[1, 1, 64, 32],  # Keep width same but different layout
            pad_value=0.0,
            memory_config=different_shard_output_config,
        )
