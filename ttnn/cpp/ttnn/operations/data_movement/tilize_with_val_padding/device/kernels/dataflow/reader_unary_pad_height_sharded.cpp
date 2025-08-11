// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Per-batch, per-core sizes
    const uint32_t in_rows_per_batch   = get_arg_val<uint32_t>(0);
    const uint32_t in_row_bytes        = get_arg_val<uint32_t>(1);
    const uint32_t out_row_bytes       = get_arg_val<uint32_t>(2);
    const uint32_t tiles_per_batch     = get_arg_val<uint32_t>(3);
    const uint32_t pad_rows_per_batch  = get_arg_val<uint32_t>(4);
    const uint32_t num_batches         = get_arg_val<uint32_t>(5);
    const uint32_t packed_pad_value    = get_arg_val<uint32_t>(6);
    const uint32_t pad_cols_bytes      = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0); // bound to input buffer, page = in_row_bytes
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1); // tile-paged CB (page = 2048 for bf16)
    constexpr uint32_t cb_id_pad = get_compile_time_arg_val(2); // page = out_row_bytes

    // Reserve enough "pages" on the bound input CB to cover all rows across batches
    cb_reserve_back(cb_id_in0, in_rows_per_batch * num_batches);
    // Pad CB: just one page that we fill with pad value
    cb_reserve_back(cb_id_pad, 1);

    // Prepare the pad page (full out_row_bytes of pad value)
    uint32_t pad_addr = get_write_ptr(cb_id_pad);
    volatile tt_l1_ptr std::uint32_t* pad = (volatile tt_l1_ptr std::uint32_t*)(pad_addr);
    for (uint32_t i = 0; i < (out_row_bytes >> 2); ++i) {
        pad[i] = packed_pad_value;
    }
    const uint64_t pad_noc_addr = get_noc_addr(pad_addr);

    // Base source address (bound to sharded buffer)
    uint64_t read_noc_addr = get_noc_addr(get_read_ptr(cb_id_in0));

    // For each batch on this core
    for (uint32_t b = 0; b < num_batches; ++b) {
        // We will write exactly tiles_per_batch pages to cb_id_out
        cb_reserve_back(cb_id_out, tiles_per_batch);
        uint32_t write_addr = get_write_ptr(cb_id_out);

        // Copy all input rows; after each row, append pad columns if needed
        for (uint32_t r = 0; r < in_rows_per_batch; ++r) {
            // copy input row
            noc_async_read(read_noc_addr, write_addr, in_row_bytes);
            read_noc_addr += in_row_bytes;
            write_addr    += in_row_bytes;

            // pad columns to reach out_row_bytes for this row
            if (pad_cols_bytes) {
                noc_async_read(pad_noc_addr, write_addr, pad_cols_bytes);
                write_addr += pad_cols_bytes;
            }
        }

        // Add fully padded rows
        for (uint32_t r = 0; r < pad_rows_per_batch; ++r) {
            noc_async_read(pad_noc_addr, write_addr, out_row_bytes);
            write_addr += out_row_bytes;
        }

        // All DMA done; commit pages for the compute kernel (tilize.cpp)
        noc_async_read_barrier();
        cb_push_back(cb_id_out, tiles_per_batch);
    }
}
