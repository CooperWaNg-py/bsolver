/* Metal entry points; compiled after the Metal prelude (metal_backend.mm) and kernel_core.h. */

kernel void scan_kernel(device const ulong* params [[buffer(0)]], device const ulong* offsets [[buffer(1)]],
                        device const ulong* gtable [[buffer(2)]], device const uint* targets [[buffer(3)]],
                        device uint* hits [[buffer(4)]], device uint* counters [[buffer(5)]],
                        uint t [[thread_position_in_grid]]) {
  scan_thread(t, params, offsets, gtable, targets, hits, counters);
}

kernel void kangaroo_kernel(device const ulong* params [[buffer(0)]], device const ulong* in [[buffer(1)]],
                            device ulong* out [[buffer(2)]], device const ulong* jumps [[buffer(3)]],
                            device ulong* dps [[buffer(4)]], device uint* counters [[buffer(5)]],
                            uint t [[thread_position_in_grid]]) {
  kangaroo_thread(t, params, in, out, jumps, dps, counters);
}
