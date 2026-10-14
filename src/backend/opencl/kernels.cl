/* OpenCL entry points; compiled after the OpenCL prelude (opencl_backend.cpp) and kernel_core.h. */

__kernel void scan_kernel(__global const ulong* params, __global const ulong* offsets, __global const ulong* gtable,
                          __global const uint* targets, __global uint* hits, __global uint* counters) {
  scan_thread((uint)get_global_id(0), params, offsets, gtable, targets, hits, counters);
}

__kernel void kangaroo_kernel(__global const ulong* params, __global const ulong* in, __global ulong* out,
                              __global const ulong* jumps, __global ulong* dps, __global uint* counters) {
  kangaroo_thread((uint)get_global_id(0), params, in, out, jumps, dps, counters);
}
