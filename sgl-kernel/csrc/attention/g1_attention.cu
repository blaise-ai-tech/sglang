#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ================================================================
// SOTA HELPER FUNCTIONS
// ================================================================

// Sigmoid implementation
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// FP4 (E2M1) Dequantization Lookup Table
// This maps the 4-bit integer representation to a float value.
// E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.
// This is a simplified lookup for demonstration. In real SOTA, this is often
// handled by specific Tensor Core instructions or PTX.
__constant__ float fp4_e2m1_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f, // Positive 0-7
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // Negative 8-15
};

// Unpack and Dequantize: Converts 1 byte (2 packed FP4s) into 2 floats
// Input: packed_byte containing [High 4 bits | Low 4 bits]
// Output: val1, val2 (passed by reference)
__device__ __forceinline__ void unpack_fp4_to_float(uint8_t packed, float scale, float* val1, float* val2) {
    uint8_t low = packed & 0x0F;
    uint8_t high = packed >> 4;

    *val1 = fp4_e2m1_lut[low] * scale;
    *val2 = fp4_e2m1_lut[high] * scale;
}

// Vectorized Load helper: Loads 16 bytes (32 FP4 elements) at once
__device__ __forceinline__ void load_128bit(const void* src, int4* dst) {
    *dst = *reinterpret_cast<const int4*>(src);
}

// ================================================================
// KERNEL
// ================================================================

__global__
void forward_kernel_fp4(
    const uint8_t* __restrict__ Q_packed,    // FP4 packed (N * d / 2 bytes)
    const uint8_t* __restrict__ K_packed,    // FP4 packed
    const uint8_t* __restrict__ V_packed,    // FP4 packed
    const float* __restrict__ Gate_Logits,     // Precomputed X * W_theta (FP32)
    const float scale_Q,                       // Quantization scale factor for Q
    const float scale_K,                       // Quantization scale factor for K
    const float scale_V,                       // Quantization scale factor for V
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    const float softmax_scale,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ O                      // Output in FP32
) {
    // -----------------------------------------------------------
    // 1. Setup & Shared Memory Allocation
    // -----------------------------------------------------------
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head indices

    // Calculate strides (d is number of elements, but storage is d/2 bytes)
    int row_stride_bytes = d / 2;

    // Global Memory Offsets
    // GridY = num_heads.
    size_t batch_head_offset_bytes = (bx * gridDim.y * N * row_stride_bytes) + (by * N * row_stride_bytes);
    size_t batch_head_offset_elems = (bx * gridDim.y * N * d) + (by * N * d);

    int lm_offset = (bx * gridDim.y * N) + (by * N);

    // Shared Memory: We store Dequantized Floats in SRAM for fast compute
    // Layout: Qi [Br][d], Kj [Bc][d], Vj [Bc][d], S [Br][Bc]
    // d elements take d*4 bytes (float)
    extern __shared__ float sram[];
    int tile_size_floats = Bc * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size_floats];
    float* Vj = &sram[tile_size_floats * 2];
    float* S  = &sram[tile_size_floats * 3];

    // -----------------------------------------------------------
    // 2. Outer Loop: Iterate over Key/Value Blocks (Tile c)
    // -----------------------------------------------------------
    for (int j = 0; j < Tc; j++) {

        // --- LOAD K and V from Global (FP4) to SRAM (Float) ---
        // Each thread loads multiple elements to maximize bandwidth.
        // Assuming Bc=32, d=64. Each thread handles specific columns.
        // For simplicity in this snippet, we use a basic loop with unpacking.
        for (int x_bytes = tx; x_bytes < row_stride_bytes; x_bytes += Bc) {
            // Index in byte array
            int k_idx = batch_head_offset_bytes + (j * Bc * row_stride_bytes) + x_bytes;

            // Load 1 byte (2 elements)
            // Note: In real SOTA, use int4 loads here for chunks of 16 bytes.
            uint8_t k_val_packed = K_packed[k_idx];
            uint8_t v_val_packed = V_packed[k_idx];

            float k1, k2, v1, v2;
            unpack_fp4_to_float(k_val_packed, scale_K, &k1, &k2);
            unpack_fp4_to_float(v_val_packed, scale_V, &v1, &v2);

            // Write to Shared Memory (coalesced float write)
            // x_bytes * 2 is the element index
            Kj[(x_bytes * 2)]     = k1;
            Kj[(x_bytes * 2) + 1] = k2;
            Vj[(x_bytes * 2)]     = v1;
            Vj[(x_bytes * 2) + 1] = v2;
        }
        __syncthreads();

        // -----------------------------------------------------------
        // 3. Inner Loop: Iterate over Query Blocks (Tile r)
        // -----------------------------------------------------------
        for (int i = 0; i < Tr; i++)  {

            // --- LOAD Q from Global (FP4) to SRAM (Float) ---
            for (int x_bytes = tx; x_bytes < row_stride_bytes; x_bytes += Bc) {
                int q_idx = batch_head_offset_bytes + (i * Br * row_stride_bytes) + x_bytes;
                uint8_t q_val_packed = Q_packed[q_idx];

                float q1, q2;
                unpack_fp4_to_float(q_val_packed, scale_Q, &q1, &q2);

                Qi[(x_bytes * 2)]     = q1;
                Qi[(x_bytes * 2) + 1] = q2;
            }
            // No sync needed here if Bc == Br and mapping is 1-to-1,
            // but for safety with arbitrary tiles:
            __syncthreads();

            // Load running statistics
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // --- COMPUTE SCORES: S = Q * K^T ---
            float row_m = -INFINITY;

            // This loop calculates one row of S per thread (tx corresponds to row in Q tile)
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                // Vectorize this dot product in production
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m) row_m = sum;
            }

            // --- ONLINE SOFTMAX: P = exp(S - max) ---
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                // Exponentiate safely
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Update Rescaling Statistics
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // --- UPDATE OUTPUT: O ---
            // We write to global memory.
            // Since this thread handles row 'tx', we iterate over embedding dim 'd'.
            for (int x = 0; x < d; x++) {
                float pv = 0;
                // P * V
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                // Retrieve current O value from HBM (High Bandwidth Memory)
                int global_out_idx = batch_head_offset_elems + (i * Br * d) + (tx * d) + x;
                float old_O = O[global_out_idx];

                // FlashAttention Update Step
                float current_O = (1 / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) * old_O)
                    + (__expf(row_m - row_m_new) * pv));

                // -------------------------------------------------------------
                // GATED ATTENTION (G1) IMPLEMENTATION
                // Formula: Y' = Y * sigmoid(X * W_theta)
                // -------------------------------------------------------------
                // We apply this only when the attention score for this element is
                // FULLY finalized. This happens at the very last iteration of the
                // outer loop (j).

                if (j == Tc - 1) {
                    // 1. Get the Gate Logit (XW)
                    // Assuming Gate_Logits is pre-computed and stored in FP32 or FP16.
                    float gate_logit = Gate_Logits[global_out_idx];

                    // 2. Apply Sigmoid Activation
                    float gate_val = sigmoid(gate_logit);

                    // 3. Apply Gate
                    current_O = current_O * gate_val;
                }
                // -------------------------------------------------------------

                // Write back
                O[global_out_idx] = current_O;
            }

            // Write back statistics
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

// ================================================================
// HOST WRAPPER
// ================================================================

torch::Tensor forward(
    torch::Tensor Q_packed, // uint8 tensor
    torch::Tensor K_packed,
    torch::Tensor V_packed,
    torch::Tensor Gate_Logits, // Float32 tensor
    float scale_Q, float scale_K, float scale_V
) {
    const int Bc = 32; const int Br = 32;

    const int B = Q_packed.size(0);
    const int nh = Q_packed.size(1);
    const int N = Q_packed.size(2);
    // d is derived from packed size. 1 byte = 2 elements.
    const int d = Q_packed.size(3) * 2;

    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt((float)d);

    // Output is kept in high precision (FP32) for stability
    // before potentially re-quantizing in the next layer
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto O = torch::zeros({B, nh, N, d}, options);
    auto l = torch::zeros({B, nh, N}, options);
    auto m = torch::full({B, nh, N}, -INFINITY, options);

    // Calculate Shared Memory (SRAM) usage
    // Qi (float), Kj (float), Vj (float), S (float)
    // We dequantize into float inside SRAM, so size is based on float size, not FP4 size
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    // Check Max Shared Memory
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Error: SRAM requirement %d exceeds device limit %d\n", sram_size, max_sram_size);
        // Add dynamic tiling logic here in production
    }

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc); // Assuming Bc threads align with warp size (32)

    forward_kernel_fp4<<<grid_dim, block_dim, sram_size>>>(
        Q_packed.data_ptr<uint8_t>(),
        K_packed.data_ptr<uint8_t>(),
        V_packed.data_ptr<uint8_t>(),
        Gate_Logits.data_ptr<float>(),
        scale_Q, scale_K, scale_V,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}
