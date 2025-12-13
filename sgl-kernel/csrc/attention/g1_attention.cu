#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__constant__ float fp4_e2m1_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void unpack_fp4_to_float(uint8_t packed, float scale, float* val1, float* val2) {
    uint8_t low = packed & 0x0F;
    uint8_t high = packed >> 4;

    *val1 = fp4_e2m1_lut[low] * scale;
    *val2 = fp4_e2m1_lut[high] * scale;
}

__device__ __forceinline__ void load_128bit(const void* src, int4* dst) {
    *dst = *reinterpret_cast<const int4*>(src);
}

__global__
void forward_kernel_fp4(
    const uint8_t* __restrict__ Q_packed,
    const uint8_t* __restrict__ K_packed,
    const uint8_t* __restrict__ V_packed,
    const float* __restrict__ Gate_Logits,
    const float scale_Q,
    const float scale_K,
    const float scale_V,
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    const float softmax_scale,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row_stride_bytes = d / 2;

    size_t batch_head_offset_bytes = (bx * gridDim.y * N * row_stride_bytes) + (by * N * row_stride_bytes);
    size_t batch_head_offset_elems = (bx * gridDim.y * N * d) + (by * N * d);

    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size_floats = Bc * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size_floats];
    float* Vj = &sram[tile_size_floats * 2];
    float* S  = &sram[tile_size_floats * 3];

    for (int j = 0; j < Tc; j++) {

        for (int x_bytes = tx; x_bytes < row_stride_bytes; x_bytes += Bc) {
            int k_idx = batch_head_offset_bytes + (j * Bc * row_stride_bytes) + x_bytes;

            uint8_t k_val_packed = K_packed[k_idx];
            uint8_t v_val_packed = V_packed[k_idx];

            float k1, k2, v1, v2;
            unpack_fp4_to_float(k_val_packed, scale_K, &k1, &k2);
            unpack_fp4_to_float(v_val_packed, scale_V, &v1, &v2);

            Kj[(x_bytes * 2)]     = k1;
            Kj[(x_bytes * 2) + 1] = k2;
            Vj[(x_bytes * 2)]     = v1;
            Vj[(x_bytes * 2) + 1] = v2;
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {

            for (int x_bytes = tx; x_bytes < row_stride_bytes; x_bytes += Bc) {
                int q_idx = batch_head_offset_bytes + (i * Br * row_stride_bytes) + x_bytes;
                uint8_t q_val_packed = Q_packed[q_idx];

                float q1, q2;
                unpack_fp4_to_float(q_val_packed, scale_Q, &q1, &q2);

                Qi[(x_bytes * 2)]     = q1;
                Qi[(x_bytes * 2) + 1] = q2;
            }
            __syncthreads();

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];
            float row_m = -INFINITY;

            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m) row_m = sum;
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                int global_out_idx = batch_head_offset_elems + (i * Br * d) + (tx * d) + x;
                float old_O = O[global_out_idx];

                float current_O = (1 / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) * old_O)
                    + (__expf(row_m - row_m_new) * pv));

                if (j == Tc - 1) {
                    float gate_logit = Gate_Logits[global_out_idx];
                    float gate_val = sigmoid(gate_logit);
                    current_O = current_O * gate_val;
                }
                O[global_out_idx] = current_O;
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}
