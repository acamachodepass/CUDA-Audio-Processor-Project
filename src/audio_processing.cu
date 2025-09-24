#include "audio_processing.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "cufft.h"
#include "math_constants.h"
#include "npp.h"
#include "wav_processing.h"

namespace audio_processing {

// File-scope static variables for CUDA device memory and cuFFT plans.
// These pointers hold the state of the audio processing pipeline on the GPU.
// The 's_' prefix indicates static storage duration (file scope).
// The 'd_' prefix is a common convention for device pointers.
static float* s_d_time_data_interleaved = nullptr;
static float* s_d_time_data_planar = nullptr;
static cufftComplex* s_d_freq_data = nullptr;
static float* s_d_window = nullptr;
static float* s_d_input_samples = nullptr;
static float* s_d_output_samples = nullptr;

static cufftHandle s_fft_plan_forward = 0;
static cufftHandle s_fft_plan_inverse = 0;
static uint16_t s_num_channels = 1;

// Helper macro to check for CUDA errors in functions that return void.
// This is a common pattern in CUDA programming for concise error handling.
#define CUDA_CHECK_VOID(call)                                       \
  do {                                                              \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__,       \
              __LINE__, cudaGetErrorString(err));                   \
      return;                                                       \
    }                                                               \
  } while (0)

// --- CUDA Kernels ---

/**
 * @brief De-interleaves a frame of audio data from LRLR... to LLL...RRR...
 *
 * This kernel rearranges the memory layout of a single audio frame to prepare
 * it for batched processing, where each channel is a contiguous block of memory.
 *
 * @param input The input buffer with interleaved samples.
 * @param output The output buffer for planar (de-interleaved) samples.
 * @param frame_size The number of samples per channel in the frame (e.g., FFT_SIZE).
 * @param num_channels The number of audio channels.
 */
__global__ void DeinterleaveKernel(const float* input, float* output,
                                   int frame_size, int num_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= frame_size * num_channels) return;

  int channel = idx % num_channels;
  int sample_in_frame = idx / num_channels;

  output[channel * frame_size + sample_in_frame] = input[idx];
}

/**
 * @brief Applies a synthesis window and performs overlap-add.
 *
 * This kernel takes the result of the inverse FFT, applies a window function
 * to it, and atomically adds the result to the final output buffer. This
 * reconstructs the final audio signal from overlapping frames.
 *
 * @param frame_data_planar The input windowed time-domain data for one frame.
 * @param output The final output buffer for the entire signal.
 * @param frame_size The number of samples per channel in the frame.
 * @param num_channels The number of audio channels.
 * @param window The synthesis window function to apply.
 */
__global__ void OverlapAddKernel(const float* frame_data_planar, float* output,
                                 int frame_size, int num_channels,
                                 const float* window) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= frame_size * num_channels) return;

  int channel = idx % num_channels;
  int sample_in_frame = idx / num_channels;

  // Apply window and perform atomic add for safe concurrent writes.
  float value =
      frame_data_planar[channel * frame_size + sample_in_frame] *
      window[sample_in_frame];
  atomicAdd(&output[idx], value);
}

/**
 * @brief Modifies frequency bin magnitudes.
 *
 * This kernel operates on the frequency-domain data after the FFT. It serves
 * as a placeholder for various timbre modification effects.
 *
 * @param freq_data A pointer to the frequency-domain data for all channels.
 * @param length The number of complex frequency bins per channel.
 * @param num_channels The number of audio channels.
 */
__global__ void ModifyFrequencyBinsKernel(cufftComplex* freq_data, int length,
                                          int num_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= length * num_channels) return;

  int channel = idx / length;
  int bin = idx % length;

  cufftComplex* channel_data = freq_data + channel * length;

  // Example modification: Boost frequencies in a specific range.
  if (bin > 100 && bin < 200) {
    channel_data[bin].x *= 1.5f;
    channel_data[bin].y *= 1.5f;
  }
}

// --- Host-Side Implementation ---

bool InitializeAudioProcessing(uint16_t num_channels,
                             const wav_processing::WavInfo& info) {
  s_num_channels = num_channels;

  // Allocate GPU memory for a single, reusable frame buffer.
  if (cudaMalloc(&s_d_time_data_planar,
                 kFftSize * s_num_channels * sizeof(float)) != cudaSuccess) {
    std::cerr << "cudaMalloc s_d_time_data_planar failed\n";
    return false;
  }
  if (cudaMalloc(&s_d_freq_data, (kFftSize / 2 + 1) * s_num_channels *
                                     sizeof(cufftComplex)) != cudaSuccess) {
    std::cerr << "cudaMalloc s_d_freq_data failed\n";
    return false;
  }

  // Create a Hann window on the host.
  std::vector<float> host_window(kFftSize);
  for (int i = 0; i < kFftSize; ++i) {
    host_window[i] =
        0.5f * (1.0f - cosf(2.0f * CUDART_PI_F * i / (kFftSize - 1)));
  }

  // Allocate and transfer the window function to the GPU.
  if (cudaMalloc(&s_d_window, kFftSize * sizeof(float)) != cudaSuccess) {
    std::cerr << "cudaMalloc s_d_window failed\n";
    return false;
  }
  if (cudaMemcpy(s_d_window, host_window.data(), kFftSize * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "cudaMemcpy s_d_window failed\n";
    return false;
  }

  // Allocate memory for the entire input and output audio signals on the GPU.
  size_t total_samples = info.num_samples_ * s_num_channels;
  if (cudaMalloc(&s_d_input_samples, total_samples * sizeof(float)) !=
      cudaSuccess) {
    std::cerr << "cudaMalloc s_d_input_samples failed\n";
    return false;
  }
  if (cudaMalloc(&s_d_output_samples, total_samples * sizeof(float)) !=
      cudaSuccess) {
    std::cerr << "cudaMalloc s_d_output_samples failed\n";
    return false;
  }

  // Create batched cuFFT plans for forward (R2C) and inverse (C2R) transforms.
  int fft_length = kFftSize;
  if (cufftPlanMany(&s_fft_plan_forward, 1, &fft_length, nullptr, 1, kFftSize,
                    nullptr, 1, kFftSize / 2 + 1, CUFFT_R2C,
                    s_num_channels) != CUFFT_SUCCESS) {
    std::cerr << "CUFFT Forward PlanMany creation failed\n";
    return false;
  }
  if (cufftPlanMany(&s_fft_plan_inverse, 1, &fft_length, nullptr, 1,
                    kFftSize / 2 + 1, nullptr, 1, kFftSize, CUFFT_C2R,
                    s_num_channels) != CUFFT_SUCCESS) {
    std::cerr << "CUFFT Inverse PlanMany creation failed\n";
    return false;
  }

  return true;
}

void CleanupAudioProcessing() {
  // Free all allocated device memory and destroy cuFFT plans.
  cudaFree(s_d_time_data_interleaved);
  cudaFree(s_d_time_data_planar);
  cudaFree(s_d_freq_data);
  cudaFree(s_d_window);
  cudaFree(s_d_input_samples);
  cudaFree(s_d_output_samples);
  cufftDestroy(s_fft_plan_forward);
  cufftDestroy(s_fft_plan_inverse);
}

// A helper function to apply the analysis window using NPP.
void ApplyWindowFunction(float* d_planar_data, int length) {
  for (int ch = 0; ch < s_num_channels; ++ch) {
    // nppsMul_32f_I performs an in-place multiplication.
    NppStatus status =
        nppsMul_32f_I(s_d_window, d_planar_data + ch * length, length);
    if (status != NPP_SUCCESS) {
      std::cerr << "NPP window apply error on channel " << ch << std::endl;
    }
  }
}

// A helper function to launch the frequency modification kernel.
void ModifyFrequencyBins(cufftComplex* d_freq, int length, int num_channels) {
  const int total_bins = length * num_channels;
  const int threads_per_block = 256;
  const int blocks = (total_bins + threads_per_block - 1) / threads_per_block;
  ModifyFrequencyBinsKernel<<<blocks, threads_per_block>>>(d_freq, length,
                                                           num_channels);
  CUDA_CHECK_VOID(cudaGetLastError());
}

void ProcessAudio(const std::vector<float>& input_samples,
                  std::vector<float>* output_samples,
                  const wav_processing::WavInfo& info) {
  size_t total_samples_interleaved = input_samples.size();
  output_samples->resize(total_samples_interleaved);

  // Perform a single, large memory transfer from host to device for efficiency.
  CUDA_CHECK_VOID(cudaMemcpy(s_d_input_samples, input_samples.data(),
                             total_samples_interleaved * sizeof(float),
                             cudaMemcpyHostToDevice));
  // Ensure the output buffer is zeroed out for the overlap-add process.
  CUDA_CHECK_VOID(cudaMemset(s_d_output_samples, 0,
                             total_samples_interleaved * sizeof(float)));

  const int hop_size = kFftSize - kOverlap;
  const size_t frame_size_interleaved = kFftSize * s_num_channels;
  const size_t hop_size_interleaved = hop_size * s_num_channels;

  // Main processing loop: iterate over the audio signal frame by frame.
  for (size_t pos = 0; pos + frame_size_interleaved <= total_samples_interleaved;
       pos += hop_size_interleaved) {
    const int threads_per_block = 256;
    const int blocks =
        (frame_size_interleaved + threads_per_block - 1) / threads_per_block;

    // Step 1: De-interleave the current frame from LRLR... to LLL...RRR...
    DeinterleaveKernel<<<blocks, threads_per_block>>>(
        s_d_input_samples + pos, s_d_time_data_planar, kFftSize,
        s_num_channels);
    CUDA_CHECK_VOID(cudaGetLastError());

    // Step 2: Apply the analysis window function to the time-domain data.
    ApplyWindowFunction(s_d_time_data_planar, kFftSize);

    // Step 3: Perform the Forward FFT (Real-to-Complex).
    if (cufftExecR2C(s_fft_plan_forward, s_d_time_data_planar, s_d_freq_data) !=
        CUFFT_SUCCESS) {
      std::cerr << "CUFFT ExecR2C failed\n";
      return;
    }

    // Step 4: Modify the signal in the frequency domain.
    ModifyFrequencyBins(s_d_freq_data, kFftSize / 2 + 1, s_num_channels);

    // Step 5: Perform the Inverse FFT (Complex-to-Real).
    if (cufftExecC2R(s_fft_plan_inverse, s_d_freq_data,
                     s_d_time_data_planar) != CUFFT_SUCCESS) {
      std::cerr << "CUFFT ExecC2R failed\n";
      return;
    }

    // Step 6: Normalize the output of the IFFT.
    for (int ch = 0; ch < s_num_channels; ++ch) {
      nppsDivC_32f_I(static_cast<float>(kFftSize),
                     s_d_time_data_planar + ch * kFftSize, kFftSize);
    }

    // Step 7: Apply synthesis window and add the result to the output buffer.
    OverlapAddKernel<<<blocks, threads_per_block>>>(
        s_d_time_data_planar, s_d_output_samples + pos, kFftSize,
        s_num_channels, s_d_window);
    CUDA_CHECK_VOID(cudaGetLastError());
  }

  // Perform a single, large memory transfer from device back to host.
  CUDA_CHECK_VOID(cudaMemcpy(output_samples->data(), s_d_output_samples,
                             total_samples_interleaved * sizeof(float),
                             cudaMemcpyDeviceToHost));
}

}  // namespace audio_processing