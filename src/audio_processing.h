#ifndef AUDIO_PROCESSING_H_
#define AUDIO_PROCESSING_H_

#include <cstdint>
#include <vector>

#include "wav_processing.h"

namespace audio_processing {

// Defines core parameters for the Short-Time Fourier Transform (STFT) process
// used in timbre modification.
constexpr int kFftSize = 1024;
constexpr int kOverlap = 256;

/**
 * @brief Initializes audio processing resources, including GPU memory and FFT plans.
 *
 * This function must be called successfully before any audio processing can be
 * performed. It sets up the necessary backend components based on the audio
 * properties.
 *
 * @param num_channels The number of channels (e.g., 1 for mono, 2 for stereo)
 * in the audio to be processed.
 * @param info A constant reference to the WavInfo struct containing metadata
 * about the audio file.
 * @return true if all resources were initialized successfully, false otherwise.
 */
bool InitializeAudioProcessing(uint16_t num_channels,
                             const wav_processing::WavInfo& info);

/**
 * @brief Releases all allocated audio processing resources.
 *
 * This function should be called when audio processing is complete to free up
 * GPU memory and other resources.
 */
void CleanupAudioProcessing();

/**
 * @brief Processes a vector of audio samples on the GPU to modify its timbre.
 *
 * This function takes interleaved input audio samples, performs STFT, modifies
 * the signal in the frequency domain, and then performs an inverse STFT to
 * reconstruct the time-domain signal.
 *
 * @param input_samples A constant reference to a vector of interleaved float
 * samples to be processed.
 * @param output_samples A pointer to a vector where the processed interleaved
 * float samples will be stored. The vector will be resized appropriately.
 * @param info A constant reference to the WavInfo struct containing metadata
 * about the audio.
 */
void ProcessAudio(const std::vector<float>& input_samples,
                  std::vector<float>* output_samples,
                  const wav_processing::WavInfo& info);

}  // namespace audio_processing

#endif  // AUDIO_PROCESSING_H_