#ifndef WAV_PROCESSING_H_
#define WAV_PROCESSING_H_

#include <cstdint>
#include <string>
#include <vector>

namespace wav_processing {

/**
 * @brief Holds the essential metadata for a WAV audio file.
 */
struct WavInfo {
  uint16_t num_channels_;
  uint32_t sample_rate_;
  uint16_t bits_per_sample_;
  // Total number of audio samples per channel.
  uint32_t num_samples_;
};

/**
 * @brief Loads a 16-bit PCM WAV audio file into a normalized float vector.
 *
 * This function reads a WAV file and converts its audio data into a vector
 * of floats, with each sample normalized to the range [-1.0, 1.0]. It
 * supports mono and stereo files. For stereo audio, the samples are
 * interleaved in the format [L, R, L, R, ...].
 *
 * @param filename The path to the input WAV file.
 * @param info A pointer to a WavInfo struct that will be populated with the
 * audio file's metadata.
 * @return A std::vector<float> containing the interleaved, normalized audio
 * samples. Returns an empty vector if the file cannot be opened or
 * is not a supported format.
 */
std::vector<float> LoadWavFile(const std::string& filename, WavInfo* info);

/**
 * @brief Saves a vector of normalized float audio samples to a 16-bit PCM WAV
 * file.
 *
 * This function takes a vector of interleaved audio samples (normalized between
 * -1.0 and 1.0) and writes them to a new WAV file. The audio metadata, such as
 * the sample rate and number of channels, is specified by the WavInfo struct.
 *
 * @param filename The path where the output WAV file will be created.
 * @param samples A constant reference to a std::vector<float> containing the
 * interleaved, normalized audio samples to save.
 * @param info A constant reference to a WavInfo struct containing the metadata
 * for the output file.
 */
void SaveWavFile(const std::string& filename,
                 const std::vector<float>& samples, const WavInfo& info);

}  // namespace wav_processing

#endif  // WAV_PROCESSING_H_