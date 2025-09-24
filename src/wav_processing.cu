#include "wav_processing.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace wav_processing {

// A struct representing the WAV file header format. The pragma pack directive
// ensures that the struct members are aligned without padding, matching the
// byte layout of a standard WAV file header on disk.
#pragma pack(push, 1)
struct WavHeader {
  char riff_[4];
  uint32_t chunk_size_;
  char wave_[4];
  char fmt_[4];
  uint32_t subchunk1_size_;
  uint16_t audio_format_;
  uint16_t num_channels_;
  uint32_t sample_rate_;
  uint32_t byte_rate_;
  uint16_t block_align_;
  uint16_t bits_per_sample_;
  char data_[4];
  uint32_t data_size_;
};
#pragma pack(pop)

std::vector<float> LoadWavFile(const std::string& filename, WavInfo* info) {
  // Open the WAV file in binary read mode.
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open WAV file: " << filename << std::endl;
    return {};
  }

  // Read the header from the file.
  WavHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));

  // Validate that the file is a supported WAV format. This reader supports
  // only 16-bit PCM mono or stereo files.
  if (std::strncmp(header.riff_, "RIFF", 4) != 0 ||
      std::strncmp(header.wave_, "WAVE", 4) != 0 ||
      header.audio_format_ != 1 ||  // 1 indicates PCM format.
      header.bits_per_sample_ != 16 ||
      (header.num_channels_ != 1 && header.num_channels_ != 2)) {
    std::cerr << "Unsupported WAV file format for: " << filename << std::endl;
    return {};
  }

  // Populate the output WavInfo struct with metadata from the header.
  info->num_channels_ = header.num_channels_;
  info->sample_rate_ = header.sample_rate_;
  info->bits_per_sample_ = header.bits_per_sample_;
  info->num_samples_ = header.data_size_ /
                       (header.num_channels_ * (header.bits_per_sample_ / 8));

  // Read the entire audio data chunk into a 16-bit integer buffer.
  std::vector<int16_t> pcm_buffer(header.data_size_ / sizeof(int16_t));
  file.read(reinterpret_cast<char*>(pcm_buffer.data()), header.data_size_);

  if (!file) {
    std::cerr << "Error reading WAV data from " << filename << std::endl;
    return {};
  }

  // Convert the 16-bit PCM samples to normalized floats in the range [-1.0, 1.0].
  std::vector<float> samples(pcm_buffer.size());
  for (size_t i = 0; i < pcm_buffer.size(); ++i) {
    // Normalization is done by dividing by 32768.0, the number of negative
    // values in a 16-bit signed integer range.
    samples[i] = static_cast<float>(pcm_buffer[i]) / 32768.0f;
  }

  return samples;
}

void SaveWavFile(const std::string& filename,
                 const std::vector<float>& samples, const WavInfo& info) {
  // Open the output file in binary write mode.
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open WAV file for writing: " << filename
              << std::endl;
    return;
  }

  // Construct the WAV header based on the provided audio info.
  WavHeader header;
  std::memcpy(header.riff_, "RIFF", 4);
  std::memcpy(header.wave_, "WAVE", 4);
  std::memcpy(header.fmt_, "fmt ", 4);
  std::memcpy(header.data_, "data", 4);

  header.subchunk1_size_ = 16;   // Size of the format chunk.
  header.audio_format_ = 1;      // 1 indicates PCM format.
  header.num_channels_ = info.num_channels_;
  header.sample_rate_ = info.sample_rate_;
  header.bits_per_sample_ = 16;  // Hardcode to 16-bit PCM output.

  // Calculate derived header fields.
  header.byte_rate_ =
      header.sample_rate_ * header.num_channels_ * header.bits_per_sample_ / 8;
  header.block_align_ = header.num_channels_ * header.bits_per_sample_ / 8;
  header.data_size_ = samples.size() * sizeof(int16_t);
  header.chunk_size_ = 36 + header.data_size_;

  // Write the populated header to the file.
  file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Convert the normalized float samples back to 16-bit PCM.
  std::vector<int16_t> pcm_buffer(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    float float_val = samples[i];
    // Clamp the sample to the valid range [-1.0, 1.0] before conversion.
    float clamped_val = (float_val < -1.0f)
                           ? -1.0f
                           : (float_val > 1.0f) ? 1.0f : float_val;
    // Denormalize by multiplying by 32767.0, the max positive value for a
    // 16-bit signed integer.
    pcm_buffer[i] = static_cast<int16_t>(clamped_val * 32767.0f);
  }

  // Write the entire PCM data buffer to the file for efficiency.
  file.write(reinterpret_cast<const char*>(pcm_buffer.data()),
             header.data_size_);

  if (!file) {
    std::cerr << "Error writing WAV data to " << filename << std::endl;
  }
}

}  // namespace wav_processing