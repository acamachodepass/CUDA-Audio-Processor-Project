// Main entry point for the CUDA audio processing application.
// This program loads a WAV file, processes it on the GPU to modify its timbre,
// and saves the result to a new WAV file.
//
// Usage:
//   ./executable "<Sound Name>" <input_file.wav> <output_file.wav>
//
// Example:
//   ./main "Female Speech" female_speech.wav processed_speech.wav

#include <iostream>
#include <string>
#include <vector>

#include "audio_processing.h"
#include "wav_processing.h"

int main(int argc, char* argv[]) {
  // 1. Parse and validate command-line arguments.
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " \"<Sound Name>\" <input_file.wav> <output_file.wav>"
              << std::endl;
    return -1;
  }

  std::string sound_sample_name = argv[1];
  std::string input_filename = "data/" + std::string(argv[2]);
  std::string output_filename = "output/" + std::string(argv[3]);

  // Print a header for the program output.
  std::cout << std::endl;
  std::cout << "--------------------------------------------------------------"
            << std::endl;
  std::cout << "RUNNING CUDA AUDIO PROCESSOR" << std::endl;
  std::cout << "--------------------------------------------------------------"
            << std::endl;

  // 2. Load the input WAV file.
  std::cout << "1. Processing audio file: " << sound_sample_name << std::endl;
  std::cout << "2. Loading audio from: " << input_filename << std::endl;
  wav_processing::WavInfo wav_info;
  std::vector<float> input_samples =
      wav_processing::LoadWavFile(input_filename, &wav_info);

  if (input_samples.empty()) {
    std::cerr << "Failed to load input audio." << std::endl;
    return -1;
  }

  // Print loaded audio file metadata.
  std::cout << "3. Channels: " << wav_info.num_channels_
            << ", Sample Rate: " << wav_info.sample_rate_
            << ", Samples/Ch: " << wav_info.num_samples_ << std::endl;

  // 3. Initialize the GPU processing environment.
  std::cout << "4. Initializing CUDA audio processing..." << std::endl;
  if (!audio_processing::InitializeAudioProcessing(wav_info.num_channels_,
                                                   wav_info)) {
    std::cerr << "Failed to initialize CUDA audio processing." << std::endl;
    return -1;
  }

  // 4. Process the audio data on the GPU.
  std::cout << "5. Processing audio on GPU..." << std::endl;
  std::vector<float> output_samples;
  audio_processing::ProcessAudio(input_samples, &output_samples, wav_info);

  // 5. Save the processed audio to the output WAV file.
  std::cout << "6. Saving processed audio to: " << output_filename
            << std::endl;
  wav_processing::SaveWavFile(output_filename, output_samples, wav_info);

  // 6. Clean up GPU resources.
  audio_processing::CleanupAudioProcessing();

  std::cout << "7. Audio processing complete." << std::endl;
  std::cout << "--------------------------------------------------------------"
            << std::endl;
  return 0;
}