#!/bin/bash
# This script runs the audio_processor on a predefined list of WAV files.

# Exit immediately if any command fails
set -e

EXECUTABLE=./audio_processor

# Check if the executable exists before running
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found. Please run 'make build' first."
    exit 1
fi

# A helper function to call the audio processor with the correct arguments
process_audio() {
    local sound_name="$1"
    local input_file="$2"
    local output_file="$3"
    
    # Execute the program with the provided arguments
    "$EXECUTABLE" "$sound_name" "$input_file" "$output_file"
}

# --- Process all audio files ---
process_audio "Hollow-Body Electric Guitar" "gtr-jazz-3.wav" "modified_gtr-jazz-3.wav"
process_audio "Plucked/Struck Strings" "bachfugue.wav" "modified_bachfugue.wav"
process_audio "Distorted Electric Guitar" "gtr-dist-jimi.wav" "modified_gtr-dist-jimi.wav"
process_audio "Distorted Electric Guitar + Feedback" "gtr-dist-yes.wav" "modified_gtr-dist-yes.wav.wav"
process_audio "Harpsichord" "harpsi-cs.wav" "modified_harpsi-cs.wav"
process_audio "Piano" "pno-cs.wav" "modified_pno-cs.wav"

echo ""
echo "All audio files processed successfully."
echo "Check the 'output/' directory for results."