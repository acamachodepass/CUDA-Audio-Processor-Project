# Audio Processor using CUDA Runtime Library, NVIDIA Performance Primitives (NPP) and cuFFT (CUDA Fast Fourier Transform)

## Overview

This project implements a high-performance audio processing pipeline on the GPU to apply timbre-altering effects to WAV audio files. It leverages several key NVIDIA technologies, including the CUDA Runtime, NVIDIA Performance Primitives (NPP) for signal processing, and the cuFFT library for fast Fourier transforms.

The initial exploration for this project was driven by a creative question: **Can the sound of a harpsichord in an audio file be algorithmically modified to sound like an electric guitar?**

After initial research and experimentation, it became clear that this goal is not achievable with the techniques employed in this project. The transformation is far more complex than simple frequency manipulation. The characteristic sound of an electric guitar involves highly complex, non-linear phenomena such as distortion (which adds rich new harmonics), feedback, and unique attack/decay envelopes that cannot be synthesized by just re-weighting existing frequencies. The FFT-based approach can change the *timbre* (the tonal quality), but it cannot fundamentally create the new harmonic structures and dynamic responses that define a different instrument.

Instead, the project pivoted to a more robust and foundational goal: **to develop a complete CUDA pipeline that combines NVIDIA NPP and cuFFT to apply general-purpose timbre change effects on audio data**. This provides a powerful framework for various signal processing experiments.

The final application processes a WAV file through the following steps:

1.  **Load & Transfer:** The application loads a standard 16-bit WAV file from disk and transfers the entire audio waveform to the GPU's global memory.
2.  **STFT Processing:** The core of the program uses the Short-Time Fourier Transform (STFT) method. The long audio signal is processed in small, overlapping chunks called "frames."
3.  **GPU Pipeline per Frame:** For each frame, the following operations occur entirely on the GPU:
      * **De-interleaving:** Stereo audio data (Left-Right-Left-Right) is converted into a planar format (Left-Left...Right-Right...) required by cuFFT and NPP.
      * **Windowing:** A Hann window function is applied using the NPP library to prevent spectral leakage.
      * **Forward FFT:** The cuFFT library is used to transform the time-domain frame into the frequency domain.
      * **Frequency Modification:** A custom CUDA kernel modifies the frequency bins. In this project, it boosts a specific range of frequencies to alter the sound's character.
      * **Inverse FFT:** The modified frequency data is transformed back into the time domain using cuFFT.
      * **Overlap-Add:** The processed frame is windowed again and added to a final output buffer, perfectly reconstructing the audio signal without artifacts.
4.  **Save Result:** Once all frames are processed, the complete modified audio signal is transferred back to the CPU and saved as a new WAV file in the `output/` directory.

The main lesson learned from this project is the critical importance of understanding both the theoretical limits of an algorithm and the practical requirements of the hardware APIs. While a simple FFT approach cannot perform musical alchemy, it serves as an excellent foundation for building a high-performance signal processing pipeline. Furthermore, the project highlights the necessity of managing data layouts (interleaved vs. planar) and using specialized libraries like NPP and cuFFT to achieve maximum performance on the GPU.

-----

## Code Description

The source code is organized in the `src/` directory in a modular fashion, separating the file I/O logic from the core GPU processing implementation.

  * `wav_processing.h` / `wav_processing.cu`

      * **Description:** This module handles all host-side file operations for reading and writing 16-bit PCM WAV files.
      * **Main Functions:**
          * `loadWavFile()`: Opens a WAV file, reads its header to extract metadata (sample rate, channels, etc.), and reads the entire audio data block into a vector of floats.
          * `saveWavFile()`: Takes a vector of processed audio samples, constructs a valid WAV header, and writes the data to a new file.

  * `audio_processing.h` / `audio_processing.cu`

      * **Description:** This is the core of the project, containing all CUDA-related logic for audio processing on the GPU.
      * **Main Functions:**
          * `initializeAudioProcessing()`: Allocates all necessary GPU memory, creates the Hann window, and sets up the forward and inverse cuFFT plans.
          * `processAudio()`: The main entry point for the GPU work. It manages the STFT loop, launching the custom CUDA kernels and library calls for de-interleaving, windowing (NPP), FFT (cuFFT), frequency modification, and overlap-add.
          * `cleanupAudioProcessing()`: Frees all allocated GPU memory and destroys the cuFFT plans.

  * `main.cpp`

      * **Description:** The main entry point for the application. It orchestrates the overall workflow.
      * **Main Function:**
          * `main()`: Parses command-line arguments, calls `loadWavFile` to get the audio data, invokes the GPU processing pipeline via `processAudio`, saves the result with `saveWavFile`, and handles all initialization and cleanup.

-----

## Proof of Execution

The code's ability to execute on a large amount of data is demonstrated by the `run.sh` batch script. Audio signals are a form of data where a few large files represent a massive number of small, sequential data points (the audio samples).

The `run.sh` script processes **six different full-length audio files**, totaling several minutes of audio. Each audio file consists of hundreds of thousands or millions of individual samples that are processed in thousands of overlapping frames. The successful execution of this script, which processes millions of data points across multiple files without error, serves as clear evidence that the CUDA pipeline is robust and capable of handling a significant data workload.

-----

## Code Organization

The project is organized into the following directories and files:

  * **`bin/`**: This directory is currently empty but is intended to hold compiled binaries or other distributable files.
  * **`data/`**: Contains the input `.wav` audio files that will be processed.
  * **`lib/`**: This directory is currently empty but is intended to hold any third-party static or dynamic libraries.
  * **`output/`**: The directory where the modified audio files (prefixed with `modified_`) are saved after processing.
  * **`src/`**: Contains all C++ and CUDA source code (`.h`, `.cu`, `.cpp`).
  * **`video/`**: Contains a sample video showing the execution of the program.

The files in the root directory are:

  * **`audio_processor`**: The compiled executable binary of the project.
  * **`INSTALL`**: A placeholder file for installation instructions.
  * **`LICENSE`**: A placeholder file for the project's software license.
  * **`Makefile`**: The build script used to compile, clean, and run the project via `make` commands.
  * **`README.md`**: This file, providing a comprehensive overview of the project.
  * **`run.sh`**: A shell script that runs the `audio_processor` on all sample files in the `data/` directory.

-----

## Key Concepts

To understand the source code, the following key concepts are essential:

  * **Short-Time Fourier Transform (STFT):** The fundamental technique used to analyze how the frequency content of a signal changes over time. It works by dividing a long signal into shorter, overlapping frames and computing the Fourier Transform for each frame.
  * **Overlap-Add Method:** A method used to perfectly reconstruct a time-domain signal after it has been modified in the STFT domain. It works by adding the overlapping portions of the processed frames back together.
  * **Window Functions (Hann Window):** Mathematical functions applied to each frame before the FFT. This is done to reduce "spectral leakage," an artifact that occurs when a finite-length segment is extracted from a continuous signal, ensuring a more accurate frequency analysis.
  * **Interleaved vs. Planar Data Layout:** A critical concept in GPU programming. Audio data is often stored in an **interleaved** format (`LRLRLR...`). However, many high-performance libraries like cuFFT and NPP require a **planar** format (`LLLL...RRRR...`). A custom CUDA kernel in this project handles this conversion.
  * **CUDA Kernels:** Functions written in C++/CUDA that are executed in parallel by many threads on the GPU, allowing for massive performance gains on data-parallel tasks like audio processing.

-----

## Architecture

  * **Supported OS:** **Linux** (developed and tested on Ubuntu 22.04).
  * **Supported SM Architectures:** Compiled for **sm\_75** (Turing architecture) but can be recompiled for other architectures by changing the `-arch` flag in the `Makefile`.
  * **Supported CPU Architecture:** **x86\_64**.
  * **CUDA APIs used:**
      * **CUDA Runtime API:** For memory management (`cudaMalloc`, `cudaMemcpy`, etc.) and kernel launching.
      * **NVIDIA Performance Primitives (NPP):** Used for fast, element-wise signal processing operations (specifically, applying the window function).
      * **cuFFT (CUDA Fast Fourier Transform):** Used for high-performance forward and inverse FFTs on the GPU.
  * **Dependencies:**
      * NVIDIA CUDA Toolkit (version 11.x or newer recommended).
      * A compatible C++ compiler (e.g., `g++`).
      * `make` build automation tool.

-----

## Makefile Instructions

You can build and run the project using the provided `Makefile`.

  * **To clean the project** (remove object files and previous results):

    ```bash
    make clean
    ```

  * **To build the executable**:

    ```bash
    make build
    ```

  * **To run the entire batch process** (cleans, builds, and runs all audio samples):

    ```bash
    make run
    ```

    Alternatively, you can run all steps with a single command:

    ```bash
    make all
    ```

-----

## Data and Output

  * **To listen to the original input data:**
    The original `.wav` files are located in the `data/` directory. You can play them using any standard media player. For example, on a Linux desktop:

    ```bash
    # Using the default media player
    xdg-open data/harpsi-cs.wav
    ```

  * **To listen to the modified output data:**
    The processed `.wav` files are saved in the `output/` directory. You can listen to them in the same way:

    ```bash
    # Using the default media player
    xdg-open output/modified_harpsi-cs.wav
    ```

-----

## Video

A sample video of the program's execution can be found in the `video/` directory.

  * **File:** `video/CUDA Project - Proof of Exection.mp4`
  * **Description:** This video shows the terminal output from running the `make all` command, demonstrating the clean, build, and batch processing steps. You can play it using any standard video player.