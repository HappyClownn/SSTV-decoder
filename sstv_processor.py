import os
import time
import logging
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
from PIL import Image
import uuid
import wave
import struct
import io

logger = logging.getLogger(__name__)

# Define SSTV mode parameters
SSTV_MODES = {
    'Scottie1': {
        'line_duration': 0.0457,
        'sync_pulse': 0.009,
        'sync_porch': 0.001,
        'separator': 0.0015,
        'pixels': 320,
        'lines': 256,
    },
    'Scottie2': {
        'line_duration': 0.0714,
        'sync_pulse': 0.009,
        'sync_porch': 0.001,
        'separator': 0.0015,
        'pixels': 320,
        'lines': 256,
    },
    'Robot36': {
        'line_duration': 0.0387,
        'sync_pulse': 0.009,
        'sync_porch': 0.003,
        'separator': 0.0015,
        'pixels': 320,
        'lines': 240,
    },
    'Martin1': {
        'line_duration': 0.0465,
        'sync_pulse': 0.004,
        'sync_porch': 0.0015,
        'separator': 0.0015,
        'pixels': 320,
        'lines': 256,
    },
    # Adding Sheeptester's default mode (Possibly Martin M1 in their implementation)
    'Sheeptester': {
        'line_duration': 0.0465,  # Based on Martin M1
        'sync_pulse': 0.004,
        'sync_porch': 0.0015,
        'separator': 0.0015,
        'pixels': 320,
        'lines': 240,  # Common resolution for online SSTV
    }
}

# Frequency boundaries for SSTV
SYNC_FREQ = 1200  # Hz
BLACK_FREQ = 1500  # Hz
WHITE_FREQ = 2300  # Hz

# Frequency boundaries specifically for Sheeptester encoder
# These values are calibrated for the sheeptester tool
SHEEPTESTER_SYNC_FREQ = 1200  # Hz
SHEEPTESTER_BLACK_FREQ = 1500  # Hz
SHEEPTESTER_WHITE_FREQ = 2300  # Hz

def create_test_wav(output_path, duration=3.0, sample_rate=44100):
    """
    Create a test WAV file that simulates an SSTV signal
    
    Args:
        output_path: Path to save the WAV file
        duration: Duration of the WAV file in seconds
        sample_rate: Sample rate of the WAV file
        
    Returns:
        str: Path to the created WAV file
    """
    # Create time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Generate test pattern frequencies
    # This simulates SSTV sync pulses and RGB scan lines
    signal_data = np.zeros_like(t)
    
    # Add sync pulses every 0.1 seconds (simplified SSTV format)
    for i in range(int(duration * 10)):
        sync_start = i * 0.1
        sync_end = sync_start + 0.005
        signal_data[(t >= sync_start) & (t < sync_end)] = np.sin(2 * np.pi * SYNC_FREQ * t[(t >= sync_start) & (t < sync_end)])
    
    # Add varying frequencies representing image data
    line_duration = 0.03  # 30ms per line
    lines = int(duration / line_duration)
    
    for i in range(lines):
        line_start = i * line_duration + 0.005  # after sync
        
        # Red channel
        r_start = line_start
        r_end = r_start + 0.008
        freq_r = BLACK_FREQ + (WHITE_FREQ - BLACK_FREQ) * ((i % lines) / lines)
        signal_data[(t >= r_start) & (t < r_end)] += 0.5 * np.sin(2 * np.pi * freq_r * t[(t >= r_start) & (t < r_end)])
        
        # Green channel
        g_start = r_end + 0.001
        g_end = g_start + 0.008
        freq_g = BLACK_FREQ + (WHITE_FREQ - BLACK_FREQ) * (((i + lines//3) % lines) / lines)
        signal_data[(t >= g_start) & (t < g_end)] += 0.5 * np.sin(2 * np.pi * freq_g * t[(t >= g_start) & (t < g_end)])
        
        # Blue channel
        b_start = g_end + 0.001
        b_end = b_start + 0.008
        freq_b = BLACK_FREQ + (WHITE_FREQ - BLACK_FREQ) * (((i + 2*lines//3) % lines) / lines)
        signal_data[(t >= b_start) & (t < b_end)] += 0.5 * np.sin(2 * np.pi * freq_b * t[(t >= b_start) & (t < b_end)])
    
    # Normalize signal
    signal_data = np.int16(signal_data / np.max(np.abs(signal_data)) * 32767)
    
    # Write to WAV file
    wavfile.write(output_path, sample_rate, signal_data)
    
    return output_path

def detect_sstv_mode(audio_data, sample_rate):
    """Detect the SSTV mode based on sync pulses pattern"""
    # This is a simplified method for mode detection
    # Real SSTV decoder would use more sophisticated methods
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Calculate spectrogram
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=1024)
    
    # Look for sync pulse patterns
    sync_idx = np.argmin(np.abs(f - SYNC_FREQ))
    sync_power = Sxx[sync_idx, :]
    
    # Look for consistent patterns matching known modes
    # This is a simplified approach - real implementation would be more complex
    for mode_name, params in SSTV_MODES.items():
        expected_sync_interval = params['line_duration'] * sample_rate
        
        # Check if we have consistent sync pulses at expected intervals
        # This is very simplified and might not work well with real signals
        sync_pulse_indices = signal.find_peaks(sync_power, height=0.5, distance=expected_sync_interval*0.9)[0]
        
        if len(sync_pulse_indices) > 10:  # Need enough lines to confirm pattern
            intervals = np.diff(sync_pulse_indices)
            if np.std(intervals) / np.mean(intervals) < 0.2:  # Check consistency
                # If intervals match expected pattern for this mode
                if abs(np.mean(intervals) - expected_sync_interval) / expected_sync_interval < 0.1:
                    return mode_name
    
    # If no known mode is detected
    return "Unknown"

def frequency_to_color(freq, channel):
    """Convert a frequency to an RGB value based on channel"""
    # Map frequency to 0-255 range
    normalized = max(0, min(255, int((freq - BLACK_FREQ) / (WHITE_FREQ - BLACK_FREQ) * 255)))
    
    if channel == 'R':
        return (normalized, 0, 0)
    elif channel == 'G':
        return (0, normalized, 0)
    elif channel == 'B':
        return (0, 0, normalized)
    else:
        return (normalized, normalized, normalized)  # Grayscale

def decode_sstv(audio_data, sample_rate, mode):
    """Decode SSTV signal to image"""
    # This is a simplified decoder - real implementation would be more complex
    
    if mode == "Unknown" or mode not in SSTV_MODES:
        # Default to Scottie1 for unknown modes
        mode = "Scottie1"
    
    # Get mode parameters
    params = SSTV_MODES[mode]
    width = params['pixels']
    height = params['lines']
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Create empty image
    img = Image.new('RGB', (width, height), (0, 0, 0))
    pixels = img.load()
    
    # Compute the spectrogram
    nperseg = int(sample_rate * 0.01)  # 10ms segments
    noverlap = int(nperseg * 0.5)
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # At each time step, find the frequency with maximum power
    freq_indices = np.argmax(Sxx, axis=0)
    frequencies = f[freq_indices]
    
    # For each pixel, extract the corresponding frequency
    # This is a very simplified approach - real implementation would be more complex
    samples_per_line = int(params['line_duration'] * sample_rate)
    samples_per_pixel = samples_per_line / width
    
    # Process R, G, B channels
    channels = ['R', 'G', 'B']
    for y in range(min(height, len(frequencies) // (width * 3))):
        for x in range(width):
            for i, channel in enumerate(channels):
                pos = int(y * samples_per_line + x * samples_per_pixel + i * width * samples_per_pixel)
                if pos < len(frequencies):
                    freq = frequencies[pos]
                    r, g, b = frequency_to_color(freq, channel)
                    
                    # Update pixel (blend with existing value)
                    existing = pixels[x, y]
                    pixels[x, y] = (
                        (existing[0] + r) // 2,
                        (existing[1] + g) // 2,
                        (existing[2] + b) // 2
                    )
    
    return img, mode

def read_wav_file_robust(wav_path):
    """
    Read a WAV file in a robust way, handling various issues with WAV headers.
    
    Args:
        wav_path: Path to the WAV file
        
    Returns:
        tuple: (sample_rate, audio_data)
    """
    try:
        # First try scipy's wavfile.read which is fast but less robust
        sample_rate, audio_data = wavfile.read(wav_path)
        return sample_rate, audio_data
    except Exception as e:
        logger.warning(f"Could not read WAV with scipy: {str(e)}")
        
        try:
            # Fall back to the wave module which is more lenient
            with wave.open(wav_path, 'rb') as wf:
                # Get WAV properties
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # Read all frames
                frame_data = wf.readframes(n_frames)
                
                # Convert binary data to numpy array
                if sample_width == 1:  # 8-bit PCM is unsigned
                    dtype = np.uint8
                elif sample_width == 2:  # 16-bit PCM is signed
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit PCM
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Convert data to numpy array
                audio_data = np.frombuffer(frame_data, dtype=dtype)
                
                # Reshape if stereo
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    
                return sample_rate, audio_data
                
        except Exception as e2:
            logger.error(f"Also failed with wave module: {str(e2)}")
            
            # Final attempt: try to fix the WAV file
            try:
                logger.info("Attempting to fix WAV file")
                # Create a fixed copy of the WAV file
                fixed_path = wav_path + ".fixed.wav"
                
                # Read raw audio data
                with open(wav_path, 'rb') as f:
                    data = f.read()
                
                # Check if it's a WAV file (starts with RIFF header)
                if data[:4] != b'RIFF':
                    # Try direct raw audio import if it doesn't have a RIFF header
                    # This might help with files from the sheeptester SSTV encoder
                    logger.warning("No RIFF header found, trying raw audio import")
                    
                    # Assume it's 16-bit signed PCM at 44100Hz (common for web audio)
                    assumed_sample_rate = 44100
                    try:
                        # Try reading as raw PCM
                        raw_audio = np.frombuffer(data, dtype=np.int16)
                        return assumed_sample_rate, raw_audio
                    except:
                        # If that fails, try assuming it's floating point
                        try:
                            raw_audio = np.frombuffer(data, dtype=np.float32)
                            # Convert to int16 scale
                            raw_audio = (raw_audio * 32767).astype(np.int16)
                            return assumed_sample_rate, raw_audio
                        except:
                            raise ValueError("Not a valid WAV file and couldn't interpret as raw audio")
                
                # Fix the nAvgBytesPerSec in the header if that's the issue
                # For PCM, it should be (sample_rate * channels * bits_per_sample / 8)
                try:
                    if b'fmt ' in data:
                        fmt_pos = data.find(b'fmt ') + 4
                        chunk_size = struct.unpack('<I', data[fmt_pos:fmt_pos+4])[0]
                        
                        # PCM format has a minimum chunk size of 16
                        if chunk_size >= 16:
                            # Get format parameters
                            fmt_data = data[fmt_pos+4:fmt_pos+4+chunk_size]
                            
                            # Calculate the correct file parameters
                            format_tag = struct.unpack('<H', fmt_data[0:2])[0]
                            channels = struct.unpack('<H', fmt_data[2:4])[0]
                            sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                            avg_bytes_per_sec = struct.unpack('<I', fmt_data[8:12])[0]
                            block_align = struct.unpack('<H', fmt_data[12:14])[0]
                            bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0] if chunk_size >= 16 else 16
                            
                            # Calculate the corrected values
                            correct_block_align = channels * bits_per_sample // 8
                            correct_bytes_per_sec = sample_rate * correct_block_align
                            
                            logger.info(f"Fixing WAV header: sample_rate={sample_rate}, "
                                        f"channels={channels}, bits={bits_per_sample}, "
                                        f"block_align={block_align}->{correct_block_align}, "
                                        f"bytes_per_sec={avg_bytes_per_sec}->{correct_bytes_per_sec}")
                            
                            # Create fixed data
                            fixed_data = bytearray(data)
                            
                            # Replace nBlockAlign (bytes 12-13 of fmt chunk)
                            fixed_data[fmt_pos+4+12:fmt_pos+4+14] = struct.pack('<H', correct_block_align)
                            
                            # Replace nAvgBytesPerSec (bytes 8-11 of fmt chunk)
                            fixed_data[fmt_pos+4+8:fmt_pos+4+12] = struct.pack('<I', correct_bytes_per_sec)
                            
                            # Write fixed WAV file
                            with open(fixed_path, 'wb') as f:
                                f.write(fixed_data)
                            
                            # Try to read the fixed file
                            sample_rate, audio_data = wavfile.read(fixed_path)
                            
                            # Clean up the fixed file
                            try:
                                os.remove(fixed_path)
                            except:
                                pass
                                
                            return sample_rate, audio_data
                except Exception as e_fix:
                    logger.error(f"Error during WAV header fix: {str(e_fix)}")
                
                # If we get here, try more aggressive fixes for sheeptester encoder format
                try:
                    # Extract raw PCM data and rebuild a valid WAV file from scratch
                    data_pos = data.find(b'data')
                    if data_pos > 0:
                        # Skip 'data' and length (8 bytes)
                        data_pos += 8
                        audio_data_raw = data[data_pos:]
                        
                        # Assume 44100Hz, 16-bit PCM (common web audio)
                        sample_rate_fix = 44100
                        audio_array = np.frombuffer(audio_data_raw, dtype=np.int16)
                        
                        # Write a completely new WAV file
                        wavfile.write(fixed_path, sample_rate_fix, audio_array)
                        
                        # Read it back
                        sample_rate, audio_data = wavfile.read(fixed_path)
                        
                        # Clean up
                        try:
                            os.remove(fixed_path)
                        except:
                            pass
                            
                        return sample_rate, audio_data
                except Exception as e_scrape:
                    logger.error(f"Failed to extract PCM from WAV: {str(e_scrape)}")
                
                # One last attempt - try an even more aggressive approach for sheeptester
                try:
                    # Sheeptester might be using Web Audio API which can produce different formats
                    # Create a completely new 44.1kHz 16-bit wav file from whatever data we found
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    # Only take the valid portion
                    max_size = (len(data) // 2) * 2  # Ensure even number of bytes for int16
                    audio_array = audio_array[:max_size//2]  # Convert bytes to int16 samples count
                    
                    wavfile.write(fixed_path, 44100, audio_array)
                    sample_rate, audio_data = wavfile.read(fixed_path)
                    
                    # Clean up
                    try:
                        os.remove(fixed_path)
                    except:
                        pass
                        
                    return sample_rate, audio_data
                    
                except Exception as e_desperate:
                    logger.error(f"All WAV recovery methods failed: {str(e_desperate)}")
                    
            except Exception as e3:
                logger.error(f"Failed to fix WAV file: {str(e3)}")
            
            # If all else fails, create a minimal audio array so we don't crash
            logger.error("Creating default audio array to avoid complete failure")
            return 44100, np.zeros(44100, dtype=np.int16)  # Return 1 second of silence

def try_sheeptester_decode(audio_data, sample_rate):
    """
    Special decoder for files from the sheeptester SSTV encoder.
    This is a more direct implementation tailored to their encoder.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Image: Decoded image
    """
    logger.info("Attempting sheeptester-specific decoding")
    
    # After analyzing sheeptester.github.io/javascripts/sstv-encoder.html:
    # 1. It encodes images line by line
    # 2. Each line starts with a 1200Hz sync pulse
    # 3. RGB values are encoded with frequencies from 1500Hz to 2300Hz
    # 4. Image is encoded in RGB order (not YUV like traditional SSTV)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize audio
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    # Apply a bandpass filter to focus on the SSTV frequencies (1200-2300 Hz)
    sos = signal.butter(10, [1000, 2500], 'bandpass', fs=sample_rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data)
    
    # Extract the instantaneous frequency using Hilbert transform
    # This closely matches how the web audio encoder works
    analytic_signal = signal.hilbert(filtered_audio)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Calculate instantaneous frequency
    # Use a smaller window for more accurate frequency detection
    window_size = 20  # samples
    inst_freq = np.zeros_like(instantaneous_phase)
    
    for i in range(window_size, len(instantaneous_phase)):
        inst_freq[i] = (instantaneous_phase[i] - instantaneous_phase[i-window_size]) / \
                        (window_size / sample_rate) / (2 * np.pi)
    
    # Remove noise and out-of-range frequencies
    inst_freq = np.clip(inst_freq, 1000, 2500)  # Focus on SSTV frequency range
    
    # Detect sync pulses (1200 Hz)
    # Sheeptester uses a fixed sync pulse duration
    sync_pulse_mask = (inst_freq >= 1150) & (inst_freq <= 1250)
    
    # Find sync pulse locations (beginning of each line)
    # Function to find consistent pulses
    def find_sync_pulses(sync_mask, min_gap=2000):
        sync_indices = []
        pulse_count = 0
        in_pulse = False
        
        for i in range(len(sync_mask)):
            if sync_mask[i] and not in_pulse:
                # Start of new pulse
                if len(sync_indices) == 0 or (i - sync_indices[-1] > min_gap):
                    sync_indices.append(i)
                    pulse_count += 1
                in_pulse = True
            elif not sync_mask[i]:
                in_pulse = False
        
        return np.array(sync_indices)
    
    # Find sync pulse starts with appropriate minimum gap
    # This is crucial for getting the right number of lines
    sync_starts = find_sync_pulses(sync_pulse_mask, min_gap=sample_rate // 10)  # At least 100ms between lines
    
    # We may need to adjust for image height - Sheeptester typically uses 256 lines
    desired_height = 256
    if len(sync_starts) > desired_height:
        # If we detected too many, select the strongest ones
        logger.info(f"Found {len(sync_starts)} sync pulses, filtering to {desired_height}")
        sync_strength = np.zeros(len(sync_starts))
        
        for i, start in enumerate(sync_starts):
            end = start + 100  # Check next 100 samples
            if end < len(sync_pulse_mask):
                sync_strength[i] = np.sum(sync_pulse_mask[start:end])
        
        # Get the indices of strongest pulses
        strongest_indices = np.argsort(sync_strength)[-desired_height:]
        strongest_indices.sort()  # Keep in chronological order
        sync_starts = sync_starts[strongest_indices]
    
    if len(sync_starts) < 10:
        logger.warning(f"Not enough sync pulses found ({len(sync_starts)}), trying alternative method")
        raise ValueError("Too few sync pulses")
    
    # Line timing parameters (inferred from sheeptester code)
    line_duration_samples = int(0.5 * sample_rate)  # Estimate line duration in samples
    if len(sync_starts) > 1:
        avg_line_spacing = np.median(np.diff(sync_starts))
        line_duration_samples = int(avg_line_spacing)
    
    # Create image
    width = 320  # Standard SSTV width
    height = len(sync_starts)
    img = Image.new('RGB', (width, height), (0, 0, 0))
    pixels = img.load()
    
    # Map frequency to color (Sheeptester uses linear mapping)
    # In sheeptester implementation, black = 1500Hz, white = 2300Hz
    min_freq = 1500
    max_freq = 2300
    freq_range = max_freq - min_freq
    
    logger.info(f"Decoding with {height} lines, line duration: {line_duration_samples/sample_rate:.3f}s")
    
    # Process each line
    for y, line_start in enumerate(sync_starts):
        if y >= height:
            break
            
        # Skip the sync pulse
        pixel_start = line_start + int(0.01 * sample_rate)  # Skip ~10ms for sync
        
        # Calculate pixel duration for this line
        # If this isn't the last line, use the time to the next line
        if y < len(sync_starts) - 1:
            next_line = sync_starts[y + 1]
            effective_line_duration = next_line - pixel_start
        else:
            effective_line_duration = line_duration_samples
        
        # Ensure we don't go beyond the audio data
        effective_line_duration = min(effective_line_duration, len(inst_freq) - pixel_start)
        
        # Each pixel gets an equal share of the line time after sync pulse
        # Split line into RGB channels (r, g, b in sequence)
        pixels_per_channel = width
        samples_per_pixel = effective_line_duration / (3 * pixels_per_channel)
        
        # Process R channel
        r_start = pixel_start
        g_start = r_start + int(effective_line_duration // 3)
        b_start = g_start + int(effective_line_duration // 3)
        
        for x in range(width):
            # Get R frequency
            r_idx = r_start + int(x * samples_per_pixel)
            if r_idx < len(inst_freq):
                r_freq = inst_freq[r_idx]
                r_val = max(0, min(255, int((r_freq - min_freq) / freq_range * 255)))
            else:
                r_val = 0
            
            # Get G frequency
            g_idx = g_start + int(x * samples_per_pixel)
            if g_idx < len(inst_freq):
                g_freq = inst_freq[g_idx]
                g_val = max(0, min(255, int((g_freq - min_freq) / freq_range * 255)))
            else:
                g_val = 0
            
            # Get B frequency
            b_idx = b_start + int(x * samples_per_pixel)
            if b_idx < len(inst_freq):
                b_freq = inst_freq[b_idx]
                b_val = max(0, min(255, int((b_freq - min_freq) / freq_range * 255)))
            else:
                b_val = 0
            
            # Set pixel color
            pixels[x, y] = (r_val, g_val, b_val)
    
    return img, "Sheeptester"
    
def try_direct_frequency_decoder(audio_data, sample_rate):
    """
    A direct frequency-to-pixel decoder specifically for Sheeptester files.
    This works by directly measuring the frequency at each point in time,
    without using spectrograms or advanced signal processing.
    
    Args:
        audio_data: Audio data
        sample_rate: Sample rate
    
    Returns:
        Image: Decoded image
    """
    logger.info("Trying direct frequency decoder for Sheeptester")
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
    
    # Filter to remove noise and focus on SSTV frequency range
    sos = signal.butter(4, [1000, 2500], 'bandpass', fs=sample_rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data)
    
    # Create image with standard SSTV dimensions
    width = 320
    height = 240
    img = Image.new('RGB', (width, height), (0, 0, 0))
    pixels = img.load()
    
    # Frequency to color mapping
    min_freq = 1500  # Sheeptester uses 1500Hz for black
    max_freq = 2300  # Sheeptester uses 2300Hz for white
    freq_range = max_freq - min_freq
    
    # Zero crossing method for frequency detection
    # This is simpler but can be more robust for these types of signals
    def estimate_frequency(audio_segment, fs):
        # Count zero crossings
        zero_crossings = np.where(np.diff(np.signbit(audio_segment)))[0]
        if len(zero_crossings) < 2:
            return 0
        # Each pair of zero crossings represents half a cycle
        avg_period = np.mean(np.diff(zero_crossings)) * 2 / fs
        return 1.0 / avg_period if avg_period > 0 else 0
    
    # Determine the time allocated per image
    total_time = len(audio_data) / sample_rate
    # Sheeptester often uses a time size of around 5-10 seconds per image
    samples_per_height = len(audio_data) // height
    samples_per_width = samples_per_height // width // 3  # Divide by 3 for R,G,B channels
    
    # Process image line by line
    for y in range(height):
        line_start = y * samples_per_height
        
        # Calculate the start of each channel for this line
        r_start = line_start
        g_start = r_start + (samples_per_height // 3)
        b_start = g_start + (samples_per_height // 3)
        
        for x in range(width):
            # Calculate the precise sample position for each pixel
            # Use a window large enough to get good frequency resolution
            window_size = min(samples_per_width, 200)  # Limit window size
            
            # Calculate frequency for each RGB channel
            r_pos = r_start + (x * samples_per_width)
            g_pos = g_start + (x * samples_per_width)
            b_pos = b_start + (x * samples_per_width)
            
            # Ensure we stay within bounds
            if r_pos + window_size <= len(filtered_audio) and \
               g_pos + window_size <= len(filtered_audio) and \
               b_pos + window_size <= len(filtered_audio):
                
                # Get frequency for each channel
                r_freq = estimate_frequency(filtered_audio[r_pos:r_pos+window_size], sample_rate)
                g_freq = estimate_frequency(filtered_audio[g_pos:g_pos+window_size], sample_rate)
                b_freq = estimate_frequency(filtered_audio[b_pos:b_pos+window_size], sample_rate)
                
                # Map frequencies to RGB values (0-255)
                r_val = max(0, min(255, int((r_freq - min_freq) / freq_range * 255)))
                g_val = max(0, min(255, int((g_freq - min_freq) / freq_range * 255)))
                b_val = max(0, min(255, int((b_freq - min_freq) / freq_range * 255)))
                
                # Set pixel color
                pixels[x, y] = (r_val, g_val, b_val)
    
    return img, "Sheeptester-Direct"
                

def process_sstv_file(wav_path, output_folder, base_filename):
    """Process an SSTV WAV file and return results"""
    start_time = time.time()
    original_file = os.path.basename(wav_path)
    logger.info(f"Processing SSTV file: {original_file}")
    
    try:
        # Read WAV file with robust method
        sample_rate, audio_data = read_wav_file_robust(wav_path)
        logger.info(f"Audio data: shape={audio_data.shape}, sample_rate={sample_rate}")
        
        # Try multiple decoding methods and save the best result
        results = []
        
        # Method 1: Try our specialized sheeptester decoder first
        try:
            logger.info("Method 1: Specialized Sheeptester decoder")
            img, mode = try_sheeptester_decode(audio_data, sample_rate)
            
            # Save result
            image_filename = f"{base_filename}_method1.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path)
            
            results.append({
                'image': img,
                'mode': mode,
                'method': 'Sheeptester specialized',
                'path': image_path
            })
            
            logger.info("Method 1 successful")
        except Exception as e1:
            logger.warning(f"Method 1 failed: {str(e1)}")
        
        # Method 2: Try direct frequency decoder
        try:
            logger.info("Method 2: Direct frequency decoder")
            img, mode = try_direct_frequency_decoder(audio_data, sample_rate)
            
            # Save result
            image_filename = f"{base_filename}_method2.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path)
            
            results.append({
                'image': img,
                'mode': mode,
                'method': 'Direct frequency',
                'path': image_path
            })
            
            logger.info("Method 2 successful")
        except Exception as e2:
            logger.warning(f"Method 2 failed: {str(e2)}")
            
        # Method 3: Standard SSTV decoder as fallback
        try:
            logger.info("Method 3: Standard SSTV decoder")
            sstv_mode = detect_sstv_mode(audio_data, sample_rate)
            img, mode = decode_sstv(audio_data, sample_rate, sstv_mode)
            
            # Save result
            image_filename = f"{base_filename}_method3.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path)
            
            results.append({
                'image': img,
                'mode': mode,
                'method': 'Standard',
                'path': image_path
            })
            
            logger.info("Method 3 successful")
        except Exception as e3:
            logger.warning(f"Method 3 failed: {str(e3)}")
        
        # Check if we have any successful results
        if not results:
            raise ValueError("All decoding methods failed")
            
        # Select the best result - for now just use method1 since it's most specialized
        # Later we can add heuristics to pick the best one, like image entropy or color variance
        best_result = results[0] if results else None
        
        # For Sheeptester files, prefer method 1 or 2 which are specialized
        for result in results:
            if 'Sheeptester' in result['mode']:
                best_result = result
                break
        
        # Save the best result as the main output
        final_image_filename = f"{base_filename}.png"
        final_image_path = os.path.join(output_folder, final_image_filename)
        
        # Copy the best image to the final output path
        best_result['image'].save(final_image_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'image_filename': final_image_filename,
            'mode': best_result['mode'],
            'method': best_result['method'],
            'duration': round(processing_time, 2),
        }
    
    except Exception as e:
        logger.exception("Error processing SSTV file")
        return {
            'success': False,
            'error': str(e)
        }
