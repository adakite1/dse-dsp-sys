use std::{ffi::{c_double, c_int}, ptr::null_mut};

use adpcm_xq_sys::{adpcm_create_context, adpcm_encode_block, adpcm_free_context};
use block_alignment::BlockAlignment;
use tracing::{debug, error, span, trace, Level};

// Bindings to the r8brain resampler
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HCDSPResampler16 {
    _unused: [u8; 0]
}
extern "C" {
    pub fn resampler16_create(SrcSampleRate: c_double, DstSampleRate: c_double, aMaxInLen: c_int, ReqTransBand: c_double ) -> *mut HCDSPResampler16;
    pub fn resampler16_destroy(resampler16: *mut HCDSPResampler16);
    pub fn resampler16_clear(resampler16: *mut HCDSPResampler16);
    pub fn resampler16_getInLenBeforeOutPos(resampler16: *mut HCDSPResampler16, ReqOutPos: c_int) -> c_int;
    pub fn resampler16_getLatency(resampler16: *mut HCDSPResampler16) -> c_int;
    pub fn resampler16_getLatencyFrac(resampler16: *mut HCDSPResampler16) -> c_double;
    pub fn resampler16_getMaxOutLen(resampler16: *mut HCDSPResampler16, MaxInLen: c_int) -> c_int;
    pub fn resampler16_process(resampler16: *mut HCDSPResampler16, ip: *mut c_double, l0: c_int, op: *mut *mut c_double) -> c_int;
}

/// Convert a 16-bit PCM sample into 64-bit floating point.
fn to_f64_audio(orig: i16) -> f64 {
    orig as f64 / 32768.0
}
/// Convert a 64-bit floating point sample into 16-bit PCM.
fn to_16bit_pcm_audio(orig: &f64) -> i16 {
    (orig * 32768.0).round() as i16
}

// Define our error types. These may be customized for our error handling cases.
// Now we will be able to write our own errors, defer to an underlying error
// implementation, or do something in between.
#[derive(Debug, Clone)]
pub struct TrackingError;

impl std::error::Error for TrackingError {  }

// Generation of an error is completely separate from how it is displayed.
// There's no need to be concerned about cluttering complex logic with the display style.
//
// Note that we don't store any extra info about the errors. This means we can't state
// which string failed to parse without modifying our types to carry that information.
impl std::fmt::Display for TrackingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Failed to track one or more indices!")
    }
}

/// Resample and ADPCM encode mono and 16-bit PCM samples.
pub fn process_mono_16bitpcm<InitDeltas>(samples: &[i16], src_sample_rate: f64, dest_sample_rate: f64, lookahead: c_int, calc_initial_deltas: InitDeltas, preferred_samples_per_block: Option<usize>, track_sample_points: &[usize]) -> (Vec<u8>, Result<Vec<usize>, TrackingError>)
where
    InitDeltas: FnOnce(&[i16]) -> i32 {
    let _span_ = span!(Level::DEBUG, "process_mono_16bitpcm", num_samples = samples.len(), src_sample_rate, dest_sample_rate, lookahead, ?preferred_samples_per_block, ?track_sample_points).entered();
    let (samples, tracked_sample_points) = resample_mono_16bitpcm(samples, src_sample_rate, dest_sample_rate, track_sample_points);
    debug!(num_samples_new = samples.len(), "Resampling done");
    adpcm_encode_mono_16bitpcm(&samples, lookahead, calc_initial_deltas, preferred_samples_per_block, &tracked_sample_points)
}

/// A collection of block alignment targets.
pub mod block_alignment {
    pub trait BlockAlignment {
        fn round_up(&self, n_samples: usize) -> usize;
        fn zero_pad_front(&self, n_samples: usize) -> usize;
        fn generate_aligned_options(&self, n_samples: usize) -> Vec<usize>;
    }
    pub struct To8Bytes();
    impl BlockAlignment for To8Bytes {
        fn round_up(&self, n_samples: usize) -> usize {
            if n_samples == 0 {
                0
            } else {
                ((n_samples - 1) | 7) + 1
            }
        }
        fn zero_pad_front(&self, n_samples: usize) -> usize {
            self.round_up(n_samples) - n_samples + 1 // The plus one is for alignment since the first sample is encoded as a part of the adpcm initializers.
        }
        fn generate_aligned_options(&self, n_samples: usize) -> Vec<usize> {
            let mut choices = Vec::with_capacity(2);
            if n_samples > 0 {
                choices.push(((n_samples - 1) | 7) + 1);
            }
            if n_samples >= 9 {
                choices.push(((n_samples - 1 - 8) | 7) + 1);
            }
            choices
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SampleRateChoicePreference {
    /// Will always pick the lower sample rate that satisfies the alignment for the looping
    Lower,
    /// Will always pick the higher sample rate that satisfies the alignment for the looping
    Higher,
    /// Will pick the sample rate closest to the originally specified sample rate
    Nearest,
    /// Will pick the sample rate that allows for the least amount of zero-padding at the front
    MinStartPad
}
/// Calculate the new sample rate that is needed in order to have a specific number of samples as output.
#[tracing::instrument(level = "trace")]
pub fn get_sample_rate_by_out_samples(src_sample_rate: f64, n_samples: usize, n_samples_target: usize) -> f64 {
    // The subtraction by 1.51 is to deal with the `ceil` in the original max_len calculation found inside r8brain
    (n_samples_target as f64 - 1.51) * (src_sample_rate / n_samples as f64)
}
/// Resample and ADPCM encode mono and 16-bit PCM samples, preserving looping by processing the non-looped and looped segments separately.
/// 
/// The `min_loop_len` specifies the target number of samples to extend the looped segment to. In practice the resulting looped segment will likely be larger than the specified minimum. A value of zero, the minimum, will always guarantee that the looped samples are processed as is, without any repetitions.
pub fn process_mono_16bitpcm_preserve_looping<InitDeltas, A>(samples: &[i16], samples_looped: &[i16], src_sample_rate: f64, dest_sample_rate: f64, lookahead: c_int, calc_initial_deltas: InitDeltas, min_loop_len: usize, block_alignment: A, sample_rate_choice_preference: SampleRateChoicePreference, preferred_samples_per_block: Option<usize>) -> (Vec<u8>, f64, Result<Vec<usize>, TrackingError>)
where
    InitDeltas: FnOnce(&[i16]) -> i32,
    A: BlockAlignment {
    let _span_ = span!(Level::DEBUG, "process_mono_16bitpcm_preserve_looping", num_samples = samples.len(), num_samples_looped = samples_looped.len(), src_sample_rate, dest_sample_rate, lookahead, min_loop_len, ?sample_rate_choice_preference, ?preferred_samples_per_block).entered();
    
    /// Calculate how many times to repeat the looped samples before resampling.
    /// 
    /// The `min_loop_len` specifies the target number of samples to extend the looped segment to. A value of zero, the minimum, will always guarantee that the looped samples are processed as is, without any repetitions.
    #[tracing::instrument(level = "trace")]
    fn repetition_factor(mut num_samples: usize, src_sample_rate: f64, dest_sample_rate: f64, min_loop_len: usize) -> usize {
        num_samples = (num_samples as f64 * (dest_sample_rate / src_sample_rate)).round() as usize;
        let mut fac = min_loop_len as f64 / num_samples as f64;
        if fac <= 0.0 {
            fac = 0.0;
        }
        if fac >= min_loop_len as f64 {
            fac = min_loop_len as f64;
        }
        trace!(num_samples_after_smplrate_change = num_samples, repeat_by = fac.round(), "Repetition factor");
        fac.round() as usize
    }
    // Extend the loop if it's too short
    let samples_looped_extended: Vec<i16> = std::iter::repeat_with(|| samples_looped.iter().cloned())
        .take(1 + repetition_factor(samples_looped.len(), src_sample_rate, dest_sample_rate, min_loop_len))
        .flatten().collect();
    
    // Resample both segments separately
    let looped_segment_dest_sample_rate;
    let resampled_len_preview = resample_len_preview(src_sample_rate, dest_sample_rate, samples_looped_extended.len());
    let choices_for_desired_out_len = block_alignment.generate_aligned_options(resampled_len_preview);
    let choices: Vec<f64> = choices_for_desired_out_len.iter().map(|x| get_sample_rate_by_out_samples(src_sample_rate, samples_looped_extended.len(), *x)).collect();
    if choices.is_empty() {
        looped_segment_dest_sample_rate = dest_sample_rate;
    } else {
        #[cfg(debug_assertions)]
        {
            let _span_verify_ = span!(Level::TRACE, "verify resample-rate candidates").entered();
            for (i, (&choice_for_desired_out_len, &choice)) in choices_for_desired_out_len.iter().zip(choices.iter()).enumerate() {
                trace!(i, choice_for_desired_out_len, choice, length_verify = resample_len_preview(src_sample_rate, choice, samples_looped_extended.len()), "Candidate for desired output length");
                assert!(choice_for_desired_out_len == resample_len_preview(src_sample_rate, choice, samples_looped_extended.len()));
            }
        }
        looped_segment_dest_sample_rate = match sample_rate_choice_preference {
            SampleRateChoicePreference::Higher => *choices.iter().max_by(|a, b| a.total_cmp(b)).unwrap(), // Safe, already checked for emptiness before.
            SampleRateChoicePreference::Lower => *choices.iter().min_by(|a, b| a.total_cmp(b)).unwrap(),
            SampleRateChoicePreference::Nearest => choices.iter().map(|&x| (x, (x - dest_sample_rate).abs())).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0,
            SampleRateChoicePreference::MinStartPad => choices.iter().map(|&x| (x, {
                // Calculate the length of the pad necessary for this sample rate.
                let resampled_len_preview = resample_len_preview(src_sample_rate, x, samples.len());
                block_alignment.zero_pad_front(resampled_len_preview)
            })).min_by(|a, b| a.1.cmp(&b.1)).unwrap().0
        };
        trace!(chosen = looped_segment_dest_sample_rate, ?sample_rate_choice_preference, among = ?choices, "Sample rate chosen");
    }

    let (mut samples, _) = resample_mono_16bitpcm(samples, src_sample_rate, looped_segment_dest_sample_rate, &[]);
    let (samples_looped, _) = resample_mono_16bitpcm(&samples_looped_extended, src_sample_rate, looped_segment_dest_sample_rate, &[]);
    
    /// Insert in-place a new element `x`, `n` times at the start of vector `v`.
    #[tracing::instrument(level = "trace")]
    fn prepend<T: Clone + std::fmt::Debug>(v: &mut Vec<T>, x: T, n: usize) {
        v.resize(v.len() + n, x);
        v.rotate_right(n);
    }
    // Zero-pad the front so that the end of the `samples` segment align perfectly with the start of the `samples_looped` segment
    let zero_pad_front = block_alignment.zero_pad_front(samples.len());
    prepend(&mut samples, 0, zero_pad_front);
    debug!(num_samples_new = samples.len(), zero_pad_front, num_samples_looped_new = samples_looped.len(), "Resampling done, non-looped segment might have been prepended with some zeroes");

    // Combine the two segments, taking note of where the loop positions have moved to
    let loop_start_in_sample_points = samples.len(); // The first sample in the loop is the sample right next to the last sample in the `samples` segment
    let loop_end_in_sample_points = samples.len() + samples_looped.len() - 1; // Inclusive range end for the loop.
    samples.extend(samples_looped); // Concat
    debug!(loop_start_in_sample_points, loop_end_in_sample_points, "Loop positions after resampling and concatenating");

    // Encode ADPCM
    let (resampled, tracking) = adpcm_encode_mono_16bitpcm(&samples, lookahead, calc_initial_deltas, preferred_samples_per_block, &[loop_start_in_sample_points, loop_end_in_sample_points]);
    debug!(size_in_bytes = resampled.len(), sample_rate = looped_segment_dest_sample_rate, "ADPCM encoding done");
    debug!(?tracking, "Final loop positions");
    (resampled, looped_segment_dest_sample_rate, tracking)
}

/// A collection of choices of algorithms for calculating the initial ADPCM deltas.
pub mod init_deltas {
    /// Calculate initial ADPCM deltas using averaging.
    #[tracing::instrument(level = "trace")]
    pub fn averaging(samples: &[i16]) -> i32 {
        let mut average_delta: i32 = 0;
        for i in (samples.len()-1)..0 {
            average_delta -= average_delta / 8;
            average_delta += (samples[i] as i32 - samples[i-1] as i32).abs();
        }
        average_delta /= 8;
        average_delta
    }
}

/// Round the provided samples per block to a valid ADPCM block size.
#[tracing::instrument(level = "trace")]
pub fn adpcm_encode_round_to_valid_block_size(preferred_samples_per_block: usize) -> usize {
    ((preferred_samples_per_block - 2) | 7) + 2
}
/// Calculate the ADPCM block size in bytes given the number of samples to be included in the block.
/// 
/// User is responsible in providing a sample count that's a valid ADPCM block size.
#[tracing::instrument(level = "trace")]
pub fn adpcm_block_size_in_bytes_preview(num_samples: usize) -> usize {
    (num_samples - 1) / 2 + 4
}

/// ADPCM encode mono and 16-bit PCM samples with the provided configurations, while tracking a series of sample points to where they land in the encoded audio.
/// 
/// The `preferred_samples_per_block` provided will be aligned to a valid ADPCM block size automatically, and the aligned value will be used instead.
pub fn adpcm_encode_mono_16bitpcm<InitDeltas>(samples: &[i16], lookahead: c_int, calc_initial_deltas: InitDeltas, preferred_samples_per_block: Option<usize>, track_sample_points: &[usize]) -> (Vec<u8>, Result<Vec<usize>, TrackingError>)
where
    InitDeltas: FnOnce(&[i16]) -> i32 {
    let _span_ = span!(Level::DEBUG, "adpcm_encode_mono_16bitpcm", num_samples = samples.len(), lookahead, preferred_samples_per_block, ?track_sample_points).entered();

    let mut tracked_to: Vec<Result<usize, TrackingError>> = vec![Err(TrackingError {  }); track_sample_points.len()];
    
    let samples_per_block = ((preferred_samples_per_block.unwrap_or(samples.len()) - 2) | 7) + 2;

    // Encode samples with ADPCM
    debug!("Calculating initial ADPCM deltas...");
    let mut initial_deltas = calc_initial_deltas(samples);
    debug!(initial_deltas, "Initial ADPCM deltas");
    let mut samples_adpcm: Vec<u8> = Vec::new();
    unsafe {
        let adpcmctx = adpcm_create_context(1, lookahead, 2, &mut initial_deltas as *mut i32);
        debug!(adpcmctx = adpcmctx as usize, lookahead, noise_shaping = 2, initial_deltas = &mut initial_deltas as *mut i32 as usize, "Created ADPCM encoding context");
        {
            let mut block_size_in_bytes = (samples_per_block - 1) / 2 + 4;
            let block_size_in_bytes_static = block_size_in_bytes;
            let _span_loop_ = span!(Level::TRACE, "adpcm chunks encoding loop").entered();
            for (i, chunk) in samples.chunks(samples_per_block).into_iter().enumerate() {
                let mut this_block_adpcm_samples = samples_per_block; // For when the file doesn't end on a full block, extra configuration is needed
                let mut this_block_pcm_samples = samples_per_block;

                // If the last chunk is not full, the chunk needs to be filled to the next valid ADPCM-encoder input block size
                if this_block_pcm_samples > chunk.len() {
                    this_block_adpcm_samples = ((chunk.len() - 2) | 7) + 2; // Round the chunk's length up to the next valid ADPCM-encoder input block size
                    trace!(final_block_num_samples_original = this_block_pcm_samples, final_block_num_samples_new = this_block_adpcm_samples, block_size_in_bytes_original = block_size_in_bytes, block_size_in_bytes_new = (this_block_adpcm_samples - 1) / 2 + 4, "Final chunk does not fill the full normal block so will be padded to the closest valid ADPCM sample count");
                    block_size_in_bytes = (this_block_adpcm_samples - 1) / 2 + 4;
                    this_block_pcm_samples = chunk.len();
                }
                let mut padded_chunk_src;
                let padded_chunk;
                if this_block_adpcm_samples > this_block_pcm_samples {
                    let last_sample = chunk[chunk.len()-1];
                    padded_chunk_src = Vec::from(chunk);
                    let dups = this_block_adpcm_samples - this_block_pcm_samples;
                    for _ in 0..dups {
                        padded_chunk_src.push(last_sample.clone());
                    }
                    trace!(dups, "Padded chunk");
                    padded_chunk = &padded_chunk_src[..];
                } else {
                    padded_chunk = chunk;
                }

                // Allocate space for the output data
                let mut adpcm_block: Vec<u8> = vec![0; block_size_in_bytes];
                let mut outbufsize: usize = 0;
                // Do the encoding
                adpcm_encode_block(adpcmctx, adpcm_block.as_mut_ptr(), &mut outbufsize as *mut usize, padded_chunk.as_ptr(), padded_chunk.len() as i32);

                if outbufsize != block_size_in_bytes {
                    error!(expected = block_size_in_bytes, got = outbufsize, "Unexpected number of bytes written in ADPCM encoded output");
                    panic!("Expected adpcm_encode_block() to write {} bytes but got {} bytes written instead!", block_size_in_bytes, outbufsize);
                }

                // Do something with the adpcm_block
                samples_adpcm.extend_from_slice(&adpcm_block);

                // See if any of the tracked indices belong to this chunk
                for (tracked, tracked_to) in track_sample_points.iter().zip(tracked_to.iter_mut()) {
                    if *tracked >= i * samples_per_block && *tracked < (i+1) * samples_per_block {
                        // If it does, the tracked index is within this chunk as well but its index would've been changed by the encoding process. Recalculate the new index and return it later
                        let index_in_this_chunk = *tracked - i * samples_per_block;
                        if index_in_this_chunk == 0 {
                            trace!(byte_position = i * block_size_in_bytes_static, chunk = i, index_in_this_chunk, "Tracked sample to ADPCM output block");
                            *tracked_to = Ok(i * block_size_in_bytes_static);
                        } else {
                            trace!(byte_position = i * block_size_in_bytes_static + 4 + (index_in_this_chunk-1) / 2, chunk = i, index_in_this_chunk, "Tracked sample to ADPCM output block");
                            *tracked_to = Ok(i * block_size_in_bytes_static + 4 + (index_in_this_chunk-1) / 2);
                        }
                    }
                }
            }
        }
        debug!("Destroying ADPCM encoding context...");
        adpcm_free_context(adpcmctx);
    }

    (samples_adpcm, tracked_to.into_iter().collect::<Result<Vec<usize>, TrackingError>>())
}

/// Calculate the maximum number of samples that will be outputted after resampling, given the resampling parameters.
#[tracing::instrument(level = "trace")]
pub fn resample_len_preview(src_sample_rate: f64, dest_sample_rate: f64, n_samples: usize) -> usize {
    unsafe {
        let resampler16 = resampler16_create(src_sample_rate, dest_sample_rate, n_samples as i32, 2.0);
        debug!(resampler16 = resampler16 as usize, "Created resampler");
        resampler16_getMaxOutLen(resampler16, n_samples as i32) as usize
    }
}
/// Calculate the position that the `x`'th sample in the original audio will land after resampling, given the resampling parameters.
#[tracing::instrument(level = "trace")]
pub fn resample_pos_preview(src_sample_rate: f64, dest_sample_rate: f64, n_samples_original: usize, x: usize) -> usize {
    let len_ratio = resample_len_preview(src_sample_rate, dest_sample_rate, n_samples_original) as f64 / n_samples_original as f64;
    (x as f64 * len_ratio).round() as usize
}

/// Resample mono and 16-bit PCM samples to the specified destination sample rate, while tracking a series of sample points to where they land in the resampled audio.
#[tracing::instrument(level = "debug")]
pub fn resample_mono_16bitpcm(samples: &[i16], src_sample_rate: f64, dest_sample_rate: f64, track_sample_points: &[usize]) -> (Vec<i16>, Vec<usize>) {
    // Map the tracked indices to the new sample rate.
    // This particular method of doing so maps 0 to 0 and in_samples_len to out_samples_len.
    let len_ratio = resample_len_preview(src_sample_rate, dest_sample_rate, samples.len()) as f64 / samples.len() as f64;
    let tracked_sample_points: Vec<usize> = track_sample_points.iter().map(|&x| (x as f64 * len_ratio).round() as usize).collect();
    debug!(from = ?track_sample_points, to = ?tracked_sample_points, "New positions of tracked sample points calculated");

    let mut samples_processed: Vec<i16>;
    if !samples.is_empty() {
        // Resampler expects floating point samples. Convert.
        let mut samples_f64: Vec<f64> = samples.iter().map(|&x| to_f64_audio(x)).collect();
        debug!("Samples converted to floating-points");

        // Resample to target sample rate.
        unsafe {
            let resampler16 = resampler16_create(src_sample_rate, dest_sample_rate, samples_f64.len() as i32, 2.0);
            debug!(resampler16 = resampler16 as usize, "Created resampler");
            let resample = |resampler16: *mut HCDSPResampler16, samples_f64: &mut Vec<f64>| {
                let _span_ = span!(Level::DEBUG, "resample", resampler16 = resampler16 as usize, num_samples = samples_f64.len()).entered();
                let mut output_array_pointer: *mut c_double = null_mut();
                let output_array_len = resampler16_process(resampler16, samples_f64.as_mut_ptr(), samples_f64.len() as i32, &mut output_array_pointer as *mut *mut c_double);
                if let None = output_array_pointer.as_ref() {
                    error!("r8brain returned null pointer for resample result");
                    panic!("Something happened during resampling!");
                }
                std::slice::from_raw_parts(output_array_pointer, output_array_len as usize).iter().map(to_16bit_pcm_audio)
            };
            samples_processed = resample(resampler16, &mut samples_f64).collect();
            let max_len = resampler16_getMaxOutLen(resampler16, samples_f64.len() as i32) as usize;
            debug!(num_samples_outputted = samples_processed.len(), num_samples_expected = max_len, flushing_required = samples_processed.len() < max_len, "Resampling...");
            while samples_processed.len() < max_len {
                samples_processed.extend(resample(resampler16, &mut vec![0.0; samples_f64.len()])); // Flush the resampler
                debug!(num_samples_outputted = samples_processed.len(), num_samples_expected = max_len, flushing_still_required = samples_processed.len() < max_len, "Resampling...");
            }
            if samples_processed.len() > max_len {
                debug!("More samples than expected, trimming down to expected length");
                samples_processed.resize(max_len, 0);
            }
            debug!(latency = resampler16_getLatency(resampler16), "Destroying resampler...");
            resampler16_destroy(resampler16);
        }
    } else {
        debug!("Zero samples provided, returning empty vec...");
        samples_processed = Vec::new();
    }

    (samples_processed, tracked_sample_points)
}
