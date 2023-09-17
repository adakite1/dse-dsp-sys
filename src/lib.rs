use std::{ffi::{c_double, c_int}, ptr::null_mut};

use adpcm_xq_sys::{adpcm_create_context, adpcm_encode_block, adpcm_free_context};

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

fn to_f64_audio(orig: i16) -> f64 {
    orig as f64 / 32768.0
}
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

pub fn process_mono(samples: &[i16], src_sample_rate: f64, dest_sample_rate: f64, lookahead: c_int, samples_per_block: Option<usize>, track_sample_points: &[usize]) -> (Vec<u8>, Result<Vec<usize>, TrackingError>) {
    let (samples, tracked_sample_points) = resample_mono_16bitpcm(samples, src_sample_rate, dest_sample_rate, track_sample_points);
    encode_adpcm_mono_16bitpcm(&samples, lookahead, samples_per_block, &tracked_sample_points)
}

pub fn process_mono_preserve_looping(samples: &[i16], samples_looped: &[i16], src_sample_rate: f64, dest_sample_rate: f64, lookahead: c_int, samples_per_block: Option<usize>) -> (Vec<u8>, Result<Vec<usize>, TrackingError>) {
    // Extend the loop if it's too short
    fn repetition_factor(mut x: usize, src_sample_rate: f64, dest_sample_rate: f64) -> usize {
        x = (x as f64 * (dest_sample_rate / src_sample_rate)).round() as usize;
        let mut fac = -((x as f64 - 100.0 - 1.0) / x as f64);
        if fac <= 0.0 {
            fac = 0.0;
        }
        if fac >= 100.0 {
            fac = 100.0;
        }
        println!("LOOPLEN AFTER SMPLRATE CHANGE: {} REPEATING BY: {}", x, fac.round());
        fac.round() as usize
    }
    let samples_looped_extended: Vec<i16> = std::iter::repeat_with(|| samples_looped.iter().cloned())
        .take(1 + repetition_factor(samples_looped.len(), src_sample_rate, dest_sample_rate))
        .flatten().collect();
    
    // Resample both segments separately
    let mut looped_segment_dest_sample_rate;
    let resampled_len_preview = resample_len_preview(src_sample_rate, dest_sample_rate, samples_looped.len());
    if resampled_len_preview >= 9 { // Minimum for these calculations to work
        fn get_sample_rate(desired_out_len: usize, src_sample_rate: f64, samples_looped: &[i16]) -> f64 {
            // The subtraction by 1.51 is to deal with the `ceil` in the original max_len calculation found inside r8brain
            (desired_out_len as f64 - 1.51) * (src_sample_rate / samples_looped.len() as f64)
        }
        let choice_1_desired_out_len = ((resampled_len_preview - 1) | 7) + 1;
        let choice_1 = get_sample_rate(choice_1_desired_out_len, src_sample_rate, samples_looped);
        let choice_2_desired_out_len = ((resampled_len_preview - 1 - 8) | 7) + 1;
        let choice_2 = get_sample_rate(choice_2_desired_out_len, src_sample_rate, samples_looped);
        println!("C1 DESIRED OUT LEN: {} LOOPED SMPLRATE: {} LENGTH VERIFY: {}", choice_1_desired_out_len, choice_1, resample_len_preview(src_sample_rate, choice_1, samples_looped.len()));
        println!("C2 DESIRED OUT LEN: {} LOOPED SMPLRATE: {} LENGTH VERIFY: {}", choice_2_desired_out_len, choice_2, resample_len_preview(src_sample_rate, choice_2, samples_looped.len()));
        assert!(choice_1_desired_out_len == resample_len_preview(src_sample_rate, choice_1, samples_looped.len()));
        assert!(choice_2_desired_out_len == resample_len_preview(src_sample_rate, choice_2, samples_looped.len()));
        if (choice_1 - dest_sample_rate).abs() <= (choice_2 - dest_sample_rate).abs() {
            looped_segment_dest_sample_rate = choice_1;
        } else {
            looped_segment_dest_sample_rate = choice_2;
        }
        let ratio;
        if looped_segment_dest_sample_rate >= dest_sample_rate {
            ratio = looped_segment_dest_sample_rate / dest_sample_rate;
        } else {
            ratio = dest_sample_rate / looped_segment_dest_sample_rate;
        }
        if ratio > 2_f64.powf(30.0/1200.0) { // If the change in sample rate will cause the sample to sound off by more than 30 cents, bail
            looped_segment_dest_sample_rate = dest_sample_rate;
        }
    } else {
        looped_segment_dest_sample_rate = dest_sample_rate;
    }

    let (mut samples, _) = resample_mono_16bitpcm(samples, src_sample_rate, dest_sample_rate, &[]);
    let (mut samples_looped, _) = resample_mono_16bitpcm(&samples_looped_extended, src_sample_rate, looped_segment_dest_sample_rate, &[]);
    
    // Zero-pad the front so that the end of the `samples` segment align perfectly with the start of the `samples_looped` segment
    fn prepend<T: Clone>(v: &mut Vec<T>, x: T, n: usize) {
        v.resize(v.len() + n, x);
        v.rotate_right(n);
    }
    let zero_pad_front = (8 - (samples.len() % 8)) % 8;
    prepend(&mut samples, 0, zero_pad_front);
    
    // Pad the back of the loop so that the end of the `samples_looped` segment also aligns with 8 sample points
    fn interp(x: f64, y: &[f64]) -> f64 {
        // Optimal 2x (2-point, 3rd-order) (z-form)
        let z = x - 1.0/2.0;
        let (even1, odd1) = (y[1]+y[0], y[1]-y[0]);
        let c0 = even1*0.50037842517188658;
        let c1 = odd1*1.00621089801788210;
        let c2 = even1*-0.004541102062639801;
        let c3 = odd1*-1.57015627178718420;
        return ((c3*z+c2)*z+c1)*z+c0;
    }
    // ONLY RUNS FOR SITUATIONS WHERE THE ABOVE CALCULATION CANNOT BE PERFORMED!
    //  This can introduce certain artifacts to the sound, but it *can* ensure that the outputted note is correct.
    if !samples_looped.is_empty() {
        let pad_back = (8 - (samples_looped.len() % 8)) % 8;
        let step = 1.0_f64 / (pad_back + 1) as f64;
        let y = [*samples_looped.last().unwrap() as f64, *samples_looped.first().unwrap() as f64];
        for i in 0..pad_back {
            samples_looped.push(interp((i+1) as f64 * step, &y).round() as i16);
        }
    }

    // Combine the two segments, taking note of where the loop positions have moved to
    let loop_start_in_sample_points = samples.len(); // The first sample in the loop is the sample right next to the last sample in the `samples` segment
    let loop_end_in_sample_points = samples.len() + samples_looped.len() - 1; // The last sample in the loop
    samples.extend(samples_looped); // Concat

    // Encode ADPCM
    encode_adpcm_mono_16bitpcm(&samples, lookahead, samples_per_block, &[loop_start_in_sample_points, loop_end_in_sample_points])
}

pub fn encode_adpcm_mono_16bitpcm(samples: &[i16], lookahead: c_int, samples_per_block: Option<usize>, track_sample_points: &[usize]) -> (Vec<u8>, Result<Vec<usize>, TrackingError>) {
    let mut tracked_to: Vec<Result<usize, TrackingError>> = vec![Err(TrackingError {  }); track_sample_points.len()];
    
    let preferred_samples_per_block;
    if let Some(samples_per_block) = samples_per_block {
        preferred_samples_per_block = samples_per_block;
    } else {
        preferred_samples_per_block = samples.len();
    }
    let samples_per_block = ((preferred_samples_per_block - 2) | 7) + 2;

    // Encode samples with ADPCM
    let mut average_delta: i32 = 0;
    for i in (samples.len()-1)..0 {
        average_delta -= average_delta / 8;
        average_delta += (samples[i] as i32 - samples[i-1] as i32).abs();
    }
    average_delta /= 8;
    let mut samples_adpcm: Vec<u8> = Vec::new();
    unsafe {
        let adpcmctx = adpcm_create_context(1, lookahead, 2, &mut average_delta as *mut i32);
        {
            let mut block_size = (samples_per_block - 1) / 2 + 4;
            let block_size_static = block_size;
            for (i, chunk) in samples.chunks(samples_per_block).into_iter().enumerate() {
                let mut this_block_adpcm_samples = samples_per_block; // For when the file doesn't end on a full block, extra configuration is needed
                let mut this_block_pcm_samples = samples_per_block;

                // If the last chunk is not full, the chunk needs to be filled to the next valid ADPCM-encoder input block size
                if this_block_pcm_samples > chunk.len() {
                    this_block_adpcm_samples = ((chunk.len() - 2) | 7) + 2; // Round the chunk's length up to the next valid ADPCM-encoder input block size
                    block_size = (this_block_adpcm_samples - 1) / 2 + 4;
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
                    padded_chunk = &padded_chunk_src[..];
                } else {
                    padded_chunk = chunk;
                }

                // Allocate space for the output data
                let mut adpcm_block: Vec<u8> = vec![0; block_size];
                let mut outbufsize: usize = 0;
                // Do the encoding
                adpcm_encode_block(adpcmctx, adpcm_block.as_mut_ptr(), &mut outbufsize as *mut usize, padded_chunk.as_ptr(), padded_chunk.len() as i32);

                if outbufsize != block_size {
                    panic!("Expected adpcm_encode_block() to write {} bytes but got {} bytes written instead!", block_size, outbufsize);
                }

                // Do something with the adpcm_block
                samples_adpcm.extend_from_slice(&adpcm_block);

                // See if any of the tracked indices belong to this chunk
                for (tracked, tracked_to) in track_sample_points.iter().zip(tracked_to.iter_mut()) {
                    if *tracked >= i * samples_per_block && *tracked < (i+1) * samples_per_block {
                        // If it does, the tracked index is within this chunk as well but its index would've been changed by the encoding process. Recalculate the new index and return it later
                        let index_in_this_chunk = *tracked - i * samples_per_block;
                        *tracked_to = Ok(i * block_size_static + 4 + index_in_this_chunk / 2);
                    }
                }
            }
        }
        adpcm_free_context(adpcmctx);
    }

    (samples_adpcm, tracked_to.into_iter().collect::<Result<Vec<usize>, TrackingError>>())
}

pub fn resample_len_preview(src_sample_rate: f64, dest_sample_rate: f64, n_samples: usize) -> usize {
    unsafe {
        let resampler16 = resampler16_create(src_sample_rate, dest_sample_rate, n_samples as i32, 2.0);
        resampler16_getMaxOutLen(resampler16, n_samples as i32) as usize
    }
}

pub fn resample_mono_16bitpcm(samples: &[i16], src_sample_rate: f64, dest_sample_rate: f64, track_sample_points: &[usize]) -> (Vec<i16>, Vec<usize>) {
    // Map the tracked indices to the new sample rate
    let tracked_sample_points: Vec<usize> = track_sample_points.iter().map(|&x| {
        let mut tracked = (x as f64 * (dest_sample_rate / src_sample_rate)).round() as usize;
        if tracked >= samples.len() {
            tracked = samples.len()-1;
        }
        tracked
    }).collect();

    let mut samples_processed: Vec<i16>;
    if !samples.is_empty() {
        // Resampler expects floating point samples. Convert.
        let mut samples_f64: Vec<f64> = samples.iter().map(|&x| to_f64_audio(x)).collect();

        // Resample to target sample rate.
        unsafe {
            let resampler16 = resampler16_create(src_sample_rate, dest_sample_rate, samples_f64.len() as i32, 2.0);
            let resample = |resampler16: *mut HCDSPResampler16, samples_f64: &mut Vec<f64>| {
                let mut output_array_pointer: *mut c_double = null_mut();
                let output_array_len = resampler16_process(resampler16, samples_f64.as_mut_ptr(), samples_f64.len() as i32, &mut output_array_pointer as *mut *mut c_double);
                if let None = output_array_pointer.as_ref() {
                    panic!("Something happened during resampling!");
                }
                std::slice::from_raw_parts(output_array_pointer, output_array_len as usize).iter().map(to_16bit_pcm_audio)
            };
            samples_processed = resample(resampler16, &mut samples_f64).collect();
            let max_len = resampler16_getMaxOutLen(resampler16, samples_f64.len() as i32) as usize;
            while samples_processed.len() < max_len {
                samples_processed.extend(resample(resampler16, &mut vec![0.0; samples_f64.len()])); // Flush the resampler
            }
            if samples_processed.len() > max_len {
                samples_processed.resize(max_len, 0);
            }
            println!("LATENCY {} SIZE {} ACTUAL_SIZE {}", resampler16_getLatency(resampler16), max_len, samples_processed.len());
            resampler16_destroy(resampler16);
        }
    } else {
        samples_processed = Vec::new();
    }

    (samples_processed, tracked_sample_points)
}
