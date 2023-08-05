use std::{ffi::{c_double, c_int, c_void}, ptr::null_mut};

// Bindings to the adpcm-xq encoder
extern "C" {
    pub fn adpcm_create_context(num_channels: c_int, lookahead: c_int, noise_shaping: c_int, initial_deltas: *mut i32) -> *mut c_void;
    pub fn adpcm_encode_block(p: *mut c_void, outbuf: *mut u8, outbufsize: *mut usize, inbuf: *const i16, inbufcount: c_int) -> c_int;
    pub fn adpcm_decode_block(outbuf: *mut i16, inbuf: *const u8, inbufsize: usize, channels: c_int) -> c_int;
    pub fn adpcm_free_context(p: *mut c_void);
}

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

pub fn process_mono(mut samples: Vec<i16>, src_sample_rate: f64, dest_sample_rate: f64, samples_per_block: usize) -> usize {
    // Resampler expects floating point samples. Convert.
    let mut samples_f64: Vec<f64> = samples.into_iter().map(to_f64_audio).collect();
    // Resample to target sample rate.
    unsafe {
        let resampler16 = resampler16_create(src_sample_rate, dest_sample_rate, samples_f64.len() as i32, 2.0);
        let mut output_array_pointer: *mut c_double = null_mut();
        let output_array_len = resampler16_process(resampler16, samples_f64.as_mut_ptr(), samples_f64.len() as i32, &mut output_array_pointer as *mut *mut c_double);
        if let None = output_array_pointer.as_ref() {
            panic!("Something happened during resampling!");
        }
        samples = std::slice::from_raw_parts(output_array_pointer, output_array_len as usize).iter().map(to_16bit_pcm_audio).collect();
        resampler16_destroy(resampler16);
    }
    // Encode samples with ADPCM
    let mut average_delta: i32 = 0;
    for i in (samples.len()-1)..0 {
        average_delta -= average_delta / 8;
        average_delta += (samples[i] as i32 - samples[i-1] as i32).abs();
    }
    average_delta /= 8;
    let mut samples_adpcm: Vec<u8> = Vec::new();
    unsafe {
        let adpcmctx = adpcm_create_context(1, 3, 2, &mut average_delta as *mut i32);
        {
            let mut block_size = (samples_per_block - 1) / 2 + 4;
            for chunk in samples.chunks(samples_per_block).into_iter() {
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
            }
            // adpcm_encode_block(adpcmctx, outbuf, outbufsize, inbuf, inbufcount);
        }
        adpcm_free_context(adpcmctx);
    }
    0
}
