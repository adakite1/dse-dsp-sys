#include "r8bbase.h"
#include "r8bconf.h"
#include "r8butil.h"
#include "CDSPResampler.h"

typedef struct HCDSPResampler16 HCDSPResampler16;

extern "C" {
    HCDSPResampler16 * resampler16_create( const double SrcSampleRate, const double DstSampleRate,
		const int aMaxInLen, const double ReqTransBand ) {
        return reinterpret_cast<HCDSPResampler16 *>(new r8b::CDSPResampler16(SrcSampleRate, DstSampleRate, aMaxInLen, ReqTransBand));
    }
    void resampler16_destroy( HCDSPResampler16 * resampler16 ) {
        delete reinterpret_cast<r8b::CDSPResampler16 *>(resampler16);
    }
    void resampler16_clear( HCDSPResampler16 * resampler16 ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->clear();
    }
    int resampler16_getInLenBeforeOutPos( HCDSPResampler16 * resampler16, const int ReqOutPos ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->getInLenBeforeOutPos(ReqOutPos);
    }
    int resampler16_getLatency( HCDSPResampler16 * resampler16 ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->getLatency();
    }
    double resampler16_getLatencyFrac( HCDSPResampler16 * resampler16 ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->getLatencyFrac();
    }
    int resampler16_getMaxOutLen( HCDSPResampler16 * resampler16, const int MaxInLen ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->getMaxOutLen(MaxInLen);
    }
    int resampler16_process( HCDSPResampler16 * resampler16, double * ip, int l0, double* * op ) {
        return reinterpret_cast<r8b::CDSPResampler16 *>(resampler16)->process(ip, l0, *op);
    }
}