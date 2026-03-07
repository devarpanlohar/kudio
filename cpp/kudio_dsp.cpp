// MSVC: must define before ANY cmath include, including pybind11 internals
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

/**
 * kudio_dsp.cpp — pybind11 Python Bindings
 * ═════════════════════════════════════════
 *
 * Exposes the C++ DSP engine to Python as the `kudio_dsp` module.
 *
 * All public classes accept and return NumPy float32 arrays with shape (N, 2).
 * Internally, C++ works on interleaved stereo (L,R,L,R,...) — the binding
 * layer handles the (N,2) ↔ interleaved conversion with zero-copy when the
 * array is already C-contiguous.
 *
 * EXPOSED CLASSES
 * ───────────────
 *   kudio_dsp.FDNReverb(mode, wet, dry)     — Schroeder+FDN hybrid reverb
 *   kudio_dsp.TruePeakLimiter(ceil_db, ...)  — oversampled true-peak limiter
 *   kudio_dsp.HarmonicExciter(drive, ...)    — tube-style harmonic excitation
 *   kudio_dsp.PsychoStereo(haas_ms, ...)     — Haas + HRTF + crossfeed
 *   kudio_dsp.SpectralShaper(preset, sr)     — 6-band cinematic EQ
 *
 * BUILD
 * ─────
 *   See CMakeLists.txt and ../build.sh
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "include/fdn.hpp"
#include "include/limiter.hpp"
#include "include/harmonic.hpp"
#include "include/psycho.hpp"
#include "include/spectral.hpp"
#include "include/allpass.hpp"

namespace py = pybind11;
using arr_f  = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ─── Helper: (N,2) ndarray → interleaved float* ──────────────────────────────
// We always work on a copy to avoid modifying the caller's array accidentally.
static inline std::vector<float> to_interleaved(const arr_f& a) {
    auto buf = a.request();
    const float* ptr = static_cast<const float*>(buf.ptr);
    const int N = static_cast<int>(buf.shape[0]);
    std::vector<float> v(N * 2);
    if (buf.ndim == 2 && buf.shape[1] == 2) {
        std::memcpy(v.data(), ptr, N * 2 * sizeof(float));
    } else {
        throw std::runtime_error("kudio_dsp: expected (N,2) float32 array");
    }
    return v;
}

// ─── Helper: interleaved float* → (N,2) ndarray ──────────────────────────────
static inline arr_f from_interleaved(const std::vector<float>& v, int N) {
    arr_f out({static_cast<py::ssize_t>(N),
               static_cast<py::ssize_t>(2)});
    std::memcpy(out.mutable_data(), v.data(), N * 2 * sizeof(float));
    return out;
}


// ═════════════════════════════════════════════════════════════════════════════
//  MODULE DEFINITION
// ═════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(kudio_dsp, m) {
    m.doc() = R"pbdoc(
        kudio_dsp — C++ DSP Engine for Kudio
        =====================================
        All classes accept/return NumPy float32 arrays of shape (N, 2).
    )pbdoc";


    // ── FDNReverb ─────────────────────────────────────────────────────────
    py::class_<HybridReverb32>(m, "FDNReverb", R"pbdoc(
        32-line Schroeder+FDN Hybrid Reverb.

        Parameters
        ----------
        mode : str
            "hall" | "cathedral" | "plate" | "chamber"
        wet : float
            Overall wet gain (0.0–1.0). Use -1 to keep preset default.
        dry : float
            Dry gain (0.0–1.0). Use -1 to keep preset default.
        sr  : int
            Sample rate (default 44100).
    )pbdoc")
        .def(py::init([](const std::string& mode, float wet, float dry, int sr) {
                 return new HybridReverb32(mode, wet, dry, sr, 2048);
             }),
             py::arg("mode") = "hall",
             py::arg("wet")  = -1.0f,
             py::arg("dry")  = -1.0f,
             py::arg("sr")   = 44100)

        .def("process", [](HybridReverb32& self, arr_f x) -> arr_f {
                 auto v = to_interleaved(x);
                 const int N = static_cast<int>(v.size() / 2);
                 self.process(v.data(), N);
                 return from_interleaved(v, N);
             },
             py::arg("x"),
             "Process one stereo chunk. Input/output: (N,2) float32.")

        .def("set_wet",   &HybridReverb32::set_wet)
        .def("set_dry",   &HybridReverb32::set_dry)
        .def("reset",     &HybridReverb32::reset);


    // ── TruePeakLimiter ───────────────────────────────────────────────────
    py::class_<TruePeakLimiter>(m, "TruePeakLimiter", R"pbdoc(
        2× oversampled true-peak brick-wall limiter.

        Parameters
        ----------
        ceil_db : float  Output ceiling in dBFS (default -1.0).
        la_ms   : float  Look-ahead in ms (default 3.0).
        rel_ms  : float  Release time in ms (default 80.0).
        sr      : int    Sample rate (default 44100).
    )pbdoc")
        .def(py::init<float, float, float, int, int>(),
             py::arg("ceil_db") = -1.0f,
             py::arg("la_ms")   =  3.0f,
             py::arg("rel_ms")  = 80.0f,
             py::arg("sr")      = 44100,
             py::arg("chunk")   = 2048)

        .def("process", [](TruePeakLimiter& self, arr_f x) -> arr_f {
                 auto v = to_interleaved(x);
                 const int N = static_cast<int>(v.size() / 2);
                 self.process(v.data(), N);
                 return from_interleaved(v, N);
             },
             py::arg("x"),
             "Process one stereo chunk. Input/output: (N,2) float32.")

        .def("reset", &TruePeakLimiter::reset);


    // ── HarmonicExciter ───────────────────────────────────────────────────
    py::class_<HarmonicExciter>(m, "HarmonicExciter", R"pbdoc(
        Parallel tube-style harmonic exciter.

        Parameters
        ----------
        drive      : float  Nonlinear drive (0–1, default 0.35).
        even_mix   : float  Even harmonic weight (0–1, default 0.6).
        odd_mix    : float  Odd harmonic weight  (0–1, default 0.4).
        wet        : float  Exciter blend (0–1, default 0.22).
        excite_hz  : float  HP knee of exciter path Hz (default 2000).
        sr         : int    Sample rate (default 44100).
    )pbdoc")
        .def(py::init<float, float, float, float, float, int>(),
             py::arg("drive")     = 0.35f,
             py::arg("even_mix")  = 0.60f,
             py::arg("odd_mix")   = 0.40f,
             py::arg("wet")       = 0.22f,
             py::arg("excite_hz") = 2000.0f,
             py::arg("sr")        = 44100)

        .def("process", [](HarmonicExciter& self, arr_f x) -> arr_f {
                 auto v = to_interleaved(x);
                 const int N = static_cast<int>(v.size() / 2);
                 self.process(v.data(), N);
                 return from_interleaved(v, N);
             },
             py::arg("x"),
             "Process one stereo chunk. Input/output: (N,2) float32.")

        .def("set_drive", &HarmonicExciter::set_drive)
        .def("set_wet",   &HarmonicExciter::set_wet);


    // ── PsychoStereo ──────────────────────────────────────────────────────
    py::class_<PsychoStereo>(m, "PsychoStereo", R"pbdoc(
        Psychoacoustic stereo processor.
        Combines Haas ITD widening, HRTF head shadow, and crossfeed.

        Parameters
        ----------
        haas_ms     : float  ITD side-channel delay ms (default 1.2).
        crossfeed   : float  Cross-channel blend 0–0.35 (default 0.18).
        cf_delay_ms : float  Crossfeed signal delay ms (default 0.35).
        ild_cutoff  : float  Head-shadow LP cutoff Hz (default 1400).
        sr          : int    Sample rate (default 44100).
    )pbdoc")
        .def(py::init<float, float, float, float, int>(),
             py::arg("haas_ms")    =  1.2f,
             py::arg("crossfeed")  =  0.18f,
             py::arg("cf_delay_ms")=  0.35f,
             py::arg("ild_cutoff") = 1400.0f,
             py::arg("sr")         = 44100)

        .def("process", [](PsychoStereo& self, arr_f x) -> arr_f {
                 auto v = to_interleaved(x);
                 const int N = static_cast<int>(v.size() / 2);
                 self.process(v.data(), N);
                 return from_interleaved(v, N);
             },
             py::arg("x"),
             "Process one stereo chunk. Input/output: (N,2) float32.")

        .def("set_crossfeed", &PsychoStereo::set_crossfeed);


    // ── SpectralShaper ────────────────────────────────────────────────────
    py::class_<SpectralShaper>(m, "SpectralShaper", R"pbdoc(
        6-band cinematic spectral shaper.

        Parameters
        ----------
        preset : str   "cinema" | "warm" | "bright" | "neutral" (default "cinema").
        sr     : int   Sample rate (default 44100).
    )pbdoc")
        .def(py::init([](const std::string& preset_name, int sr) {
                 return new SpectralShaper(SpectralShaper::preset(preset_name), sr);
             }),
             py::arg("preset") = "cinema",
             py::arg("sr")     = 44100)

        .def("process", [](SpectralShaper& self, arr_f x) -> arr_f {
                 auto v = to_interleaved(x);
                 const int N = static_cast<int>(v.size() / 2);
                 self.process(v.data(), N);
                 return from_interleaved(v, N);
             },
             py::arg("x"),
             "Process one stereo chunk. Input/output: (N,2) float32.")

        .def("set_band", &SpectralShaper::set_band,
             py::arg("band"), py::arg("gain_db"), py::arg("sr") = 44100,
             "Adjust one band gain (0=sub,1=warmth,2=mud,3=pres,4=brill,5=air).");
}