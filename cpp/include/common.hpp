#pragma once
/**
 * common.hpp — Shared definitions for all Kudio DSP headers
 * ══════════════════════════════════════════════════════════
 * Include this FIRST in every DSP header.
 * Defines kPI_F and the RESTRICT macro exactly once.
 */

// ── π constant (no dependency on M_PI or _USE_MATH_DEFINES) ─────────────────
static constexpr float kPI_F = 3.14159265358979323846f;

// ── Restrict pointer qualifier (GCC/Clang vs MSVC) ───────────────────────────
#ifdef _MSC_VER
#  define RESTRICT __restrict
#else
#  define RESTRICT __restrict__
#endif
