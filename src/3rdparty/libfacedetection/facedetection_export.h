#pragma once

#if defined(_WIN32)
#  if defined(FACEDETECTION_EXPORT_SYMBOLS)
#    define FACEDETECTION_EXPORT __declspec(dllexport)
#  else
#    define FACEDETECTION_EXPORT __declspec(dllimport)
#  endif
#else
#  define FACEDETECTION_EXPORT
#endif

// Helper to avoid issues if not defined
#ifndef FACEDETECTION_EXPORT
#define FACEDETECTION_EXPORT
#endif
