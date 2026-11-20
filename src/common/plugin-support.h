#pragma once

// Logging macros for OBS plugin
#include <obs-module.h>

#define LOG_INFO(...) blog(LOG_INFO, "[style-transfer] " __VA_ARGS__)
#define LOG_WARNING(...) blog(LOG_WARNING, "[style-transfer] " __VA_ARGS__)
#define LOG_ERROR(...) blog(LOG_ERROR, "[style-transfer] " __VA_ARGS__)

// Add more plugin support macros/utilities here as needed
