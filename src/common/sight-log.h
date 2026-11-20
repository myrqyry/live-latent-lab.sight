#pragma once

// Shared logging hook for Sight processors that are compiled both into the
// OBS plugin and into standalone ONNX test binaries. printf/std::cout never
// reach the OBS log, and libobs cannot be linked into the test binaries, so
// processors call sight_logf() and the plugin installs a blog-backed sink at
// module load. The default sink writes to stderr, which ctest and the smoke
// scripts already capture. Log level constants and bstrdup/bfree come from
// libobs util/base.h, which is header-only for those symbols and linkable
// without the rest of libobs.

void sight_log_set_sink(void (*sink)(int level, const char *msg));

#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
void sight_logf(int level, const char *format, ...);
