#include "sight-log.h"

#include <cstdarg>
#include <cstdio>

namespace {
constexpr size_t kLineCap = 1024;
void (*g_sink)(int level, const char *msg) = nullptr;
} // namespace

void sight_log_set_sink(void (*sink)(int level, const char *msg))
{
    g_sink = sink;
}

void sight_logf(int level, const char *format, ...)
{
    char line[kLineCap];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);

    if (g_sink) {
        g_sink(level, line);
        return;
    }
    fprintf(stderr, "%s\n", line);
}
