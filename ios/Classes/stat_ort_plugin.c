// Relative imports so the shared C sources in ../../src are compiled into the
// iOS build. The podspec's source_files only globs Classes/, so this forwarder
// pulls in every translation unit (a unity build). See ../stat_ort_plugin.podspec.
#include "../../src/compute_fft.c"
#include "../../src/mel.c"
#include "../../src/normalize_white_space.c"
#include "../../src/load_vocab.c"
#include "../../src/vaani_utils.c"
#include "../../src/vaani_core.c"
#include "../../src/vaani_decode.c"
#include "../../src/vaani_vad.c"
#include "../../src/vaani_speaker.c"
#include "../../src/vaani_stream.c"
#include "../../src/stat_ort_plugin.c"
