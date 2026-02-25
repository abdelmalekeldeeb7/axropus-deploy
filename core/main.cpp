// Thin runner for the persistent engine.
//
// Responsibilities:
// - Parse model path from argv.
// - Initialize the engine once.
// - Drive the persistent decode loop (batched) until completion or SIGINT/SIGTERM.
// - Tick metrics (wall-clock sampling) without reinitializing the engine.
// - Ensure clean shutdown on exit.
//
// Non-goals:
// - No inference logic here (sampling, decoding, printing are owned by the engine).
// - No prompt ingestion or scheduling in this runner.

#include "bindings.h"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

volatile std::sig_atomic_t g_stop = 0;

void handle_signal(int) {
  g_stop = 1;
}

int32_t parse_env_i32(const char * name, int32_t default_value) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }

  char * end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (!end || end == v || *end != '\0') {
    return default_value;
  }
  if (parsed <= 0 || parsed > INT32_MAX) {
    return default_value;
  }
  return static_cast<int32_t>(parsed);
}

struct EngineGuard {
  bool inited = false;
  ~EngineGuard() {
    if (inited) {
      engine_shutdown();
    }
  }
};

}  // namespace

int main(int argc, char ** argv) {
  const char * exe = (argc > 0 && argv && argv[0]) ? argv[0] : "korith_dynamic";
  if (argc < 2 || !argv) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--prompt <text>] [--prompt-file <path>] [-c <n_ctx>] [--n-predict <n>] "
                 "[--korith-amf-path <dir>]\n",
                 exe);
    return 2;
  }

  const char * model_path = nullptr;
  const char * prompt_arg = nullptr;
  const char * prompt_file_arg = nullptr;
  int32_t n_ctx = 0;
  int32_t n_predict = 0;
  for (int i = 1; i < argc; ++i) {
    const char * arg = argv[i];
    if (!arg || arg[0] == '\0') {
      continue;
    }
    if (std::strcmp(arg, "--korith-amf-path") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: --korith-amf-path requires a directory\n");
        return 2;
      }
      (void) setenv("KORITH_AMF_PATH", argv[i + 1], 1);
      i += 1;
      continue;
    }
    if (std::strcmp(arg, "--prompt") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: --prompt requires a string\n");
        return 2;
      }
      prompt_arg = argv[i + 1];
      i += 1;
      continue;
    }
    if (std::strcmp(arg, "--prompt-file") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: --prompt-file requires a path\n");
        return 2;
      }
      prompt_file_arg = argv[i + 1];
      i += 1;
      continue;
    }
    if (std::strcmp(arg, "-c") == 0 || std::strcmp(arg, "--ctx-size") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires an integer\n", arg);
        return 2;
      }
      char * end = nullptr;
      const long parsed = std::strtol(argv[i + 1], &end, 10);
      if (!end || end == argv[i + 1] || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        std::fprintf(stderr, "error: invalid %s value\n", arg);
        return 2;
      }
      n_ctx = static_cast<int32_t>(parsed);
      i += 1;
      continue;
    }
    if (std::strcmp(arg, "--n-predict") == 0 || std::strcmp(arg, "-n") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires an integer\n", arg);
        return 2;
      }
      char * end = nullptr;
      const long parsed = std::strtol(argv[i + 1], &end, 10);
      if (!end || end == argv[i + 1] || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        std::fprintf(stderr, "error: invalid %s value\n", arg);
        return 2;
      }
      n_predict = static_cast<int32_t>(parsed);
      i += 1;
      continue;
    }
    if (!model_path) {
      model_path = arg;
    }
  }
  if (!model_path || model_path[0] == '\0') {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--prompt <text>] [--prompt-file <path>] [-c <n_ctx>] [--n-predict <n>] "
                 "[--korith-amf-path <dir>]\n",
                 exe);
    return 2;
  }
  if (prompt_arg && prompt_arg[0] != '\0' &&
      prompt_file_arg && prompt_file_arg[0] != '\0') {
    std::fprintf(stderr, "error: use only one of --prompt or --prompt-file\n");
    return 2;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  if (prompt_file_arg && prompt_file_arg[0] != '\0') {
    (void) setenv("KORITH_PROMPT_FILE", prompt_file_arg, 1);
    (void) unsetenv("KORITH_PROMPT");
  }
  if (prompt_arg && prompt_arg[0] != '\0') {
    (void) setenv("KORITH_PROMPT", prompt_arg, 1);
    (void) unsetenv("KORITH_PROMPT_FILE");
  }
  if (n_ctx > 0) {
    const std::string v = std::to_string(n_ctx);
    (void) setenv("KORITH_N_CTX", v.c_str(), 1);
  }
  if (n_predict > 0) {
    const std::string v = std::to_string(n_predict);
    (void) setenv("KORITH_MAX_TOKENS", v.c_str(), 1);
  }

  const int32_t batch_tokens = parse_env_i32("KORITH_BATCH_TOKENS", /* default = */ 256);

  EngineGuard guard;
  if (!engine_init(model_path)) {
    std::fprintf(stderr, "error: engine_init failed\n");
    return 1;
  }
  guard.inited = true;

  int exit_code = 0;
  while (!g_stop) {
    const int32_t printed = engine_step(batch_tokens);
    engine_metrics_tick();

    if (printed < 0) {
      std::fprintf(stderr, "\nerror: engine_step failed\n");
      exit_code = 1;
      break;
    }

    // 0 means finished or no progress (engine-owned semantics).
    if (printed == 0) {
      break;
    }
  }

  return exit_code;
}
