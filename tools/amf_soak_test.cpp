#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct RunResult {
  std::uint64_t tokens_generated = 0;
  double hit_rate = std::numeric_limits<double>::quiet_NaN();
  double avg_skip_ratio = std::numeric_limits<double>::quiet_NaN();
  double avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
  double roi_slope = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t bytes = 0;
  std::uint64_t evictions = 0;
  std::uint64_t mf_updates = 0;
  std::uint64_t replay_disables = 0;
  int replay_mask = -1;
  int replay_mask_flips = 0;
  bool replay_violation = false;
  bool saw_health = false;
  bool saw_summary = false;
  bool saw_amf_summary = false;
  bool ok = false;
};

std::string read_file(const std::string & path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::string content;
  std::string line;
  while (std::getline(in, line)) {
    content.append(line);
    content.push_back('\n');
  }
  return content;
}

bool parse_key_double(const std::string & line, const char * key, double & out) {
  const std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  const char * start = line.c_str() + pos + std::strlen(key);
  char * end = nullptr;
  const double v = std::strtod(start, &end);
  if (!end || end == start) {
    return false;
  }
  out = v;
  return true;
}

bool parse_key_u64(const std::string & line, const char * key, std::uint64_t & out) {
  const std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  const char * start = line.c_str() + pos + std::strlen(key);
  char * end = nullptr;
  const unsigned long long v = std::strtoull(start, &end, 10);
  if (!end || end == start) {
    return false;
  }
  out = static_cast<std::uint64_t>(v);
  return true;
}

bool parse_hex_mask(const std::string & line, const char * key, int & out) {
  const std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  const char * start = line.c_str() + pos + std::strlen(key);
  char * end = nullptr;
  const unsigned long v = std::strtoul(start, &end, 16);
  if (!end || end == start) {
    return false;
  }
  out = static_cast<int>(v);
  return true;
}

RunResult run_once(const std::string & model_path,
                   const std::string & prompt,
                   std::uint64_t n_predict) {
  RunResult result{};

  const std::string predict = std::to_string(n_predict);
  (void) setenv("KORITH_PROMPT", prompt.c_str(), 1);
  (void) setenv("KORITH_MAX_TOKENS", predict.c_str(), 1);
  (void) setenv("KORITH_BENCHMARK_MODE", "1", 1);
  (void) setenv("KORITH_MIN_TOKENS_TO_GENERATE", predict.c_str(), 1);

  const std::string cmd = "./build/korith_dynamic " + model_path + " 2>&1";
  std::FILE * pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return result;
  }

  char buf[4096];
  while (std::fgets(buf, sizeof(buf), pipe)) {
    const std::string line(buf);
    if (line.find("[KORITH_RUN_SUMMARY]") != std::string::npos) {
      (void) parse_key_u64(line, "tokens_generated=", result.tokens_generated);
      result.saw_summary = true;
      continue;
    }
    if (line.find("[KORITH_HEALTH]") != std::string::npos) {
      (void) parse_key_double(line, "hit_rate=", result.hit_rate);
      (void) parse_key_double(line, "avg_skip_ratio=", result.avg_skip_ratio);
      (void) parse_key_double(line, "avg_roi_ema=", result.avg_roi_ema);
      (void) parse_key_double(line, "roi_slope=", result.roi_slope);
      (void) parse_key_u64(line, "evictions=", result.evictions);
      (void) parse_key_u64(line, "mf_updates=", result.mf_updates);
      (void) parse_key_u64(line, "replay_disables=", result.replay_disables);
      result.saw_health = true;
      continue;
    }
    if (line.find("[AMF_SUMMARY]") != std::string::npos) {
      (void) parse_key_u64(line, "bytes=", result.bytes);
      result.saw_amf_summary = true;
      continue;
    }
    if (line.find("[MF_APPLY]") != std::string::npos) {
      int mask = 0;
      if (parse_hex_mask(line, "replay_mask=0x", mask)) {
        if (result.replay_mask >= 0 && mask != result.replay_mask) {
          result.replay_mask_flips += 1;
        }
        result.replay_mask = mask;
      }
      continue;
    }
    if (line.find("[AMF_REPLAY_OK]") != std::string::npos) {
      if (result.replay_mask == 1) {
        result.replay_violation = true;
      }
      continue;
    }
  }

  const int rc = pclose(pipe);
  result.ok = (rc == 0) && result.saw_health && result.saw_summary && result.saw_amf_summary;
  return result;
}

bool within_ratio(double a, double b, double max_ratio) {
  const double denom = std::max(1e-9, std::max(a, b));
  return (std::fabs(a - b) / denom) <= max_ratio;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc < 3 || !argv) {
    std::fprintf(stderr, "usage: %s <model.gguf> <prompt.txt>\n", (argc > 0) ? argv[0] : "amf_soak_test");
    return 2;
  }

  const std::string model_path = argv[1];
  const std::string prompt_path = argv[2];
  const std::string prompt = read_file(prompt_path);
  if (prompt.empty()) {
    std::fprintf(stderr, "error: failed to read prompt\n");
    return 3;
  }

  const std::string near_prompt = prompt + "\n\nNear-miss suffix for AMF probe.\n";
  const std::string cold_prompt = "Write a short list of risks in deploying caching systems.\n";

  const std::string amf_dir = "./build/amf_soak_store";
  (void) std::filesystem::remove_all(amf_dir);
  (void) std::filesystem::create_directories(amf_dir);
  (void) setenv("KORITH_ENABLE_AMF", "1", 1);
  (void) setenv("KORITH_AMF_PATH", amf_dir.c_str(), 1);

  const std::uint64_t warm_tokens = 512;
  const std::uint64_t run_tokens = 3072;

  std::vector<RunResult> runs;
  runs.push_back(run_once(model_path, prompt, warm_tokens));
  runs.push_back(run_once(model_path, prompt, warm_tokens));
  runs.push_back(run_once(model_path, prompt, run_tokens));
  runs.push_back(run_once(model_path, near_prompt, run_tokens));
  runs.push_back(run_once(model_path, cold_prompt, run_tokens));

  std::uint64_t total_tokens = 0;
  for (const RunResult & r : runs) {
    if (!r.ok) {
      return 10;
    }
    total_tokens += r.tokens_generated;
    if (r.replay_violation) {
      return 11;
    }
    if (r.replay_mask_flips > 1) {
      return 12;
    }
  }

  if (total_tokens < 10000) {
    return 13;
  }

  const RunResult & rep = runs[2];
  const RunResult & near = runs[3];
  const RunResult & cold = runs[4];

  const double hit_rate_avg = (rep.hit_rate + near.hit_rate + cold.hit_rate) / 3.0;
  if (!std::isfinite(hit_rate_avg) || hit_rate_avg < 0.40) {
    return 20;
  }

  const double max_skip = std::max({rep.avg_skip_ratio, near.avg_skip_ratio, cold.avg_skip_ratio});
  const double min_skip = std::min({rep.avg_skip_ratio, near.avg_skip_ratio, cold.avg_skip_ratio});
  if (!std::isfinite(max_skip) || !std::isfinite(min_skip) || (max_skip - min_skip) > 0.15) {
    return 21;
  }

  if ((rep.hit_rate > 0.0 && (!std::isfinite(rep.avg_roi_ema) || rep.avg_roi_ema < 1.0)) ||
      (near.hit_rate > 0.0 && (!std::isfinite(near.avg_roi_ema) || near.avg_roi_ema < 1.0)) ||
      (cold.hit_rate > 0.0 && (!std::isfinite(cold.avg_roi_ema) || cold.avg_roi_ema < 1.0))) {
    return 22;
  }

  if (!within_ratio(static_cast<double>(rep.bytes),
                    static_cast<double>(near.bytes), 0.50) ||
      !within_ratio(static_cast<double>(near.bytes),
                    static_cast<double>(cold.bytes), 0.50)) {
    return 23;
  }

  return 0;
}
