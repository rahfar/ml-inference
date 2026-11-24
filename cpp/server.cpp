// The C++ server: the same API, models and timing headers as serving/, built
// for speed. Separate from the Python code; bench.run launches it directly.
//
//   make cpp
//   cpp/build/server --model lstm --workers 4 --threads 1
//   cpp/build/server --model catboost --profile     # model-only cost, like `make profile`
//
// POST /predict   body: (n, 150) float32, JSON nested lists or raw bytes -> (n, 30)
// GET  /health
//
// Drogon runs --io-threads event loops, which parse HTTP and decode bodies.
//   --workers N   requests go to a queue; N inference threads each take
//                 everything waiting (up to 64 rows) as one predict() call:
//                 Python's `batched` strategy, minus the GIL
//   --workers 0   predict() runs on the event loop thread that got the request
//
// Replies carry the same Server-Timing (decode / wait / predict / encode, ms)
// and X-Batch-Size headers as serving/handler.py, which the bench aggregates.

#include <ATen/Parallel.h>
#include <torch/script.h>

#include <drogon/drogon.h>
#include <simdjson.h>

#include "c_api.h"  // CatBoost's model evaluation library

#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// Shapes, as in models/data.py.
constexpr size_t HISTORY_STEPS = 30, HISTORY_FEATURES = 5, FUTURE_STEPS = 15;
constexpr size_t INPUT_SIZE = HISTORY_STEPS * HISTORY_FEATURES;  // 150 floats per row
constexpr size_t OUTPUT_SIZE = FUTURE_STEPS * 2;                 // 30 floats per row
constexpr size_t MAX_BATCH = 64;                                 // rows per predict() call

using Clock = std::chrono::steady_clock;
using Callback = std::function<void(const drogon::HttpResponsePtr&)>;

double seconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

// ---------------------------------------------------------------------------
// Pre/post-processing (models/data.py): models see offsets from the last
// observed position, not absolute lat/lon.

void to_relative(const float* x, size_t n, float* out) {
  for (size_t r = 0; r < n; ++r, x += INPUT_SIZE, out += INPUT_SIZE) {
    const float* last = x + (HISTORY_STEPS - 1) * HISTORY_FEATURES;
    const float lat = last[0], lon = last[1];
    std::memcpy(out, x, INPUT_SIZE * sizeof(float));
    for (size_t t = 0; t < HISTORY_STEPS; ++t) {
      out[t * HISTORY_FEATURES] -= lat;
      out[t * HISTORY_FEATURES + 1] -= lon;
    }
  }
}

void add_last_position(const float* x, size_t n, float* y) {  // offsets -> positions
  for (size_t r = 0; r < n; ++r, x += INPUT_SIZE, y += OUTPUT_SIZE) {
    const float* last = x + (HISTORY_STEPS - 1) * HISTORY_FEATURES;
    for (size_t t = 0; t < FUTURE_STEPS; ++t) {
      y[2 * t] += last[0];
      y[2 * t + 1] += last[1];
    }
  }
}

// ---------------------------------------------------------------------------
// Models: predict((n, 150) absolute history) -> (n, 30) absolute positions.
// One instance is shared by all threads; predict() may run concurrently.

struct Model {
  virtual ~Model() = default;
  virtual void predict(const float* x, size_t n, float* y) = 0;
};

struct Noop final : Model {  // the serving overhead alone
  void predict(const float*, size_t n, float* y) override { std::fill_n(y, n * OUTPUT_SIZE, 0.0f); }
};

class CatBoost final : public Model {  // single-threaded per call: the C API has no thread setting
 public:
  explicit CatBoost(const std::string& path) : handle_(ModelCalcerCreate()) {
    if (!LoadFullModelFromFile(handle_, path.c_str())) throw std::runtime_error(GetErrorString());
    if (GetFloatFeaturesCount(handle_) != INPUT_SIZE || GetDimensionsCount(handle_) != OUTPUT_SIZE)
      throw std::runtime_error(path + ": unexpected number of features or outputs");
  }
  ~CatBoost() override { ModelCalcerDelete(handle_); }

  void predict(const float* x, size_t n, float* y) override {
    std::vector<float> rel(n * INPUT_SIZE);
    to_relative(x, n, rel.data());
    std::vector<const float*> rows(n);
    for (size_t r = 0; r < n; ++r) rows[r] = rel.data() + r * INPUT_SIZE;
    std::vector<double> offsets(n * OUTPUT_SIZE);
    if (!CalcModelPredictionFlat(handle_, n, rows.data(), INPUT_SIZE, offsets.data(), offsets.size()))
      throw std::runtime_error(GetErrorString());
    std::copy(offsets.begin(), offsets.end(), y);  // double -> float, like Python's astype
    add_last_position(x, n, y);
  }

 private:
  ModelCalcerHandle* handle_;
};

class Lstm final : public Model {  // the TorchScript export of models/lstm.py (cpp/export_lstm.py)
 public:
  Lstm(const std::string& path, int threads) : threads_(threads) {
    torch::jit::setGraphExecutorOptimize(false);  // the graph is 3 ops; no re-specializing per batch size
    module_ = torch::jit::load(path);
    module_.eval();
  }

  void predict(const float* x, size_t n, float* y) override {
    // torch's thread count is per calling thread (OpenMP), as in models/lstm.py.
    thread_local int current_threads = -1;
    if (current_threads != threads_) at::set_num_threads(current_threads = threads_);

    std::vector<float> rel(n * INPUT_SIZE);
    to_relative(x, n, rel.data());
    c10::InferenceMode guard;
    auto seq = torch::from_blob(rel.data(), {static_cast<int64_t>(n), HISTORY_STEPS, HISTORY_FEATURES});
    at::Tensor offsets = module_.forward({seq}).toTensor().contiguous();
    std::memcpy(y, offsets.data_ptr<float>(), n * OUTPUT_SIZE * sizeof(float));
    add_last_position(x, n, y);
  }

 private:
  int threads_;
  torch::jit::Module module_;
};

// ---------------------------------------------------------------------------
// Codecs (serving/codecs.py): JSON nested lists, or raw little-endian float32.

struct Codec {
  const char* content_type;
  bool (*decode)(std::string_view body, std::vector<float>& out);
  std::string (*encode)(const float* y, size_t n);
};

bool json_decode(std::string_view body, std::vector<float>& out) {
  thread_local simdjson::ondemand::parser parser;
  thread_local std::string buffer;  // simdjson reads up to SIMDJSON_PADDING bytes past the end
  buffer.assign(body);
  buffer.resize(body.size() + simdjson::SIMDJSON_PADDING);
  try {
    auto doc = parser.iterate(buffer.data(), body.size(), buffer.size());
    for (auto item : doc.get_array()) {  // [[150 numbers], ...] or a flat [150 numbers]
      simdjson::ondemand::value value = item.value();
      if (value.type() == simdjson::ondemand::json_type::array) {
        for (auto number : value.get_array()) out.push_back(static_cast<float>(double(number.get_double())));
      } else {
        out.push_back(static_cast<float>(double(value.get_double())));
      }
    }
    return doc.at_end();
  } catch (const simdjson::simdjson_error&) {
    return false;
  }
}

std::string json_encode(const float* y, size_t n) {
  std::string out;
  out.reserve(n * (OUTPUT_SIZE * 12 + 2) + 2);
  char number[32];
  out += '[';
  for (size_t r = 0; r < n; ++r) {
    out += r ? ",[" : "[";
    for (size_t c = 0; c < OUTPUT_SIZE; ++c) {
      if (c) out += ',';
      // Shortest text that reads back as the same float32 (Python writes float64 repr).
      auto [end, _] = std::to_chars(number, number + sizeof number, y[r * OUTPUT_SIZE + c]);
      out.append(number, end);
    }
    out += ']';
  }
  out += ']';
  return out;
}

bool raw_decode(std::string_view body, std::vector<float>& out) {
  if (body.size() % sizeof(float)) return false;
  out.resize(body.size() / sizeof(float));
  std::memcpy(out.data(), body.data(), body.size());
  return true;
}

std::string raw_encode(const float* y, size_t n) {
  return {reinterpret_cast<const char*>(y), n * OUTPUT_SIZE * sizeof(float)};
}

const Codec* find_codec(const std::string& name) {
  static const Codec json{"application/json", json_decode, json_encode};
  static const Codec raw{"application/octet-stream", raw_decode, raw_encode};
  return name == "json" ? &json : name == "raw" ? &raw : nullptr;
}

// ---------------------------------------------------------------------------
// Request handling: decode on the event loop, predict inline or via the queue.

drogon::HttpResponsePtr text_reply(drogon::HttpStatusCode status, const std::string& body) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(status);
  resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
  resp->setBody(body);
  return resp;
}

struct Request {  // one POST /predict on its way through the server
  std::vector<float> x;
  size_t rows = 0;
  Clock::time_point received, decoded;
  Callback reply;
};

class Server {
 public:
  Server(Model& model, const Codec& codec, size_t workers) : model_(model), codec_(codec) {
    for (size_t i = 0; i < workers; ++i) workers_.emplace_back([this] { work(); });
  }

  ~Server() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  void handle(const drogon::HttpRequestPtr& req, Callback&& reply) {
    Request r{.received = Clock::now(), .reply = std::move(reply)};
    if (!codec_.decode(req->body(), r.x) || r.x.empty() || r.x.size() % INPUT_SIZE)
      return r.reply(text_reply(drogon::k400BadRequest, "cannot decode body as (n, 150) float32"));
    r.rows = r.x.size() / INPUT_SIZE;
    r.decoded = Clock::now();

    if (workers_.empty()) {  // --workers 0: predict right here, on the event loop
      std::vector<Request> batch;
      batch.push_back(std::move(r));
      predict_and_reply(batch);
      return;
    }
    {
      std::lock_guard lock(mutex_);
      queue_.push_back(std::move(r));
    }
    ready_.notify_one();
  }

 private:
  void work() {  // one inference thread
    while (true) {
      std::vector<Request> batch;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (stopping_) return;
        size_t rows = 0;
        do {  // everything waiting, up to MAX_BATCH rows (as serving/strategies.py)
          rows += queue_.front().rows;
          batch.push_back(std::move(queue_.front()));
          queue_.pop_front();
        } while (!queue_.empty() && rows < MAX_BATCH);
      }
      predict_and_reply(batch);
    }
  }

  void predict_and_reply(std::vector<Request>& batch) {
    size_t rows = 0;
    for (const auto& r : batch) rows += r.rows;
    std::vector<float> joined;
    const float* x = batch[0].x.data();
    if (batch.size() > 1) {
      joined.reserve(rows * INPUT_SIZE);
      for (const auto& r : batch) joined.insert(joined.end(), r.x.begin(), r.x.end());
      x = joined.data();
    }

    std::vector<float> y(rows * OUTPUT_SIZE);
    const auto start = Clock::now();
    try {
      model_.predict(x, rows, y.data());
    } catch (const std::exception& e) {
      for (auto& r : batch) r.reply(text_reply(drogon::k500InternalServerError, e.what()));
      return;
    }
    const auto done = Clock::now();
    const double predict_s = seconds(start, done);

    const float* slice = y.data();
    for (auto& r : batch) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setContentTypeString(codec_.content_type);
      resp->setBody(codec_.encode(slice, r.rows));
      slice += r.rows * OUTPUT_SIZE;
      const auto encoded = Clock::now();
      char timing[160];
      std::snprintf(timing, sizeof timing, "decode;dur=%.3f, wait;dur=%.3f, predict;dur=%.3f, encode;dur=%.3f",
                    seconds(r.received, r.decoded) * 1e3, (seconds(r.decoded, done) - predict_s) * 1e3,
                    predict_s * 1e3, seconds(done, encoded) * 1e3);
      resp->addHeader("Server-Timing", timing);
      resp->addHeader("X-Batch-Size", std::to_string(rows));
      r.reply(resp);  // Drogon sends it from the connection's own event loop
    }
  }

  Model& model_;
  const Codec& codec_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Request> queue_;
  bool stopping_ = false;
  std::vector<std::thread> workers_;
};

// ---------------------------------------------------------------------------
// Command line, model-only profiling, main.

struct Options {
  std::string model = "lstm", codec = "json", host = "127.0.0.1", weights = "models/weights";
  int threads = 1;           // torch threads per predict() call
  size_t workers = 1;        // inference threads; 0 = predict on the event loop
  size_t io_threads = 2;     // Drogon event loops
  uint16_t port = 8000;
  bool profile = false;
};

[[noreturn]] void usage(const std::string& error) {
  std::cerr << "error: " << error << "\n"
            << "usage: server [--model lstm|catboost|noop] [--codec json|raw] [--threads N]\n"
            << "              [--workers N] [--io-threads N] [--host H] [--port P]\n"
            << "              [--weights DIR] [--profile]\n";
  std::exit(2);
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--profile") {
      o.profile = true;
      continue;
    }
    const auto eq = arg.find('=');  // --key=value or --key value
    const std::string key = arg.substr(0, eq);
    std::string value;
    if (eq != std::string::npos) value = arg.substr(eq + 1);
    else if (i + 1 < argc) value = argv[++i];
    else usage("missing value for " + key);
    try {
      if (key == "--model") o.model = value;
      else if (key == "--codec") o.codec = value;
      else if (key == "--threads") o.threads = std::stoi(value);
      else if (key == "--workers") o.workers = std::stoul(value);
      else if (key == "--io-threads") o.io_threads = std::stoul(value);
      else if (key == "--host") o.host = value;
      else if (key == "--port") o.port = static_cast<uint16_t>(std::stoul(value));
      else if (key == "--weights") o.weights = value;
      else usage("unknown option " + key);
    } catch (const std::logic_error&) {
      usage("bad value for " + key + ": " + value);
    }
  }
  if (!find_codec(o.codec)) usage("unknown codec " + o.codec);
  if (o.threads < 1 || o.io_threads < 1) usage("--threads and --io-threads must be >= 1");
  return o;
}

std::unique_ptr<Model> load_model(const Options& o) {
  if (o.model == "noop") return std::make_unique<Noop>();
  if (o.model == "catboost") return std::make_unique<CatBoost>(o.weights + "/catboost.cbm");
  if (o.model == "lstm") return std::make_unique<Lstm>(o.weights + "/lstm.ts", o.threads);
  usage("unknown model " + o.model);
}

std::vector<float> straight_tracks(size_t n) {  // plausible input for profiling
  std::vector<float> x(n * INPUT_SIZE);
  for (size_t r = 0; r < n; ++r) {
    const float lat0 = 54.0f + r % 10, lon0 = r % 20, speed = 5.0f + r % 15, course = 0.1f * r;
    for (size_t t = 0; t < HISTORY_STEPS; ++t) {
      float* step = &x[r * INPUT_SIZE + t * HISTORY_FEATURES];
      const float hours = t * 10.0f / 3600.0f;
      step[0] = lat0 + speed * hours * std::cos(course);
      step[1] = lon0 + speed * hours * std::sin(course) / std::cos(lat0 * 3.14159265f / 180.0f);
      step[2] = speed;
      step[3] = std::sin(course);
      step[4] = std::cos(course);
    }
  }
  return x;
}

int profile(Model& model, const Options& o) {  // the C++ twin of models/profile.py
  const auto x = straight_tracks(256);
  std::printf("%s, %d threads (C++)\n batch    ms/call  us/sample\n", o.model.c_str(), o.threads);
  for (size_t batch : {1, 4, 16, 64, 256}) {
    std::vector<float> y(batch * OUTPUT_SIZE);
    for (int i = 0; i < 3; ++i) model.predict(x.data(), batch, y.data());  // warmup
    size_t calls = 0;
    double elapsed = 0;
    for (const auto start = Clock::now(); (elapsed = seconds(start, Clock::now())) < 1.0; ++calls)
      model.predict(x.data(), batch, y.data());
    const double per_call = elapsed / calls;
    std::printf("%6zu %10.3f %10.1f\n", batch, per_call * 1e3, per_call * 1e6 / batch);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse_args(argc, argv);
  std::unique_ptr<Model> model;
  try {
    model = load_model(o);
  } catch (const std::exception& e) {
    std::cerr << "cannot load " << o.model << ": " << e.what() << "\n";
    return 1;
  }
  if (o.profile) return profile(*model, o);

  const Codec& codec = *find_codec(o.codec);
  Server server(*model, codec, o.workers);
  drogon::app()
      .registerHandler(
          "/health",
          [](const drogon::HttpRequestPtr&, Callback&& reply) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            resp->setBody(R"({"status": "ok"})");
            reply(resp);
          },
          {drogon::Get})
      .registerHandler(
          "/predict",
          [&server](const drogon::HttpRequestPtr& req, Callback&& reply) { server.handle(req, std::move(reply)); },
          {drogon::Post})
      .addListener(o.host, o.port)
      .setThreadNum(o.io_threads)
      .setLogLevel(trantor::Logger::kWarn)
      .enableServerHeader(false)
      .enableDateHeader(false)
      .setIdleConnectionTimeout(0)           // keep benchmark connections open
      .setKeepaliveRequestsNumber(0)         // no limit on requests per connection
      .setClientMaxBodySize(256 << 20)       // the codec experiment sends MBs of JSON
      .setClientMaxMemoryBodySize(256 << 20) // ... keep them in memory, not temp files
      .run();                                // until SIGTERM / SIGINT
  return 0;
}
