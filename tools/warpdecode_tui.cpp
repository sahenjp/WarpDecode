#include "gguf.hpp"
#include "gpu_arch.hpp"
#include "model_profile.hpp"
#include "qwen38_tuning.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(trim(item));
    return out;
}

std::string json_escape(const std::string& in) {
    std::ostringstream o;
    for (unsigned char c : in) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
                else o << c;
        }
    }
    return o.str();
}

std::string human_size(uintmax_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream s;
    s << std::fixed << std::setprecision(gib >= 10 ? 1 : 2) << gib << " GiB";
    return s.str();
}

struct GpuInfo {
    int index = -1;
    std::string name;
    int memory_mib = 0;
    int sm = 0;
    bool selected = false;
};

struct ModelInfo {
    fs::path path;
    uintmax_t bytes = 0;
    size_t shard_count = 1;
    std::string arch = "?";
    std::string quant = "?";
    wd::ModelProfile profile;
    bool supported = false;
    std::string note;
    bool queued = false;
};

struct Config {
    std::vector<fs::path> model_dirs;
    int ctx = 4096;
    int generate = 128;
    int start_token = 1;
    bool autotune = true;
};

fs::path config_path() {
    const char* home = std::getenv("HOME");
    fs::path base = home ? fs::path(home) : fs::current_path();
    return base / ".config" / "warpdecode" / "tui.conf";
}

Config default_config() {
    Config c;
    const char* env = std::getenv("WARPDECODE_MODEL_DIRS");
    if (env && *env) {
        for (const auto& p : split(env, ':')) if (!p.empty()) c.model_dirs.emplace_back(p);
    }
    const char* home = std::getenv("HOME");
    c.model_dirs.emplace_back(fs::current_path() / "models");
    if (home) c.model_dirs.emplace_back(fs::path(home) / "models");
    c.model_dirs.emplace_back("/models");
    return c;
}

Config load_config() {
    Config c = default_config();
    const fs::path p = config_path();
    std::ifstream f(p);
    if (!f) return c;
    c.model_dirs.clear();
    std::string line;
    while (std::getline(f, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string k = trim(line.substr(0, pos));
        const std::string v = trim(line.substr(pos + 1));
        try {
            if (k == "model_dir") c.model_dirs.emplace_back(v);
            else if (k == "ctx") c.ctx = std::stoi(v);
            else if (k == "generate") c.generate = std::stoi(v);
            else if (k == "start_token") c.start_token = std::stoi(v);
            else if (k == "autotune") c.autotune = (v != "0");
        } catch (...) {}
    }
    if (c.model_dirs.empty()) c = default_config();
    return c;
}

void save_config(const Config& c) {
    const fs::path p = config_path();
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::trunc);
    for (const auto& d : c.model_dirs) f << "model_dir=" << d.string() << "\n";
    f << "ctx=" << c.ctx << "\n";
    f << "generate=" << c.generate << "\n";
    f << "start_token=" << c.start_token << "\n";
    f << "autotune=" << (c.autotune ? 1 : 0) << "\n";
}

std::vector<GpuInfo> parse_gpu_lines(std::istream& in) {
    std::vector<GpuInfo> out;
    std::string line;
    while (std::getline(in, line)) {
        auto f = split(line, ',');
        if (f.size() < 4) continue;
        try {
            GpuInfo g;
            g.index = std::stoi(f[0]);
            g.name = f[1];
            g.memory_mib = std::stoi(f[2]);
            const auto cc = split(f[3], '.');
            if (cc.size() != 2) continue;
            g.sm = std::stoi(cc[0]) * 10 + std::stoi(cc[1]);
            out.push_back(std::move(g));
        } catch (...) {}
    }
    return out;
}

std::vector<GpuInfo> discover_gpus(const fs::path& fixture = {}) {
    if (!fixture.empty()) {
        std::ifstream f(fixture);
        return parse_gpu_lines(f);
    }
    FILE* p = ::popen("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return {};
    std::stringstream ss;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), p)) ss << buf;
    ::pclose(p);
    return parse_gpu_lines(ss);
}

std::string quant_name(wd::GgmlType t) {
    switch (t) {
        case wd::GgmlType::Q4_0: return "Q4_0";
        case wd::GgmlType::F16: return "F16";
        case wd::GgmlType::F32: return "F32";
        default: return "other";
    }
}

ModelInfo inspect_model(const fs::path& p) {
    ModelInfo m;
    m.path = p;
    std::error_code ec;
    m.bytes = fs::file_size(p, ec);
    try {
        wd::GgufFile g(p.string(), false);
        m.profile = wd::profile_model(g, p);
        m.arch = m.profile.arch;
        m.quant = m.profile.quant_hint;

        // Dense legacy models can derive the exact type from token_embd.  qwen4exp
        // split GGUFs may have a metadata-only first shard, so never require a tensor
        // just to show the model in the TUI.
        if (!m.profile.qwen38_flash_next) {
            if (const auto* t = g.optional_tensor("token_embd.weight")) m.quant = quant_name(t->type);
        }
        m.supported = m.profile.runnable && m.quant == "Q4_0";
        if (m.profile.qwen38_flash_next) m.note = "qwen3.8専用planner: runtime配線中";
        else if (!m.supported) m.note = "現エンジン対象外";
    } catch (const std::exception& e) {
        m.note = e.what();
    }
    return m;
}

struct GgufGroup {
    fs::path first;
    uintmax_t bytes = 0;
    size_t shards = 0;
};

std::string gguf_group_key(const fs::path& p, int* shard_index = nullptr) {
    static const std::regex split_re(R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$)", std::regex::icase);
    std::smatch m;
    const std::string name = p.filename().string();
    if (!std::regex_match(name, m, split_re)) {
        if (shard_index) *shard_index = 1;
        return p.string();
    }
    if (shard_index) *shard_index = std::stoi(m[2].str());
    return (p.parent_path() / (m[1].str() + "-of-" + m[3].str())).string();
}

std::vector<ModelInfo> discover_models(const std::vector<fs::path>& dirs, size_t limit = 500) {
    std::map<std::string, GgufGroup> groups;
    std::set<fs::path> seen;
    for (const auto& dir : dirs) {
        std::error_code ec;
        if (!fs::exists(dir, ec)) continue;
        fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
        for (; it != end && groups.size() < limit; it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (!it->is_regular_file(ec)) continue;
            auto p = it->path();
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return char(std::tolower(c)); });
            if (ext != ".gguf") continue;
            auto canon = fs::weakly_canonical(p, ec);
            if (ec) canon = p;
            if (!seen.insert(canon).second) continue;

            int shard_index = 1;
            const std::string key = gguf_group_key(canon, &shard_index);
            auto& gr = groups[key];
            gr.bytes += fs::file_size(canon, ec);
            ++gr.shards;
            if (gr.first.empty() || shard_index == 1) gr.first = canon;
        }
        if (groups.size() >= limit) break;
    }

    std::vector<ModelInfo> out;
    out.reserve(groups.size());
    for (const auto& [_, gr] : groups) {
        if (gr.first.empty()) continue;
        auto m = inspect_model(gr.first);
        m.bytes = gr.bytes;
        m.shard_count = gr.shards;
        out.push_back(std::move(m));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ return a.path.filename().string() < b.path.filename().string(); });
    return out;
}

fs::path project_root(const char* argv0) {
    std::error_code ec;
    fs::path cwd = fs::current_path();
    if (fs::exists(cwd / "bench" / "build.sh", ec)) return cwd;
    fs::path exe = fs::weakly_canonical(argv0, ec);
    if (!ec) {
        fs::path p = exe.parent_path();
        if (fs::exists(p / "bench" / "build.sh", ec)) return p;
        if (fs::exists(p.parent_path() / "bench" / "build.sh", ec)) return p.parent_path();
    }
    return cwd;
}

fs::path engine_path(const fs::path& root, int sm) {
    return root / ("build-sm" + std::to_string(sm)) / "warpdecode";
}

class RawTerminal {
public:
    void enable() {
        if (active_ || !::isatty(STDIN_FILENO)) return;
        if (::tcgetattr(STDIN_FILENO, &old_) != 0) return;
        termios t = old_;
        t.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        if (::tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) == 0) active_ = true;
    }
    void disable() {
        if (active_) { ::tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_); active_ = false; }
    }
    ~RawTerminal() { disable(); }
private:
    termios old_{};
    bool active_ = false;
};

enum class Key { Up, Down, Left, Right, Tab, Enter, Space, Other, Quit };

std::pair<Key,char> read_key() {
    char c = 0;
    if (::read(STDIN_FILENO, &c, 1) != 1) return {Key::Quit, 0};
    if (c == '\033') {
        char seq[2]{};
        if (::read(STDIN_FILENO, &seq[0], 1) != 1 || ::read(STDIN_FILENO, &seq[1], 1) != 1) return {Key::Other, 0};
        if (seq[0] == '[') {
            if (seq[1] == 'A') return {Key::Up, 0};
            if (seq[1] == 'B') return {Key::Down, 0};
            if (seq[1] == 'C') return {Key::Right, 0};
            if (seq[1] == 'D') return {Key::Left, 0};
        }
    }
    if (c == '\t') return {Key::Tab, c};
    if (c == '\r' || c == '\n') return {Key::Enter, c};
    if (c == ' ') return {Key::Space, c};
    if (c == 'q' || c == 'Q') return {Key::Quit, c};
    return {Key::Other, c};
}

int run_child(const std::vector<std::string>& args) {
    if (args.empty()) return 127;
    pid_t pid = ::fork();
    if (pid < 0) return 127;
    if (pid == 0) {
        std::vector<char*> av;
        av.reserve(args.size() + 1);
        for (const auto& s : args) av.push_back(const_cast<char*>(s.c_str()));
        av.push_back(nullptr);
        ::execv(av[0], av.data());
        std::perror("execv");
        _exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

std::vector<std::string> engine_args(const fs::path& engine, const ModelInfo& m, const GpuInfo& g, const Config& c) {
    std::vector<std::string> a = {
        engine.string(), "--model", m.path.string(), "--ctx", std::to_string(c.ctx),
        "--generate", std::to_string(c.generate), "--device", std::to_string(g.index),
        "--start-token", std::to_string(c.start_token)
    };
    if (!c.autotune) a.push_back("--no-autotune");
    return a;
}

std::string prompt_line(RawTerminal& term, const std::string& label) {
    term.disable();
    std::cout << "\033[2J\033[H" << label << std::flush;
    std::string s;
    std::getline(std::cin, s);
    term.enable();
    return trim(s);
}

class Tui {
public:
    Tui(fs::path root, Config cfg, std::vector<GpuInfo> gpus, std::vector<ModelInfo> models)
        : root_(std::move(root)), cfg_(std::move(cfg)), gpus_(std::move(gpus)), models_(std::move(models)) {
        if (!gpus_.empty()) gpus_[0].selected = true;
    }

    int run() {
        if (!::isatty(STDIN_FILENO) || !::isatty(STDOUT_FILENO)) {
            std::cerr << "TUIには対話端末が必要です。状態確認は --dump-state を使ってください。\n";
            return 2;
        }
        term_.enable();
        bool quit = false;
        while (!quit) {
            draw();
            auto [key, ch] = read_key();
            switch (key) {
                case Key::Quit: quit = true; break;
                case Key::Tab: focus_gpu_ = !focus_gpu_; break;
                case Key::Up: move(-1); break;
                case Key::Down: move(1); break;
                case Key::Enter: set_primary(); break;
                case Key::Space: toggle_selected(); break;
                case Key::Other: handle_hotkey(ch, quit); break;
                default: break;
            }
        }
        term_.disable();
        std::cout << "\033[2J\033[H";
        return 0;
    }

private:
    void rescan() {
        models_ = discover_models(cfg_.model_dirs);
        model_cursor_ = std::min(model_cursor_, models_.empty() ? size_t(0) : models_.size() - 1);
        gpus_ = discover_gpus();
        if (!gpus_.empty()) {
            gpu_cursor_ = std::min(gpu_cursor_, gpus_.size() - 1);
            gpus_[gpu_cursor_].selected = true;
        }
        status_ = "再スキャンしました";
    }

    void move(int delta) {
        if (focus_gpu_) {
            if (gpus_.empty()) return;
            gpu_cursor_ = (gpu_cursor_ + gpus_.size() + delta) % gpus_.size();
        } else {
            if (models_.empty()) return;
            model_cursor_ = (model_cursor_ + models_.size() + delta) % models_.size();
        }
    }

    void set_primary() {
        if (focus_gpu_) primary_gpu_ = gpu_cursor_;
        else primary_model_ = model_cursor_;
    }

    void toggle_selected() {
        if (focus_gpu_) {
            if (!gpus_.empty()) gpus_[gpu_cursor_].selected = !gpus_[gpu_cursor_].selected;
        } else {
            if (!models_.empty()) models_[model_cursor_].queued = !models_[model_cursor_].queued;
        }
    }

    void handle_hotkey(char ch, bool&) {
        switch (ch) {
            case 'l': case 'L': launch_single(); break;
            case 'm': case 'M': launch_multi(); break;
            case 'b': case 'B': build_selected(); break;
            case 'r': case 'R': rescan(); break;
            case 'a': case 'A': add_model_dir(); break;
            case 'c': case 'C': edit_number("コンテキスト長: ", cfg_.ctx, 16); break;
            case 'n': case 'N': edit_number("生成トークン数: ", cfg_.generate, 1); break;
            case 't': case 'T': cfg_.autotune = !cfg_.autotune; save_config(cfg_); status_ = cfg_.autotune ? "autotune: ON" : "autotune: OFF"; break;
            case 'p': case 'P': show_plan(); break;
            default: break;
        }
    }

    void show_plan() {
        if (models_.empty()) { status_ = "モデルがありません"; return; }
        const auto& m = models_[primary_model_ < models_.size() ? primary_model_ : 0];
        const GpuInfo* gpu = gpus_.empty() ? nullptr : &gpus_[primary_gpu_ < gpus_.size() ? primary_gpu_ : 0];
        term_.disable();
        std::cout << "\033[2J\033[HWarpDecode model plan\n\n";
        std::cout << "model: " << m.path.filename().string() << "\n";
        std::cout << "family: " << m.profile.family << "  arch=" << m.arch << "  quant=" << m.quant << "\n";
        std::cout << "size: " << human_size(m.bytes) << "  shards=" << m.shard_count << "\n";
        if (gpu) {
            const auto gp = wd::kernel_plan_for_sm(gpu->sm);
            std::cout << "gpu: " << gpu->name << " sm_" << gpu->sm << "\n";
            std::cout << "low-bit decode: " << gp.decode << "\n";
        }
        if (m.profile.qwen38_flash_next) {
            std::cout << "layout: " << m.profile.layers << " layers = " << m.profile.gdn_layers << " GDN + " << m.profile.qsa_layers << " QSA\n";
            std::cout << "moe: " << m.profile.experts << " experts, " << m.profile.routed_experts << " routed + " << m.profile.shared_experts << " shared/token\n";
            std::cout << "QSA: " << m.profile.qsa_path << "\n";
            std::cout << "PLE: " << m.profile.ple_path << "\n";
            std::cout << "MoE: " << m.profile.moe_path << "\n";
            std::cout << "MTP: " << m.profile.mtp_path << "\n";
            if (gpu) {
                const auto tune = wd::qwen38_tuning_for_gpu(gpu->sm, gpu->memory_mib);
                std::cout << "tune: QSA=" << tune.qsa_block_budget << " blocks/" << tune.qsa_token_budget << " tokens"
                          << " PLE-prefetch=" << tune.ple_prefetch_rows
                          << " rows PLE-cache=" << tune.ple_hot_cache_mib << " MiB"
                          << " expert-cache=" << tune.expert_cache_mib << " MiB"
                          << " prefetch-depth=" << tune.expert_prefetch_depth << "\n";
                std::cout << "MTP auto: draft={2,3,4}, reward=accepted_target_tokens/gpu_us\n";
            }
            std::cout << "runtime: planner ready / qwen4exp forward kernels not wired yet\n";
        } else {
            std::cout << "decode: " << m.profile.decode_path << "\n";
            std::cout << "runtime: " << (m.supported ? "ready" : "unsupported") << "\n";
        }
        std::cout << "\nEnterで戻ります..." << std::flush;
        std::string dummy; std::getline(std::cin, dummy);
        term_.enable();
        status_ = "planを表示しました";
    }

    void edit_number(const std::string& label, int& value, int minv) {
        const std::string s = prompt_line(term_, label);
        try {
            int x = std::stoi(s);
            if (x < minv) throw std::runtime_error("range");
            value = x;
            save_config(cfg_);
            status_ = "設定を保存しました";
        } catch (...) { status_ = "数値が不正です"; }
    }

    void add_model_dir() {
        const std::string s = prompt_line(term_, "追加するモデルディレクトリ: ");
        if (s.empty()) { status_ = "キャンセル"; return; }
        fs::path p = fs::path(s);
        std::error_code ec;
        p = fs::weakly_canonical(p, ec);
        if (ec || !fs::is_directory(p, ec)) { status_ = "ディレクトリが見つかりません"; return; }
        cfg_.model_dirs.push_back(p);
        save_config(cfg_);
        models_ = discover_models(cfg_.model_dirs);
        status_ = "モデルディレクトリを追加しました";
    }

    bool validate_single() {
        if (models_.empty()) { status_ = "モデルがありません"; return false; }
        if (gpus_.empty()) { status_ = "NVIDIA GPUが見つかりません"; return false; }
        if (primary_model_ >= models_.size()) primary_model_ = 0;
        if (primary_gpu_ >= gpus_.size()) primary_gpu_ = 0;
        if (!models_[primary_model_].supported) { status_ = "選択モデルは現バックエンド未対応です"; return false; }
        if (gpus_[primary_gpu_].sm < 50) { status_ = "選択GPUはSM50未満です"; return false; }
        return true;
    }

    void launch_single() {
        if (!validate_single()) return;
        const auto& g = gpus_[primary_gpu_];
        const auto& m = models_[primary_model_];
        const fs::path engine = engine_path(root_, g.sm);
        if (!fs::exists(engine)) { status_ = "エンジン未ビルド: Bでsm_" + std::to_string(g.sm) + "をビルド"; return; }
        term_.disable();
        std::cout << "\033[2J\033[H起動: " << m.path.filename().string() << " / " << g.name << "\n\n";
        const int rc = run_child(engine_args(engine, m, g, cfg_));
        std::cout << "\n終了コード: " << rc << "\nEnterでTUIへ戻ります..." << std::flush;
        std::string dummy; std::getline(std::cin, dummy);
        term_.enable();
        status_ = "単体実行が終了しました";
    }

    void launch_multi() {
        std::vector<size_t> mi, gi;
        for (size_t i = 0; i < models_.size(); ++i) if (models_[i].queued) mi.push_back(i);
        for (size_t i = 0; i < gpus_.size(); ++i) if (gpus_[i].selected) gi.push_back(i);
        if (mi.empty() && !models_.empty()) mi.push_back(primary_model_);
        if (mi.empty() || gi.empty()) { status_ = "モデルとGPUをSpaceで選択してください"; return; }
        if (mi.size() > gi.size()) { status_ = "独立同時起動はモデル数以上のGPUが必要です"; return; }
        for (size_t i : mi) if (!models_[i].supported) { status_ = "キューに未対応モデルがあります"; return; }
        for (size_t k = 0; k < mi.size(); ++k) {
            if (!fs::exists(engine_path(root_, gpus_[gi[k]].sm))) { status_ = "未ビルドGPUがあります。Bで先にビルドしてください"; return; }
        }
        term_.disable();
        std::cout << "\033[2J\033[H" << mi.size() << "個の独立モデルを同時起動します\n";
        std::vector<pid_t> children;
        for (size_t k = 0; k < mi.size(); ++k) {
            const auto args = engine_args(engine_path(root_, gpus_[gi[k]].sm), models_[mi[k]], gpus_[gi[k]], cfg_);
            pid_t pid = ::fork();
            if (pid == 0) {
                std::vector<char*> av;
                for (const auto& s : args) av.push_back(const_cast<char*>(s.c_str()));
                av.push_back(nullptr);
                ::execv(av[0], av.data());
                _exit(127);
            }
            if (pid > 0) children.push_back(pid);
        }
        for (pid_t p : children) { int st = 0; while (::waitpid(p, &st, 0) < 0 && errno == EINTR) {} }
        std::cout << "\n全プロセス終了。EnterでTUIへ戻ります..." << std::flush;
        std::string dummy; std::getline(std::cin, dummy);
        term_.enable();
        status_ = "複数モデル実行が終了しました";
    }

    void build_selected() {
        std::set<int> sms;
        for (const auto& g : gpus_) if (g.selected) sms.insert(g.sm);
        if (sms.empty() && !gpus_.empty()) sms.insert(gpus_[primary_gpu_].sm);
        if (sms.empty()) { status_ = "GPUがありません"; return; }
        term_.disable();
        std::cout << "\033[2J\033[H";
        int worst = 0;
        for (int sm : sms) {
            std::cout << "\n=== sm_" << sm << " をビルド ===\n";
            const fs::path script = root_ / "bench" / "build.sh";
            worst = std::max(worst, run_child({script.string(), "sm" + std::to_string(sm)}));
        }
        std::cout << "\n終了コード: " << worst << "\nEnterでTUIへ戻ります..." << std::flush;
        std::string dummy; std::getline(std::cin, dummy);
        term_.enable();
        status_ = worst == 0 ? "ビルド完了" : "ビルドに失敗しました";
    }

    void draw() const {
        std::cout << "\033[2J\033[H";
        std::cout << "warpdecode control  |  batch=1 CUDA inference\n";
        std::cout << "Tab: モデル/GPU  ↑↓: 移動  Enter: 主選択  Space: 複数選択\n";
        std::cout << "L: 起動  M: 独立複数起動  B: ビルド  P: model plan  A: モデル場所追加  R: 再読込  C/N: ctx/token  T: autotune  Q: 終了\n";
        std::cout << "ctx=" << cfg_.ctx << "  generate=" << cfg_.generate << "  autotune=" << (cfg_.autotune ? "on" : "off") << "\n\n";

        std::cout << (focus_gpu_ ? "   " : "▶ ") << "Models (" << models_.size() << ")\n";
        if (models_.empty()) std::cout << "   GGUFが見つかりません。Aでディレクトリを追加\n";
        const size_t mbeg = models_.empty() ? 0 : (model_cursor_ > 5 ? model_cursor_ - 5 : 0);
        const size_t mend = std::min(models_.size(), mbeg + 11);
        for (size_t i = mbeg; i < mend; ++i) {
            const auto& m = models_[i];
            std::cout << ((!focus_gpu_ && i == model_cursor_) ? " > " : "   ")
                      << (i == primary_model_ ? "*" : " ") << (m.queued ? "[x] " : "[ ] ")
                      << m.path.filename().string() << "  " << human_size(m.bytes)
                      << (m.shard_count > 1 ? "  x" + std::to_string(m.shard_count) + " shards" : "")
                      << "  " << m.arch << "/" << m.quant
                      << (m.profile.qwen38_flash_next ? "  [Q38-plan]" : (m.supported ? "" : "  !")) << "\n";
        }

        std::cout << "\n" << (focus_gpu_ ? "▶ " : "   ") << "GPUs (" << gpus_.size() << ")\n";
        if (gpus_.empty()) std::cout << "   nvidia-smiからGPUを取得できません\n";
        for (size_t i = 0; i < gpus_.size(); ++i) {
            const auto& g = gpus_[i];
            const auto plan = wd::kernel_plan_for_sm(g.sm);
            std::cout << ((focus_gpu_ && i == gpu_cursor_) ? " > " : "   ")
                      << (i == primary_gpu_ ? "*" : " ") << (g.selected ? "[x] " : "[ ] ")
                      << "GPU" << g.index << " " << g.name << "  " << (g.memory_mib / 1024.0) << " GiB  sm_" << g.sm
                      << "  target=" << plan.decode << "\n";
        }

        std::cout << "\nstatus: " << (status_.empty() ? "準備完了" : status_) << "\n";
        std::cout << "* = 単体起動対象 / [x] = 複数起動・ビルド対象\n" << std::flush;
    }

    fs::path root_;
    Config cfg_;
    std::vector<GpuInfo> gpus_;
    std::vector<ModelInfo> models_;
    RawTerminal term_;
    bool focus_gpu_ = false;
    size_t model_cursor_ = 0, gpu_cursor_ = 0;
    size_t primary_model_ = 0, primary_gpu_ = 0;
    std::string status_;
};

void dump_state(const fs::path& fixture, const std::vector<fs::path>& override_dirs) {
    Config c = load_config();
    if (!override_dirs.empty()) c.model_dirs = override_dirs;
    const auto gpus = discover_gpus(fixture);
    const auto models = discover_models(c.model_dirs);
    std::cout << "{\n  \"gpus\": [";
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (i) std::cout << ',';
        const auto p = wd::kernel_plan_for_sm(gpus[i].sm);
        std::cout << "\n    {\"index\":" << gpus[i].index << ",\"name\":\"" << json_escape(gpus[i].name)
                  << "\",\"memory_mib\":" << gpus[i].memory_mib << ",\"sm\":" << gpus[i].sm
                  << ",\"family\":\"" << p.family << "\",\"target_decode\":\"" << p.decode << "\"}";
    }
    std::cout << "\n  ],\n  \"models\": [";
    for (size_t i = 0; i < models.size(); ++i) {
        if (i) std::cout << ',';
        const auto& m = models[i];
        std::cout << "\n    {\"path\":\"" << json_escape(m.path.string()) << "\",\"bytes\":" << m.bytes
                  << ",\"shards\":" << m.shard_count
                  << ",\"arch\":\"" << json_escape(m.arch) << "\",\"quant\":\"" << m.quant
                  << "\",\"family\":\"" << json_escape(m.profile.family)
                  << "\",\"qwen38_flash_next\":" << (m.profile.qwen38_flash_next ? "true" : "false")
                  << ",\"supported\":" << (m.supported ? "true" : "false") << "}";
    }
    std::cout << "\n  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path fixture;
        std::vector<fs::path> dirs;
        bool dump = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--dump-state") dump = true;
            else if (a == "--gpu-fixture" && i + 1 < argc) fixture = argv[++i];
            else if (a == "--model-dir" && i + 1 < argc) dirs.emplace_back(argv[++i]);
            else if (a == "-h" || a == "--help") {
                std::cout << "warpdecode-tui [--dump-state] [--gpu-fixture file] [--model-dir dir]\n";
                return 0;
            } else throw std::runtime_error("不明な引数: " + a);
        }
        if (dump) { dump_state(fixture, dirs); return 0; }
        Config cfg = load_config();
        if (!dirs.empty()) cfg.model_dirs = dirs;
        return Tui(project_root(argv[0]), cfg, discover_gpus(), discover_models(cfg.model_dirs)).run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
