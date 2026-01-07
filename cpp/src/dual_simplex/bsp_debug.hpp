/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Enable BSP debug macros. The actual logging is controlled at runtime via
// environment variables (CUOPT_BSP_DEBUG_*). This define enables the macro
// infrastructure; without it, all BSP_DEBUG_* macros become complete no-ops.
#define BSP_DEBUG_ENABLED

#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/mip_node.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// ============================================================================
// BSP Debug Event Types
// ============================================================================

enum class bsp_log_event_t {
  HORIZON_START,       // New horizon begins
  HORIZON_END,         // Horizon completed
  NODE_ASSIGNED,       // Node assigned to worker
  NODE_SOLVE_START,    // Worker starts solving node
  NODE_SOLVE_END,      // Worker finishes node (with result type)
  NODE_PAUSED,         // Node paused at horizon
  NODE_RESUMED,        // Paused node resumed
  NODE_BRANCHED,       // Node branched into children
  NODE_PRUNED,         // Node pruned (cutoff)
  NODE_INTEGER,        // Integer solution found
  NODE_INFEASIBLE,     // Node infeasible
  NODE_FATHOMED,       // Node fathomed
  FINAL_ID_ASSIGNED,   // Final ID assigned during sync
  HEURISTIC_RECEIVED,  // Heuristic solution received
  INCUMBENT_UPDATE,    // New best solution found
  SYNC_PHASE_START,    // Sync phase begins
  SYNC_PHASE_END,      // Sync phase ends
  WORKER_IDLE,         // Worker has no work
};

inline const char* bsp_log_event_name(bsp_log_event_t event)
{
  switch (event) {
    case bsp_log_event_t::HORIZON_START: return "HORIZON_START";
    case bsp_log_event_t::HORIZON_END: return "HORIZON_END";
    case bsp_log_event_t::NODE_ASSIGNED: return "NODE_ASSIGNED";
    case bsp_log_event_t::NODE_SOLVE_START: return "NODE_SOLVE_START";
    case bsp_log_event_t::NODE_SOLVE_END: return "NODE_SOLVE_END";
    case bsp_log_event_t::NODE_PAUSED: return "NODE_PAUSED";
    case bsp_log_event_t::NODE_RESUMED: return "NODE_RESUMED";
    case bsp_log_event_t::NODE_BRANCHED: return "NODE_BRANCHED";
    case bsp_log_event_t::NODE_PRUNED: return "NODE_PRUNED";
    case bsp_log_event_t::NODE_INTEGER: return "NODE_INTEGER";
    case bsp_log_event_t::NODE_INFEASIBLE: return "NODE_INFEASIBLE";
    case bsp_log_event_t::NODE_FATHOMED: return "NODE_FATHOMED";
    case bsp_log_event_t::FINAL_ID_ASSIGNED: return "FINAL_ID_ASSIGNED";
    case bsp_log_event_t::HEURISTIC_RECEIVED: return "HEURISTIC_RECEIVED";
    case bsp_log_event_t::INCUMBENT_UPDATE: return "INCUMBENT_UPDATE";
    case bsp_log_event_t::SYNC_PHASE_START: return "SYNC_PHASE_START";
    case bsp_log_event_t::SYNC_PHASE_END: return "SYNC_PHASE_END";
    case bsp_log_event_t::WORKER_IDLE: return "WORKER_IDLE";
    default: return "UNKNOWN";
  }
}

// ============================================================================
// BSP Debug Settings
// ============================================================================

/**
 * @brief BSP debug settings controlled via environment variables.
 *
 * Environment variables:
 *   CUOPT_BSP_DEBUG_LOG=1       Enable structured event log
 *   CUOPT_BSP_DEBUG_TIMELINE=1  Emit ASCII timeline visualization
 *   CUOPT_BSP_DEBUG_TREE=1      Emit DOT tree state files
 *   CUOPT_BSP_DEBUG_JSON=1      Emit JSON state dumps
 *   CUOPT_BSP_DEBUG_TRACE=1     Emit determinism trace file
 *   CUOPT_BSP_DEBUG_ALL=1       Enable all debug output
 *   CUOPT_BSP_DEBUG_DIR=path    Output directory (default: ./bsp_debug/)
 *   CUOPT_BSP_DEBUG_LEVEL=N     Log level: 0=off, 1=major events, 2=all events
 */
struct bsp_debug_settings_t {
  bool enable_event_log{false};
  bool enable_timeline{false};
  bool enable_tree_dot{false};
  bool enable_state_json{false};
  bool enable_determinism_trace{false};
  std::string output_dir{"./bsp_debug/"};
  int log_level{1};  // 0=off, 1=major events, 2=all events

  bool any_enabled() const
  {
    return enable_event_log || enable_timeline || enable_tree_dot || enable_state_json ||
           enable_determinism_trace;
  }

  void enable_all()
  {
    enable_event_log         = true;
    enable_timeline          = true;
    enable_tree_dot          = true;
    enable_state_json        = true;
    enable_determinism_trace = true;
  }

  /**
   * @brief Initialize settings from environment variables.
   * @return A bsp_debug_settings_t populated from environment variables.
   */
  static bsp_debug_settings_t from_environment()
  {
    bsp_debug_settings_t settings;

    auto get_env_bool = [](const char* name) -> bool {
      const char* val = std::getenv(name);
      if (val == nullptr) return false;
      return std::string(val) == "1" || std::string(val) == "true" || std::string(val) == "TRUE";
    };

    auto get_env_int = [](const char* name, int default_val) -> int {
      const char* val = std::getenv(name);
      if (val == nullptr) return default_val;
      try {
        return std::stoi(val);
      } catch (...) {
        return default_val;
      }
    };

    auto get_env_string = [](const char* name, const std::string& default_val) -> std::string {
      const char* val = std::getenv(name);
      if (val == nullptr) return default_val;
      return std::string(val);
    };

    // Check for CUOPT_BSP_DEBUG_ALL first
    if (get_env_bool("CUOPT_BSP_DEBUG_ALL")) {
      settings.enable_all();
    } else {
      settings.enable_event_log         = get_env_bool("CUOPT_BSP_DEBUG_LOG");
      settings.enable_timeline          = get_env_bool("CUOPT_BSP_DEBUG_TIMELINE");
      settings.enable_tree_dot          = get_env_bool("CUOPT_BSP_DEBUG_TREE");
      settings.enable_state_json        = get_env_bool("CUOPT_BSP_DEBUG_JSON");
      settings.enable_determinism_trace = get_env_bool("CUOPT_BSP_DEBUG_TRACE");
    }

    settings.output_dir = get_env_string("CUOPT_BSP_DEBUG_DIR", "./bsp_debug/");
    settings.log_level  = get_env_int("CUOPT_BSP_DEBUG_LEVEL", 1);

    return settings;
  }
};

// ============================================================================
// Timeline Event (for ASCII visualization)
// ============================================================================

struct timeline_event_t {
  double vt_start;
  double vt_end;
  int worker_id;
  int node_id;
  int final_id;
  std::string result;  // "BRANCH", "INTEGER", "FATHOMED", "PAUSED", etc.
};

// ============================================================================
// BSP Debug Logger Class
// ============================================================================

template <typename i_t, typename f_t>
class bsp_debug_logger_t {
 public:
  bsp_debug_logger_t() : start_time_(std::chrono::steady_clock::now()) {}

  void set_settings(const bsp_debug_settings_t& settings)
  {
    settings_ = settings;
    if (settings_.any_enabled()) {
      // Create output directory if it doesn't exist
      try {
        std::filesystem::create_directories(settings_.output_dir);
      } catch (const std::filesystem::filesystem_error& e) {
        // Silently disable debug output if we can't create the directory
        settings_.enable_event_log         = false;
        settings_.enable_timeline          = false;
        settings_.enable_tree_dot          = false;
        settings_.enable_state_json        = false;
        settings_.enable_determinism_trace = false;
      }

      // Clear timeline file at start (it uses append mode for each horizon)
      if (settings_.enable_timeline) {
        std::string filename = settings_.output_dir + "bsp_timeline.txt";
        std::ofstream file(filename, std::ios::trunc);
        if (file.is_open()) {
          file << "BSP Timeline Visualization\n";
          file << "==========================\n";
          file.close();
        }
      }
    }
  }

  void set_num_workers(int num_workers) { num_workers_ = num_workers; }

  void set_horizon_step(double step) { horizon_step_ = step; }

  // ========================================================================
  // Event Logging
  // ========================================================================

  void log_horizon_start(int horizon_num, double vt_start, double vt_end)
  {
    current_horizon_  = horizon_num;
    horizon_vt_start_ = vt_start;
    horizon_vt_end_   = vt_end;

    if (settings_.enable_event_log) {
      log_event(vt_start,
                -1,
                bsp_log_event_t::HORIZON_START,
                -1,
                -1,
                "horizon=[" + std::to_string(vt_start) + "," + std::to_string(vt_end) + ")");
    }

    if (settings_.enable_determinism_trace) {
      trace_ss_ << "H" << horizon_num << ":START:vt=[" << vt_start << "," << vt_end << ")\n";
    }

    // Clear timeline events for new horizon
    timeline_events_.clear();
  }

  void log_horizon_end(int horizon_num, double vt)
  {
    if (settings_.enable_event_log) { log_event(vt, -1, bsp_log_event_t::HORIZON_END, -1, -1, ""); }

    if (settings_.enable_timeline) { emit_timeline_for_horizon(horizon_num); }
  }

  void log_node_assigned(double vt, int worker_id, i_t node_id, i_t final_id, f_t lower_bound)
  {
    if (settings_.enable_event_log) {
      log_event(vt,
                worker_id,
                bsp_log_event_t::NODE_ASSIGNED,
                node_id,
                final_id,
                "lb=" + std::to_string(lower_bound));
    }

    if (settings_.enable_determinism_trace) {
      assign_trace_ss_ << "W" << worker_id << "=" << final_id << ",";
    }
  }

  void flush_assign_trace()
  {
    if (settings_.enable_determinism_trace && assign_trace_ss_.str().length() > 0) {
      std::string s = assign_trace_ss_.str();
      if (!s.empty() && s.back() == ',') s.pop_back();
      trace_ss_ << "H" << current_horizon_ << ":ASSIGN:" << s << "\n";
      assign_trace_ss_.str("");
      assign_trace_ss_.clear();
    }
  }

  void log_solve_start(
    double vt, int worker_id, i_t node_id, i_t final_id, double work_limit, bool is_resumed)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log && settings_.log_level >= 2) {
      log_event_unlocked(
        vt,
        worker_id,
        is_resumed ? bsp_log_event_t::NODE_RESUMED : bsp_log_event_t::NODE_SOLVE_START,
        node_id,
        final_id,
        "work_limit=" + std::to_string(work_limit));
    }

    // Start timeline event
    current_solve_start_[worker_id] = vt;
    current_solve_node_[worker_id]  = node_id;
    current_solve_fid_[worker_id]   = final_id;
  }

  void log_solve_end(
    double vt, int worker_id, i_t node_id, i_t final_id, const std::string& result, f_t lower_bound)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log) {
      log_event_unlocked(vt,
                         worker_id,
                         bsp_log_event_t::NODE_SOLVE_END,
                         node_id,
                         final_id,
                         "result=" + result + ",lb=" + std::to_string(lower_bound));
    }

    // Complete timeline event
    if (settings_.enable_timeline) {
      timeline_event_t te;
      te.vt_start  = current_solve_start_[worker_id];
      te.vt_end    = vt;
      te.worker_id = worker_id;
      te.node_id   = node_id;
      te.final_id  = final_id;
      te.result    = result;
      timeline_events_.push_back(te);
    }
  }

  void log_branched(double vt, int worker_id, i_t parent_id, i_t down_id, i_t up_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log) {
      log_event_unlocked(vt,
                         worker_id,
                         bsp_log_event_t::NODE_BRANCHED,
                         parent_id,
                         -1,
                         "children=" + std::to_string(down_id) + "," + std::to_string(up_id));
    }

    if (settings_.enable_determinism_trace) {
      events_trace_ss_ << "BRANCH(" << vt << ",W" << worker_id << "," << parent_id << "->"
                       << down_id << "," << up_id << "),";
    }
  }

  void log_paused(double vt, int worker_id, i_t node_id, i_t final_id, f_t accumulated_vt)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log) {
      log_event_unlocked(vt,
                         worker_id,
                         bsp_log_event_t::NODE_PAUSED,
                         node_id,
                         final_id,
                         "acc_vt=" + std::to_string(accumulated_vt));
    }

    if (settings_.enable_determinism_trace) {
      events_trace_ss_ << "PAUSE(" << vt << ",W" << worker_id << "," << node_id << "),";
    }
  }

  void log_integer(double vt, int worker_id, i_t node_id, f_t objective)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log) {
      log_event_unlocked(vt,
                         worker_id,
                         bsp_log_event_t::NODE_INTEGER,
                         node_id,
                         -1,
                         "obj=" + std::to_string(objective));
    }

    if (settings_.enable_determinism_trace) {
      events_trace_ss_ << "INT(" << vt << ",W" << worker_id << "," << objective << "),";
    }
  }

  void log_fathomed(double vt, int worker_id, i_t node_id, f_t lower_bound)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log && settings_.log_level >= 2) {
      log_event_unlocked(vt,
                         worker_id,
                         bsp_log_event_t::NODE_FATHOMED,
                         node_id,
                         -1,
                         "lb=" + std::to_string(lower_bound));
    }
  }

  void log_infeasible(double vt, int worker_id, i_t node_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log && settings_.log_level >= 2) {
      log_event_unlocked(vt, worker_id, bsp_log_event_t::NODE_INFEASIBLE, node_id, -1, "");
    }
  }

  void log_pruned(double vt, i_t node_id, i_t final_id, f_t lower_bound, f_t upper_bound)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (settings_.enable_event_log && settings_.log_level >= 2) {
      log_event_unlocked(
        vt,
        -1,
        bsp_log_event_t::NODE_PRUNED,
        node_id,
        final_id,
        "lb=" + std::to_string(lower_bound) + ",ub=" + std::to_string(upper_bound));
    }
  }

  void log_sync_phase_start(double vt, size_t num_events)
  {
    if (settings_.enable_event_log) {
      log_event(vt,
                -1,
                bsp_log_event_t::SYNC_PHASE_START,
                -1,
                -1,
                "num_events=" + std::to_string(num_events));
    }
  }

  void log_sync_phase_end(double vt)
  {
    if (settings_.enable_event_log) {
      log_event(vt, -1, bsp_log_event_t::SYNC_PHASE_END, -1, -1, "");
    }

    // Flush event trace
    if (settings_.enable_determinism_trace) {
      std::string s = events_trace_ss_.str();
      if (!s.empty() && s.back() == ',') s.pop_back();
      trace_ss_ << "H" << current_horizon_ << ":EVENTS:" << s << "\n";
      events_trace_ss_.str("");
      events_trace_ss_.clear();
    }
  }

  void log_final_id_assigned(i_t provisional_id, i_t final_id)
  {
    if (settings_.enable_determinism_trace) {
      final_ids_trace_ss_ << provisional_id << "->" << final_id << ",";
    }
  }

  void flush_final_ids_trace()
  {
    if (settings_.enable_determinism_trace && final_ids_trace_ss_.str().length() > 0) {
      std::string s = final_ids_trace_ss_.str();
      if (!s.empty() && s.back() == ',') s.pop_back();
      trace_ss_ << "H" << current_horizon_ << ":FINAL_IDS:" << s << "\n";
      final_ids_trace_ss_.str("");
      final_ids_trace_ss_.clear();
    }
  }

  void log_heuristic_received(double vt, f_t objective)
  {
    if (settings_.enable_event_log) {
      log_event(
        vt, -1, bsp_log_event_t::HEURISTIC_RECEIVED, -1, -1, "obj=" + std::to_string(objective));
    }

    if (settings_.enable_determinism_trace) {
      trace_ss_ << "H" << current_horizon_ << ":HEURISTIC:" << objective << "@" << vt << "\n";
    }
  }

  void log_incumbent_update(double vt, f_t objective, const std::string& source)
  {
    if (settings_.enable_event_log) {
      log_event(vt,
                -1,
                bsp_log_event_t::INCUMBENT_UPDATE,
                -1,
                -1,
                "obj=" + std::to_string(objective) + ",source=" + source);
    }

    if (settings_.enable_determinism_trace) {
      trace_ss_ << "H" << current_horizon_ << ":INCUMBENT:" << objective << "@" << vt << "("
                << source << ")\n";
    }
  }

  void log_heap_order(const std::vector<i_t>& final_ids)
  {
    if (settings_.enable_determinism_trace) {
      trace_ss_ << "H" << current_horizon_ << ":HEAP_ORDER:";
      for (size_t i = 0; i < final_ids.size(); ++i) {
        trace_ss_ << final_ids[i];
        if (i < final_ids.size() - 1) trace_ss_ << ",";
      }
      trace_ss_ << "\n";
    }
  }

  // ========================================================================
  // Tree State (DOT format)
  // ========================================================================

  void emit_tree_state(int horizon_num, const mip_node_t<i_t, f_t>& root, f_t upper_bound)
  {
    if (!settings_.enable_tree_dot) return;

    std::string filename =
      settings_.output_dir + "bsp_tree_h" + std::to_string(horizon_num) + ".dot";
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "digraph BSPTree_Horizon" << horizon_num << " {\n";
    file << "  rankdir=TB;\n";
    file << "  node [shape=box];\n\n";

    // Traverse tree and emit nodes
    emit_tree_node_recursive(file, &root, upper_bound);

    file << "}\n";
    file.close();
  }

  // ========================================================================
  // JSON State Dump
  // ========================================================================

  template <typename WorkerPool>
  void emit_state_json(int horizon_num,
                       double vt_start,
                       double vt_end,
                       i_t next_final_id,
                       f_t upper_bound,
                       f_t lower_bound,
                       i_t nodes_explored,
                       i_t nodes_unexplored,
                       const WorkerPool& workers,
                       const std::vector<mip_node_t<i_t, f_t>*>& heap_nodes,
                       const bb_event_batch_t<i_t, f_t>& events)
  {
    if (!settings_.enable_state_json) return;

    std::string filename =
      settings_.output_dir + "bsp_state_h" + std::to_string(horizon_num) + ".json";
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "{\n";
    file << "  \"horizon\": " << horizon_num << ",\n";
    file << "  \"vt_range\": [" << vt_start << ", " << vt_end << "],\n";
    file << "  \"bsp_next_final_id\": " << next_final_id << ",\n";
    file << "  \"upper_bound\": " << (upper_bound >= 1e30 ? "\"inf\"" : std::to_string(upper_bound))
         << ",\n";
    file << "  \"lower_bound\": " << lower_bound << ",\n";
    file << "  \"nodes_explored\": " << nodes_explored << ",\n";
    file << "  \"nodes_unexplored\": " << nodes_unexplored << ",\n";

    // Workers
    file << "  \"workers\": [\n";
    for (int i = 0; i < workers.size(); ++i) {
      const auto& w = workers[i];
      file << "    {\n";
      file << "      \"id\": " << i << ",\n";
      file << "      \"clock\": " << w.clock << ",\n";
      if (w.current_node != nullptr) {
        file << "      \"current_node\": {\"id\": " << w.current_node->node_id
             << ", \"final_id\": " << w.current_node->final_id
             << ", \"acc_vt\": " << w.current_node->accumulated_vt << "},\n";
      } else {
        file << "      \"current_node\": null,\n";
      }
      file << "      \"local_queue_size\": " << w.local_queue.size() << "\n";
      file << "    }" << (i < workers.size() - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    // Heap
    file << "  \"heap\": [\n";
    for (size_t i = 0; i < heap_nodes.size(); ++i) {
      const auto* n = heap_nodes[i];
      file << "    {\"id\": " << n->node_id << ", \"final_id\": " << n->final_id
           << ", \"lb\": " << n->lower_bound << "}" << (i < heap_nodes.size() - 1 ? "," : "")
           << "\n";
    }
    file << "  ],\n";

    // Events this horizon
    file << "  \"events_count\": " << events.events.size() << "\n";

    file << "}\n";
    file.close();
  }

  // ========================================================================
  // Finalization
  // ========================================================================

  void finalize()
  {
    if (settings_.enable_event_log) { flush_event_log(); }

    if (settings_.enable_determinism_trace) { flush_trace(); }
  }

 private:
  bsp_debug_settings_t settings_;
  std::chrono::steady_clock::time_point start_time_;
  int num_workers_{0};
  double horizon_step_{5.0};
  int current_horizon_{0};
  double horizon_vt_start_{0.0};
  double horizon_vt_end_{0.0};

  std::mutex mutex_;
  std::stringstream log_ss_;
  std::stringstream trace_ss_;
  std::stringstream assign_trace_ss_;
  std::stringstream events_trace_ss_;
  std::stringstream final_ids_trace_ss_;

  // Timeline tracking
  std::vector<timeline_event_t> timeline_events_;
  std::map<int, double> current_solve_start_;
  std::map<int, i_t> current_solve_node_;
  std::map<int, i_t> current_solve_fid_;

  double get_real_time() const
  {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
  }

  void log_event(double vt,
                 int worker_id,
                 bsp_log_event_t event,
                 i_t node_id,
                 i_t final_id,
                 const std::string& details)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    log_event_unlocked(vt, worker_id, event, node_id, final_id, details);
  }

  void log_event_unlocked(double vt,
                          int worker_id,
                          bsp_log_event_t event,
                          i_t node_id,
                          i_t final_id,
                          const std::string& details)
  {
    log_ss_ << std::fixed << std::setprecision(3) << vt << "\t" << get_real_time() << "\t"
            << current_horizon_ << "\t" << (worker_id < 0 ? "COORD" : std::to_string(worker_id))
            << "\t" << bsp_log_event_name(event) << "\t"
            << (node_id < 0 ? "-" : std::to_string(node_id)) << "\t"
            << (final_id < 0 ? "-" : std::to_string(final_id)) << "\t" << details << "\n";
  }

  void flush_event_log()
  {
    std::string filename = settings_.output_dir + "bsp_events.log";
    std::ofstream file(filename);
    if (file.is_open()) {
      file << "VT\tREAL_TIME\tHORIZON\tWORKER\tEVENT\tNODE_ID\tFINAL_ID\tDETAILS\n";
      file << log_ss_.str();
      file.close();
    }
  }

  void flush_trace()
  {
    std::string filename = settings_.output_dir + "bsp_trace.txt";
    std::ofstream file(filename);
    if (file.is_open()) {
      file << "# BSP Determinism Trace\n";
      file << "# Compare this file across runs - must be identical for determinism\n\n";
      file << trace_ss_.str();
      file.close();
    }
  }

  void emit_timeline_for_horizon(int horizon_num)
  {
    std::string filename = settings_.output_dir + "bsp_timeline.txt";
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;

    const int width       = 80;
    const double vt_min   = horizon_vt_start_;
    const double vt_max   = horizon_vt_end_;
    const double vt_range = vt_max - vt_min;

    // Avoid division by zero
    if (vt_range <= 0.0) {
      file << "\nHorizon " << horizon_num << ": Empty or invalid VT range\n";
      return;
    }

    file << "\n";
    file << std::string(width, '=') << "\n";
    file << "Horizon " << horizon_num << ": VT [" << vt_min << ", " << vt_max << ")\n";
    file << std::string(width, '-') << "\n";

    // VT scale
    file << "     VT: ";
    for (int i = 0; i <= 5; ++i) {
      double vt = vt_min + (vt_range * i / 5.0);
      file << std::setw(10) << std::fixed << std::setprecision(1) << vt;
    }
    file << "\n";

    // Worker rows
    for (int w = 0; w < num_workers_; ++w) {
      file << "  W" << w << ":    ";

      std::string row(60, '-');

      for (const auto& te : timeline_events_) {
        if (te.worker_id != w) continue;

        int start_col = static_cast<int>((te.vt_start - vt_min) / vt_range * 60);
        int end_col   = static_cast<int>((te.vt_end - vt_min) / vt_range * 60);
        start_col     = std::max(0, std::min(59, start_col));
        end_col       = std::max(0, std::min(59, end_col));

        char fill_char = '#';
        if (te.result == "PAUSED")
          fill_char = '%';
        else if (te.result == "INTEGER")
          fill_char = '*';
        else if (te.result == "FATHOMED" || te.result == "INFEASIBLE")
          fill_char = 'x';

        for (int c = start_col; c <= end_col; ++c) {
          row[c] = fill_char;
        }
      }

      file << row << "\n";

      // Labels row
      file << "         ";
      std::string labels(60, ' ');
      for (const auto& te : timeline_events_) {
        if (te.worker_id != w) continue;
        int mid_col = static_cast<int>(((te.vt_start + te.vt_end) / 2 - vt_min) / vt_range * 60);
        mid_col     = std::max(0, std::min(55, mid_col));

        std::string label = "N" + std::to_string(te.final_id);
        for (size_t i = 0; i < label.size() && mid_col + i < 60; ++i) {
          labels[mid_col + i] = label[i];
        }
      }
      file << labels << "\n";
    }

    file << std::string(width, '-') << "\n";
    file << "Legend: ### solving  %%% paused  *** integer  xxx fathomed/infeasible\n";
    file << std::string(width, '=') << "\n";
  }

  void emit_tree_node_recursive(std::ofstream& file,
                                const mip_node_t<i_t, f_t>* node,
                                f_t upper_bound)
  {
    if (node == nullptr) return;

    // Determine fill color based on status
    std::string color = "white";
    std::string status_str;
    switch (node->status) {
      case node_status_t::PENDING:
        color      = "lightgray";
        status_str = "PENDING";
        break;
      case node_status_t::INTEGER_FEASIBLE:
        color      = "lightgreen";
        status_str = "INTEGER";
        break;
      case node_status_t::INFEASIBLE:
        color      = "lightcoral";
        status_str = "INFEASIBLE";
        break;
      case node_status_t::FATHOMED:
        color      = "lightsalmon";
        status_str = "FATHOMED";
        break;
      case node_status_t::HAS_CHILDREN:
        color      = "lightblue";
        status_str = "BRANCHED";
        break;
      case node_status_t::NUMERICAL:
        color      = "orange";
        status_str = "NUMERICAL";
        break;
    }

    if (node->bsp_state == bsp_node_state_t::PAUSED) {
      color      = "yellow";
      status_str = "PAUSED";
    }

    file << "  N" << node->node_id << " [label=\"N" << node->node_id;
    if (node->final_id >= 0) file << " (fid=" << node->final_id << ")";
    file << "\\nlb=" << std::fixed << std::setprecision(1) << node->lower_bound;
    file << "\\n" << status_str;
    if (node->bsp_state == bsp_node_state_t::PAUSED) {
      file << "\\nacc_vt=" << node->accumulated_vt;
    }
    file << "\" style=filled fillcolor=" << color << "];\n";

    // Emit edges to children
    if (node->get_down_child() != nullptr) {
      file << "  N" << node->node_id << " -> N" << node->get_down_child()->node_id << " [label=\"x"
           << node->branch_var << "<=" << std::floor(node->fractional_val) << "\"];\n";
      emit_tree_node_recursive(file, node->get_down_child(), upper_bound);
    }

    if (node->get_up_child() != nullptr) {
      file << "  N" << node->node_id << " -> N" << node->get_up_child()->node_id << " [label=\"x"
           << node->branch_var << ">=" << std::ceil(node->fractional_val) << "\"];\n";
      emit_tree_node_recursive(file, node->get_up_child(), upper_bound);
    }
  }
};

// ============================================================================
// Convenience Macros
// ============================================================================
//
// These macros provide a clean interface for BSP debug logging.
// - When BSP_DEBUG_ENABLED is defined: macros are active (controlled by settings at runtime)
// - When BSP_DEBUG_ENABLED is not defined: macros are no-ops (zero overhead)
//
// The 'settings' parameter is a bsp_debug_settings_t& that controls which features are enabled.

#ifdef BSP_DEBUG_ENABLED

#define BSP_DEBUG_LOG_HORIZON_START(settings, logger, h, vs, ve)         \
  do {                                                                   \
    if ((settings).any_enabled()) (logger).log_horizon_start(h, vs, ve); \
  } while (0)
#define BSP_DEBUG_LOG_HORIZON_END(settings, logger, h, vt)         \
  do {                                                             \
    if ((settings).any_enabled()) (logger).log_horizon_end(h, vt); \
  } while (0)
#define BSP_DEBUG_LOG_NODE_ASSIGNED(settings, logger, vt, w, nid, fid, lb)         \
  do {                                                                             \
    if ((settings).any_enabled()) (logger).log_node_assigned(vt, w, nid, fid, lb); \
  } while (0)
#define BSP_DEBUG_FLUSH_ASSIGN_TRACE(settings, logger)           \
  do {                                                           \
    if ((settings).any_enabled()) (logger).flush_assign_trace(); \
  } while (0)
#define BSP_DEBUG_LOG_SOLVE_START(settings, logger, vt, w, nid, fid, wl, resumed)         \
  do {                                                                                    \
    if ((settings).any_enabled()) (logger).log_solve_start(vt, w, nid, fid, wl, resumed); \
  } while (0)
#define BSP_DEBUG_LOG_SOLVE_END(settings, logger, vt, w, nid, fid, result, lb)         \
  do {                                                                                 \
    if ((settings).any_enabled()) (logger).log_solve_end(vt, w, nid, fid, result, lb); \
  } while (0)
#define BSP_DEBUG_LOG_BRANCHED(settings, logger, vt, w, pid, did, uid)         \
  do {                                                                         \
    if ((settings).any_enabled()) (logger).log_branched(vt, w, pid, did, uid); \
  } while (0)
#define BSP_DEBUG_LOG_PAUSED(settings, logger, vt, w, nid, fid, acc)         \
  do {                                                                       \
    if ((settings).any_enabled()) (logger).log_paused(vt, w, nid, fid, acc); \
  } while (0)
#define BSP_DEBUG_LOG_INTEGER(settings, logger, vt, w, nid, obj)         \
  do {                                                                   \
    if ((settings).any_enabled()) (logger).log_integer(vt, w, nid, obj); \
  } while (0)
#define BSP_DEBUG_LOG_FATHOMED(settings, logger, vt, w, nid, lb)         \
  do {                                                                   \
    if ((settings).any_enabled()) (logger).log_fathomed(vt, w, nid, lb); \
  } while (0)
#define BSP_DEBUG_LOG_INFEASIBLE(settings, logger, vt, w, nid)         \
  do {                                                                 \
    if ((settings).any_enabled()) (logger).log_infeasible(vt, w, nid); \
  } while (0)
#define BSP_DEBUG_LOG_PRUNED(settings, logger, vt, nid, fid, lb, ub)         \
  do {                                                                       \
    if ((settings).any_enabled()) (logger).log_pruned(vt, nid, fid, lb, ub); \
  } while (0)
#define BSP_DEBUG_LOG_SYNC_PHASE_START(settings, logger, vt, ne)         \
  do {                                                                   \
    if ((settings).any_enabled()) (logger).log_sync_phase_start(vt, ne); \
  } while (0)
#define BSP_DEBUG_LOG_SYNC_PHASE_END(settings, logger, vt)         \
  do {                                                             \
    if ((settings).any_enabled()) (logger).log_sync_phase_end(vt); \
  } while (0)
#define BSP_DEBUG_LOG_FINAL_ID_ASSIGNED(settings, logger, pid, fid)         \
  do {                                                                      \
    if ((settings).any_enabled()) (logger).log_final_id_assigned(pid, fid); \
  } while (0)
#define BSP_DEBUG_FLUSH_FINAL_IDS_TRACE(settings, logger)           \
  do {                                                              \
    if ((settings).any_enabled()) (logger).flush_final_ids_trace(); \
  } while (0)
#define BSP_DEBUG_LOG_HEURISTIC_RECEIVED(settings, logger, vt, obj)         \
  do {                                                                      \
    if ((settings).any_enabled()) (logger).log_heuristic_received(vt, obj); \
  } while (0)
#define BSP_DEBUG_LOG_INCUMBENT_UPDATE(settings, logger, vt, obj, src)         \
  do {                                                                         \
    if ((settings).any_enabled()) (logger).log_incumbent_update(vt, obj, src); \
  } while (0)
#define BSP_DEBUG_LOG_HEAP_ORDER(settings, logger, fids)         \
  do {                                                           \
    if ((settings).any_enabled()) (logger).log_heap_order(fids); \
  } while (0)
#define BSP_DEBUG_EMIT_TREE_STATE(settings, logger, h, root, ub)         \
  do {                                                                   \
    if ((settings).any_enabled()) (logger).emit_tree_state(h, root, ub); \
  } while (0)
#define BSP_DEBUG_EMIT_STATE_JSON(                                                      \
  settings, logger, h, vs, ve, nfid, ub, lb, ne, nu, workers, heap, events)             \
  do {                                                                                  \
    if ((settings).any_enabled())                                                       \
      (logger).emit_state_json(h, vs, ve, nfid, ub, lb, ne, nu, workers, heap, events); \
  } while (0)
#define BSP_DEBUG_FINALIZE(settings, logger)           \
  do {                                                 \
    if ((settings).any_enabled()) (logger).finalize(); \
  } while (0)

#else

#define BSP_DEBUG_LOG_HORIZON_START(settings, logger, h, vs, ve)                  ((void)0)
#define BSP_DEBUG_LOG_HORIZON_END(settings, logger, h, vt)                        ((void)0)
#define BSP_DEBUG_LOG_NODE_ASSIGNED(settings, logger, vt, w, nid, fid, lb)        ((void)0)
#define BSP_DEBUG_FLUSH_ASSIGN_TRACE(settings, logger)                            ((void)0)
#define BSP_DEBUG_LOG_SOLVE_START(settings, logger, vt, w, nid, fid, wl, resumed) ((void)0)
#define BSP_DEBUG_LOG_SOLVE_END(settings, logger, vt, w, nid, fid, result, lb)    ((void)0)
#define BSP_DEBUG_LOG_BRANCHED(settings, logger, vt, w, pid, did, uid)            ((void)0)
#define BSP_DEBUG_LOG_PAUSED(settings, logger, vt, w, nid, fid, acc)              ((void)0)
#define BSP_DEBUG_LOG_INTEGER(settings, logger, vt, w, nid, obj)                  ((void)0)
#define BSP_DEBUG_LOG_FATHOMED(settings, logger, vt, w, nid, lb)                  ((void)0)
#define BSP_DEBUG_LOG_INFEASIBLE(settings, logger, vt, w, nid)                    ((void)0)
#define BSP_DEBUG_LOG_PRUNED(settings, logger, vt, nid, fid, lb, ub)              ((void)0)
#define BSP_DEBUG_LOG_SYNC_PHASE_START(settings, logger, vt, ne)                  ((void)0)
#define BSP_DEBUG_LOG_SYNC_PHASE_END(settings, logger, vt)                        ((void)0)
#define BSP_DEBUG_LOG_FINAL_ID_ASSIGNED(settings, logger, pid, fid)               ((void)0)
#define BSP_DEBUG_FLUSH_FINAL_IDS_TRACE(settings, logger)                         ((void)0)
#define BSP_DEBUG_LOG_HEURISTIC_RECEIVED(settings, logger, vt, obj)               ((void)0)
#define BSP_DEBUG_LOG_INCUMBENT_UPDATE(settings, logger, vt, obj, src)            ((void)0)
#define BSP_DEBUG_LOG_HEAP_ORDER(settings, logger, fids)                          ((void)0)
#define BSP_DEBUG_EMIT_TREE_STATE(settings, logger, h, root, ub)                  ((void)0)
#define BSP_DEBUG_EMIT_STATE_JSON(                                          \
  settings, logger, h, vs, ve, nfid, ub, lb, ne, nu, workers, heap, events) \
  ((void)0)
#define BSP_DEBUG_FINALIZE(settings, logger) ((void)0)

#endif

}  // namespace cuopt::linear_programming::dual_simplex
