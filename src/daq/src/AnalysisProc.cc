#include <RAT/AnalysisProc.hh>
#include <RAT/Log.hh>
#include <RAT/WaveformAnalysis.hh>
#include <vector>

namespace RAT {

AnalysisProc::AnalysisProc() : Processor("analysis") {}

Processor::Result AnalysisProc::Event(RAT_DS& ds, RAT_EV& ev) { return Processor::OK; }

std::tuple<std::vector<std::pair<int, int>>, int> FindCrossings(const std::vector<double>& waveform, double threshold,
                                                                int crossing_diff = 1) {
  std::vector<std::pair<int, int>> crossings;
  int crossing_time = 0;

  bool crossed = false;
  int cross_start = -1;

  // Scan over the entire waveform
  for (int i = 0; i < waveform.size(); ++i) {
    double voltage = waveform[i];

    // If we crossed below threshold
    if (voltage < -threshold) {
      if (!crossed) {
        cross_start = std::max(i - crossing_diff, 0);  // Include buffer before crossing
      }
      crossed = true;
      crossing_time++;
    }
    // If we are above threshold
    else if (voltage >= -threshold) {
      if (crossed) {
        int cross_end =
            std::min(i + crossing_diff, static_cast<int>(waveform.size()));  // Include buffer after crossing
        if (!crossings.empty() && cross_start - crossings.back().second <= crossing_diff) {
          // Join with previous crossing if close enough
          crossings.back().second = cross_end;
        } else {
          crossings.push_back(std::make_pair(cross_start, cross_end));
        }
      }
      crossed = false;
    }
  }

  // If the last region is still crossing, close it
  if (crossed) {
    int cross_end = std::min(static_cast<int>(waveform.size()), static_cast<int>(waveform.size()) + crossing_diff);
    if (!crossings.empty() && cross_start - crossings.back().second <= crossing_diff) {
      crossings.back().second = cross_end;
    } else {
      crossings.push_back(std::make_pair(cross_start, cross_end));
    }
  }

  return std::make_tuple(crossings, crossing_time);
}

}  // namespace RAT