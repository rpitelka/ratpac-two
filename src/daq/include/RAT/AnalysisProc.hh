#ifndef __RAT_AnalysisProc__
#define __RAT_AnalysisProc__

#include <RAT/Processor.hh>
#include <tuple>
#include <vector>

namespace RAT {

class AnalysisProc : public Processor {
 public:
  AnalysisProc();
  virtual ~AnalysisProc();
  virtual Processor::Result Event(RAT_DS& ds, RAT_EV& ev);
  std::tuple<std::vector<std::pair<int, int>>, int> FindCrossings(
      const std::vector<double>& waveform,
      double threshold);  // Returns vector of tuples (crossing_start, crossing_end)
  std::vector<double> Correlate(const std::vector<double>& waveform, );

 protected:
}
}  // namespace RAT

#endif