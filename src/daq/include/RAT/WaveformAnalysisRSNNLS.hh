////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisRSNNLS
///
/// \brief Apply lognormal fit to digitized waveforms
///
/// \author Ravi Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     XX Jun 2025: Initial commit
///
/// \details
/// This class provides full support for analysis of the
/// digitized waveform via
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisRSNNLS__
#define __RAT_WaveformAnalysisRSNNLS__

#include <TObject.h>

#include <Eigen/Dense>
#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Digitizer.hh>
#include <RAT/PMTPulse.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <vector>

namespace RAT {

class WaveformAnalysisRSNNLS : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisRSNNLS() : WaveformAnalysisRSNNLS("rsNNLS"){};
  WaveformAnalysisRSNNLS(std::string config_name) : WaveformAnalyzerBase("WaveformAnalysisRSNNLS", config_name) {
    Configure(config_name);
  };
  virtual ~WaveformAnalysisRSNNLS(){};
  void Configure(const std::string &config_name) override;
  virtual void SetD(std::string param, double value) override;
  virtual void SetI(std::string param, int value) override;

  PMTPulse MakeTemplate(DS::DigitPMT *digitpmt);
  Eigen::MatrixXd MakeTemplateMatrix(PMTPulse *pulse_template, int signal_length, int upsample_factor);
  std::pair<std::vector<int>, Eigen::VectorXd> Thresholded_rsNNLS(const Eigen::VectorXd &m, const Eigen::MatrixXd &S,
                                                                  double threshold = 1.0);

 protected:
  // Digitizer settings
  DBLinkPtr fDigit;

  // Pulse shape parameters
  double fPulseWidthMean;
  double fChargeMean;
  double fWeightThreshold;

  // Analysis constants
  int fUpsampleFactor;

  // Coming from WaveformPrep
  double fDigitTimeInWindow;

  void DoAnalysis(DS::DigitPMT *pmt, const std::vector<UShort_t> &digitWfm) override;
};

}  // namespace RAT

#endif
