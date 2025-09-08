////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisMultiLognormal
///
/// \brief Apply multi lognormal fit to digitized waveforms
///
/// \author Ravi Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     08 Sep 2025: Initial commit
///
/// \details
/// This class provides full support for analysis of the
/// digitized waveform via a multi lognormal fit,
/// using npe estimation from likelihood processor.
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisMultiLognormal__
#define __RAT_WaveformAnalysisMultiLognormal__

#include <TObject.h>

#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Digitizer.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <vector>

namespace RAT {

class WaveformAnalysisMultiLognormal : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisMultiLognormal() : WaveformAnalysisMultiLognormal("MultiLognormalFit"){};
  WaveformAnalysisMultiLognormal(std::string config_name)
      : WaveformAnalyzerBase("WaveformAnalysisMultiLognormal", config_name) {
    Configure(config_name);
  };
  virtual ~WaveformAnalysisMultiLognormal(){};
  void Configure(const std::string &config_name) override;
  virtual void SetD(std::string param, double value) override;

  // Fit the digitized waveform using a lognormal function
  void FitWaveform(const std::vector<double> &voltWfm);

 protected:
  // Digitizer settings
  DBLinkPtr fDigit;

  // Analysis constants
  double fFitShape;
  double fFitScale;

  // Coming from WaveformPrep
  double fDigitTimeInWindow;

  // Coming from nhit predictor
  double fNPE;

  // Fitted variables
  std::vector<double> fFittedTimes;
  std::vector<double> fFittedCharges;
  double fFittedBaseline;
  double fChi2NDF;

  void DoAnalysis(DS::DigitPMT *pmt, const std::vector<UShort_t> &digitWfm) override;
};

}  // namespace RAT

#endif
