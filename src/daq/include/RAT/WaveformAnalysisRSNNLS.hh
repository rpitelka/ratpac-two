////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisRSNNLS
///
/// \brief Perform reverse sparse non-negative least squares fitting on digitized waveforms
///
/// \author Ravi Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     21 Aug 2025: Initial commit
///
/// \details
/// This class provides full support for analysis of the
/// digitized waveform via reverse sparse non-negative least squares fitting.
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisRSNNLS__
#define __RAT_WaveformAnalysisRSNNLS__

#include <TObject.h>

#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Digitizer.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <vector>

#include "TMatrixD.h"

namespace RAT {

class WaveformAnalysisRSNNLS : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisRSNNLS() : WaveformAnalysisRSNNLS("rsNNLS"){};
  WaveformAnalysisRSNNLS(std::string config_name) : WaveformAnalyzerBase("WaveformAnalysisRSNNLS", config_name) {
    Configure(config_name);
  };
  virtual ~WaveformAnalysisRSNNLS(){};

  // Build the dictionary matrix for NNLS fitting
  void BuildDictionaryMatrix();

  void Configure(const std::string &config_name) override;

 protected:
  // Digitizer settings
  DBLinkPtr fDigit;

  // shaping parameters for SPE waveform
  double vpe_scale;   // Lognormal `m`
  double vpe_shape;   // Lognormal `sigma`
  double vpe_charge;  // nominal charge of a PE.
  double vpe_integral =
      -9999;  // to be overwritten by the charge * termohms. Set to invalid here because termOhms is not known yet.
  double ds;  // internal sampling period in ns.
  size_t vpe_nsamples;  // number of samples in the VPE waveform.
  std::vector<double> vpe_norm,
      vpe_norm_flipped;  // single PE waveform and its flipped version. Both normalized to integral 1.
  TMatrixD fW;           // Dictionary matrix for NNLS

  double epsilon = 1e-10;  // Small value to avoid division by zero in deconvolution

  size_t max_iterations;  // max iterations to run in deconvolution

  double weight_threshold;  // minimum weight for a component to be considered significant
  double charge_threshold;  // remove a hit if it has charge below this threshold.

  // Analysis constants
  double upsample_factor;

  double fVoltageRes;                // Voltage resolution for ADC conversion
  std::vector<double> reco_times;    // Reconstructed PE times
  std::vector<double> reco_charges;  // Reconstructed PE charges
  double chi2ndf;                    // Chi-squared per degree of freedom
  int iterations_ran;                // Number of iterations performed

  void DoAnalysis(DS::DigitPMT *pmt, const std::vector<UShort_t> &digitWfm) override;

  // Fit the digitized waveform using reverse sparse non-negative least squares
  TVectorD Thresholded_rsNNLS(const std::vector<double> &voltWfm, const double threshold);
};

}  // namespace RAT

#endif
