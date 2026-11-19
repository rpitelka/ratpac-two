////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisNPEEstimation
///
/// \brief Estimate the number of photoelectrons in each pulse reconstructed by another waveform analyzer
///
/// \author Ravi Carpen Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     03 Nov 2026: Initial commit
///     16 Nov 2026: Single-PE charge per PMT model from PMTCHARGE, PE time spread
///     17 Nov 2026: PE time spread per pulse from a seed figure of merit
///
/// \details
/// NPEEstimation starts from the photoelectrons reconstructed by a seed analyzer (e.g. RAVEN or
/// MinuitFit), which must run before it in the macro:
/// 1. Reads the seed PE times and charges. Repeated times, from an analyzer that already split a
///    pulse into integer PEs, are summed back into one pulse
/// 2. Estimates the number of PEs in each pulse from its charge, with a gaussian single-PE
///    charge PDF centered on spe_charge scaled by the channel's charge_scale calibration. With
///    spe_charge_from_pmtcharge, its mean and width are those of the PMTCHARGE distribution of the
///    channel's PMT model instead
/// 3. Emits that many PEs sharing the pulse charge, at the pulse time, or with pe_time_spread at
///    the quantiles of a gaussian of that sigma centered on it. With pe_time_spread_fom, the sigma
///    of each pulse is that seed figure of merit (e.g. MinuitFit's pe_spread), and pe_time_spread
///    is used where it is invalid
///
/// Because pulses are rebuilt from repeated times, running it on its own output gives the same
/// result unless the PE times are spread. The waveform itself is not used.
///
/// The result is stored under output_name, so several instances can run in one macro.
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisNPEEstimation__
#define __RAT_WaveformAnalysisNPEEstimation__

#include <TObject.h>

#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace RAT {

class WaveformAnalysisNPEEstimation : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisNPEEstimation() : WaveformAnalysisNPEEstimation("NPEEstimation"){};

  WaveformAnalysisNPEEstimation(std::string config_name)
      : WaveformAnalyzerBase("WaveformAnalysisNPEEstimation", config_name) {
    Configure(config_name);
  };

  virtual ~WaveformAnalysisNPEEstimation(){};

  void Configure(const std::string &config_name) override;

  void SetS(std::string param, std::string value) override;
  void SetD(std::string param, double value) override;
  void SetI(std::string param, int value) override;

  /// The configured output_name rather than the processor name
  std::string GetAnalyzerName() const override;

 protected:
  DBLinkPtr fDigit;

  std::string seed_analyzer;  ///< Analyzer whose result is read
  std::string output_name;    ///< Name the result is stored under
  bool seed_missing_warned;   ///< Whether the missing seed result has already been reported

  double spe_charge;               ///< Mean single-PE charge (pC), before the per-channel charge_scale
  double spe_charge_width;         ///< Width of the single-PE charge distribution (pC)
  bool spe_charge_from_pmtcharge;  ///< Take the single-PE charge mean and width per PMT model from PMTCHARGE
  size_t max_pes;                  ///< Upper limit for the number of PEs in one pulse
  double pe_time_spread;           ///< Sigma (ns) of the gaussian the PEs of a pulse are spread over; 0 = none
  std::string pe_time_spread_fom;  ///< Seed figure of merit giving the sigma per pulse; empty = pe_time_spread
  bool spread_fom_missing_warned;  ///< Whether the missing pe_time_spread_fom has already been reported

  /// PMTCHARGE mean and standard deviation (pC) per PMT model
  std::map<std::string, std::pair<double, double>> fModelSPECache;

  void DoAnalysis(DS::DigitPMT *digitpmt, const std::vector<UShort_t> &digitWfm) override;

  /// Die unless the result read and the result written are different
  void CheckNames() const;

  /// Single-PE charge mean and width (pC) of a PMT model's PMTCHARGE distribution, cached
  std::pair<double, double> SPEForModel(const std::string &model_name);
};

}  // namespace RAT

#endif
