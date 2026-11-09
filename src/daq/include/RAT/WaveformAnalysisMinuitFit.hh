////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisMinuitFit
///
/// \brief Refine the photoelectron times and charges of another waveform analyzer with a Minuit fit
///
/// \author Ravi Carpen Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     29 Oct 2026: Initial commit
///
/// \details
/// MinuitFit starts from the photoelectrons reconstructed by a seed analyzer (e.g. RAVEN),
/// which must run before it in the macro, and fits them to the waveform with Minuit2:
/// 1. Reads the seed PE times and charges. Repeated times, from an analyzer splitting a hit
///    into integer PEs, are spread apart so the fit can resolve them
/// 2. Groups PEs whose fit windows overlap into clusters, each fit independently
/// 3. Minimizes the chi-square of each cluster over the PE times, each confined to a window
///    around its seed time, the PE charges, a template width scale shared by the cluster and,
///    optionally, a baseline offset
/// 4. Emits the fitted PEs, or the seed PEs with fit_valid = 0 when the fit fails or the
///    cluster has too many PEs to fit
///
/// The PE count is fixed by the seed; the fit moves PEs but never adds or removes them.
///
/// The SPE template matches RAVEN's: the shape comes per PMT from PMTPULSE, the table the
/// waveform generator samples, scaled by the per-channel pulse_width_scale calibration.
///
/// Template types supported:
/// - Lognormal
/// - Gaussian
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisMinuitFit__
#define __RAT_WaveformAnalysisMinuitFit__

#include <Math/Minimizer.h>
#include <TObject.h>

#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Digitizer.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace RAT {

class WaveformAnalysisMinuitFit : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisMinuitFit() : WaveformAnalysisMinuitFit("MinuitFit"){};

  WaveformAnalysisMinuitFit(std::string config_name) : WaveformAnalyzerBase("WaveformAnalysisMinuitFit", config_name) {
    Configure(config_name);
  };

  virtual ~WaveformAnalysisMinuitFit(){};

  /// SPE template shape for one PMT: the lognormal 'sigma' and 'm'; the gaussian
  /// template uses only the second, as its sigma.
  using TemplateShape = std::pair<double, double>;

  void Configure(const std::string &config_name) override;

  void SetS(std::string param, std::string value) override;
  void SetD(std::string param, double value) override;
  void SetI(std::string param, int value) override;

  /// Chi-square of the current cluster for parameters x
  double Chi2(const double *x);

  /// Gradient of Chi2() with respect to every parameter
  void Chi2Gradient(const double *x, double *grad);

 protected:
  /// One seed PE, in raw digitizer time and in charge before gain calibration
  struct SeedPE {
    double seed_time;   ///< Time reported by the seed analyzer, the center of the time window
    double start_time;  ///< Starting time of the fit, spread apart from repeated seed times
    double charge;      ///< Seed charge
  };

  /// Seed PEs fit together, and the waveform samples they are fit to
  struct Cluster {
    std::vector<SeedPE> pes;
    int first_sample;
    int last_sample;
  };

  DBLinkPtr fDigit;

  std::string seed_analyzer;  ///< Analyzer whose result seeds the fit
  bool seed_missing_warned;   ///< Whether the missing seed result has already been reported

  int template_type;  ///< Template type: 0=lognormal, 1=gaussian

  // LogNormal template parameters
  double lognormal_scale;  ///< LogNormal 'm' parameter for SPE template
  double lognormal_shape;  ///< LogNormal 'sigma' parameter for SPE template

  // Gaussian template parameters
  double gaussian_width;  ///< Gaussian 'sigma' parameter for SPE template

  bool width_from_pmtpulse;  ///< Take the template shape per PMT from PMTPULSE instead of the parameters above

  std::map<std::string, TemplateShape> fModelShapeCache;  ///< PMTPULSE shape per PMT model, before per-channel scaling

  // Fit configuration
  double noise_rms;          ///< Per-sample noise RMS (mV) weighting the chi-square
  double time_window;        ///< Each PE time is limited to its seed time +/- this window (ns)
  double duplicate_spread;   ///< Spacing (ns) of the starting times of PEs sharing a seed time
  double charge_max_factor;  ///< Each PE charge is limited to [0, charge_max_factor * seed charge]
  double fit_window_pre;     ///< Samples fit before a PE's time window (ns)
  double fit_window_post;    ///< Samples fit after a PE's time window (ns)
  size_t max_cluster_pes;    ///< Clusters with more PEs than this are emitted unfit

  bool float_width;        ///< Fit a template width scale per cluster
  double width_scale_min;  ///< Lower limit of the template width scale
  double width_scale_max;  ///< Upper limit of the template width scale

  bool float_baseline;    ///< Fit a baseline offset per cluster
  double baseline_limit;  ///< The baseline offset is limited to +/- this value (mV)

  // Minimizer configuration
  std::unique_ptr<ROOT::Math::Minimizer> fMinimizer;
  int minuit_strategy;      ///< Minuit2 strategy: 0, 1 or 2
  double minuit_tolerance;  ///< Minuit2 EDM tolerance
  int minuit_max_calls;     ///< Maximum chi-square evaluations per cluster

  // State of the cluster being fit, read by Chi2() and Chi2Gradient()
  std::vector<double> fFitTimes;  ///< Sample times (ns)
  std::vector<double> fFitVolts;  ///< Sample voltages (mV)
  TemplateShape fFitShape;        ///< Template shape before the fitted width scale
  size_t fFitNPE;                 ///< Number of PEs in the cluster

  void DoAnalysis(DS::DigitPMT *digitpmt, const std::vector<UShort_t> &digitWfm) override;

  /// Template shape for one PMT: the PMTPULSE shape for its model, with the lognormal 'm' or
  /// the gaussian sigma scaled by the channel's pulse_width_scale calibration.
  TemplateShape ShapeForPMT(int pmtid);

  /// Template shape built from the configured lognormal/gaussian parameters, used when
  /// PMTPULSE is switched off or cannot describe a model.
  TemplateShape ConfiguredShape() const;

  /// PMTPULSE shape for one PMT model, falling back to the configured parameters when the
  /// table has no usable entry for it.
  TemplateShape ShapeForModel(const std::string &model_name);

  /// Unit-charge SPE template at a lag from the PE time, in 1/ns, and its derivatives with
  /// respect to the lag and to the template scale
  double Template(double lag, double scale, double &dlag, double &dscale) const;

  /// Read the seed PEs of one channel, sorted by time
  std::vector<SeedPE> ReadSeed(DS::DigitPMT *digitpmt, double gain_calibration);

  /// Group seed PEs whose fit windows overlap
  std::vector<Cluster> BuildClusters(const std::vector<SeedPE> &pes, int nsamples) const;

  /// Fit one cluster and add its PEs to the result
  void FitCluster(const Cluster &cluster, const std::vector<double> &voltWfm, const TemplateShape &shape,
                  DS::WaveformAnalysisResult *fit_result, double gain_calibration);

  /// Add a cluster's seed PEs to the result unchanged, flagged as unfit
  void EmitSeed(const Cluster &cluster, DS::WaveformAnalysisResult *fit_result, double gain_calibration,
                int fit_status);

  /// Apply the minimizer configuration to fMinimizer
  void ConfigureMinimizer();
};

}  // namespace RAT

#endif
