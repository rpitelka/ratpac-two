////////////////////////////////////////////////////////////////////
/// \class RAT::WaveformAnalysisRAVEN
///
/// \brief Perform reverse sparse non-negative least squares fitting and NPE likelihood estimation on digitized
/// waveforms
///
/// \author Ravi Carpen Pitelka <rpitelka@sas.upenn.edu>
///
/// REVISION HISTORY:\n
///     12 Sep 2025: Initial commit
///     12 Nov 2025: Added to ratpac-two
///     15 Jan 2026: Added region-based processing and NPE estimation features
///     09 Feb 2026: Renamed to RAVEN
///
/// \details
/// RAVEN (Reverse Analysis of Voltage Events with Nonegativity) is a waveform analysis algorithm
/// that performs reverse sparse non-negative least squares (rsNNLS) analysis, followed by
/// NPE likelihood estimation on digitized PMT waveforms to reconstruct photoelectron times and charges.
///
/// The algorithm uses region-based processing for improved efficiency:
/// 1. Builds a dictionary matrix of time-shifted templates
/// 2. Identifies threshold crossing regions in the waveform for localized processing
/// 3. For each region, extracts relevant dictionary submatrix and applies NNLS fitting
/// 4. Uses iterative thresholding to remove low-weight components and redistribute weights
/// 5. Extracts PE times and charges from remaining significant weights
///
/// Template types supported:
/// - Lognormal
/// - Gaussian
////////////////////////////////////////////////////////////////////
#ifndef __RAT_WaveformAnalysisRAVEN__
#define __RAT_WaveformAnalysisRAVEN__

#include <TMatrixDfwd.h>
#include <TObject.h>

#include <RAT/DB.hh>
#include <RAT/DS/DigitPMT.hh>
#include <RAT/Digitizer.hh>
#include <RAT/Processor.hh>
#include <RAT/WaveformAnalyzerBase.hh>
#include <string>
#include <utility>
#include <vector>

namespace RAT {

class WaveformAnalysisRAVEN : public WaveformAnalyzerBase {
 public:
  WaveformAnalysisRAVEN() : WaveformAnalysisRAVEN("RAVEN"){};

  WaveformAnalysisRAVEN(std::string config_name) : WaveformAnalyzerBase("WaveformAnalysisRAVEN", config_name) {
    Configure(config_name);
  };

  virtual ~WaveformAnalysisRAVEN(){};

  void BuildDictionaryMatrix(int nsamples, double digitizer_period);

  void Configure(const std::string &config_name) override;

  void SetD(std::string param, double value) override;
  void SetI(std::string param, int value) override;

 protected:
  DBLinkPtr fDigit;

  bool process_threshold_crossing;  ///< Whether to use threshold crossing region processing
  double voltage_threshold;         ///< Voltage threshold for threshold crossing region detection
  int threshold_region_padding;     ///< Number of samples to pad around threshold crossing regions

  int template_type;  ///< Template type: 0=lognormal, 1=gaussian

  // LogNormal template parameters
  double lognormal_scale;  ///< LogNormal 'm' parameter for SPE template
  double lognormal_shape;  ///< LogNormal 'sigma' parameter for SPE template

  // Gaussian template parameters
  double gaussian_width;  ///< Gaussian 'sigma' parameter for SPE template

  double vpe_charge;  ///< Nominal charge of single PE in pC

  // Algorithm configuration
  TMatrixD fW;             ///< Dictionary matrix for NNLS (nsamples × dict_size)
  double epsilon;          ///< NNLS convergence tolerance
  size_t max_iterations;   ///< Maximum iterations for iterative thresholding
  double upsample_factor;  ///< Dictionary upsampling factor for sub-sample resolution

  // Thresholding parameters
  double weight_threshold;     ///< Minimum weight threshold for component significance
  double weight_merge_window;  ///< Time window (ns) for merging nearby weights before NPE estimation

  // NPE estimation parameters
  bool npe_estimate;                 ///< Whether to perform NPE estimation on resolved wave packets
  double npe_estimate_charge_width;  ///< Width of Gaussian single-PE charge distribution
  size_t npe_estimate_max_pes;       ///< Upper limit for NPE estimation

  // Dictionary management
  bool dictionary_built;           ///< Flag to track if dictionary has been built
  int cached_nsamples;             ///< Cached number of samples for dictionary
  double cached_digitizer_period;  ///< Cached digitizer period for dictionary

  // Global fit configuration
  bool global_fit_enabled;             ///< Enable NLOPT global fit after rsNNLS
  std::string global_fit_algo;         ///< NLOPT algorithm name (gradient-free only)
  double global_fit_time_window;       ///< Half-width (ns) of time float window per PE
  double global_fit_weight_max_scale;  ///< Upper bound on weight = initial * this factor
  int global_fit_max_evals;            ///< Max NLOPT function evaluations per region
  double global_fit_tolerance;         ///< NLOPT xtol_rel convergence criterion
  bool global_fit_float_npe;           ///< Try N-1 and N+1 PE configurations

  void DoAnalysis(DS::DigitPMT *digitpmt, const std::vector<UShort_t> &digitWfm) override;

  /// Perform reverse sparse NNLS with iterative thresholding on a region submatrix
  TVectorD Thresholded_rsNNLS(const TMatrixD &W_region, const TVectorD &voltVec, const double threshold,
                              double &chi2ndf_out, int &iterations_out);

  /// Find threshold crossing regions in waveform for efficient processing
  std::vector<std::pair<int, int>> FindThresholdRegions(const std::vector<double> &voltWfm, double threshold,
                                                        int region_padding);

  /// Process a single threshold crossing region with rsNNLS
  void ProcessThresholdRegion(const std::vector<double> &voltWfm, int start_sample, int end_sample,
                              DS::WaveformAnalysisResult *fit_result, double gain_calibration);

  /// Extract photoelectrons from significant weights in the region.
  /// Calls MergeNearbyWeights internally then delegates to ExtractPhotoelectronsFromMerged.
  void ExtractPhotoelectrons(const TVectorD &region_weights, int dict_start, int dict_cols, int start_sample,
                             int end_sample, double chi2ndf, int iterations_ran, DS::WaveformAnalysisResult *fit_result,
                             double gain_calibration);

  /// Extract photoelectrons from pre-merged (time_ns, weight) pairs.
  void ExtractPhotoelectronsFromMerged(const std::vector<std::pair<double, double>> &merged_weights, int start_sample,
                                       int end_sample, double chi2ndf, int iterations_ran, int global_fit_npe_delta,
                                       DS::WaveformAnalysisResult *fit_result, double gain_calibration);

  /// Merge nearby weights within a time window to prevent PE overcounting
  /// Returns vector of (time, merged_weight) pairs
  std::vector<std::pair<double, double>> MergeNearbyWeights(const TVectorD &region_weights, int dict_start,
                                                            int dict_cols, double merge_window);

  /// Evaluate the SPE template at sample_time_ns for a PE arriving at delay_ns.
  /// Returns the signed voltage contribution (negative, matching dictionary convention).
  double EvaluateTemplate(double sample_time_ns, double delay_ns) const;

  /// Compute chi2/ndf between region_vec and the model reconstructed from weights.
  double ComputeModelChi2(const TVectorD &region_vec, const std::vector<std::pair<double, double>> &weights,
                          int start_sample) const;

  /// Run NLOPT global fit to refine (time, weight) pairs against the observed region voltage.
  /// Updates chi2ndf_out with the post-fit chi2/ndf. Returns the refined weights.
  std::vector<std::pair<double, double>> GlobalFitWeights(const TVectorD &region_vec,
                                                          const std::vector<std::pair<double, double>> &initial_weights,
                                                          int start_sample, double &chi2ndf_out, int &npe_delta_out);
};

}  // namespace RAT

#endif
