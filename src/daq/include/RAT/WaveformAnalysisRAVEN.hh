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
#include <map>
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

  /// Build a dictionary of time-shifted templates into W_out. For the gaussian
  /// template, `width` is the SER sigma to use (per-PMT-type selectable).
  void BuildDictionaryMatrix(int nsamples, double digitizer_period, double width, TMatrixD &W_out);

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

  // Optional per-PMT-type gaussian widths. The true SER width differs between
  // PMT models (e.g. Eos 8" r14688 ~1.6 ns vs 12" r11780 ~3.0 ns); a single
  // template width mis-fits the other models and produces satellite
  // components. PMT types listed here use the paired width; others fall back
  // to gaussian_width.
  std::vector<int> gaussian_width_types;      ///< PMT types with a dedicated template width
  std::vector<double> gaussian_width_values;  ///< Template width (ns) per listed PMT type

  double vpe_charge;  ///< Nominal charge of single PE in pC

  // Algorithm configuration
  std::map<int, TMatrixD> fWCache;  ///< Dictionary per template (key: width in ps; -1 = lognormal)
  double epsilon;                   ///< NNLS convergence tolerance
  size_t max_iterations;            ///< Maximum iterations for iterative thresholding
  double upsample_factor;           ///< Dictionary upsampling factor for sub-sample resolution

  // Thresholding parameters
  double weight_threshold;     ///< Minimum weight threshold for component significance
  double weight_merge_window;  ///< Time window (ns) for merging nearby weights before NPE estimation

  // Noise-scaled NNLS stopping (optional). When both are > 0, the NNLS
  // optimality tolerance becomes nnls_noise_nsigma * noise_sigma * max column
  // norm of the region dictionary — the gradient level expected from pure
  // noise — instead of the fixed nnls_tolerance. This stops the solver from
  // fitting noise fluctuations with additional low-weight components.
  double noise_sigma;        ///< Gaussian white-noise sigma of the waveform (mV); 0 disables
  double nnls_noise_nsigma;  ///< Number of noise sigmas for the NNLS stopping level

  // Position refinement. Reverse pursuit can only remove components, so a
  // component misplaced by the initial NNLS solve (typically ~1 sample early,
  // on the steep leading edge) is otherwise locked in. When enabled, each
  // remaining component is tentatively moved to nearby free dictionary columns
  // and the move is kept if it lowers the fit residual.
  bool refine_positions;  ///< Enable post-pruning position refinement

  // NPE estimation parameters
  bool npe_estimate;                 ///< Whether to perform NPE estimation on resolved wave packets
  double npe_estimate_charge_width;  ///< Width of Gaussian single-PE charge distribution
  size_t npe_estimate_max_pes;       ///< Upper limit for NPE estimation

  // Dictionary management
  int cached_nsamples;             ///< Cached number of samples for dictionary
  double cached_digitizer_period;  ///< Cached digitizer period for dictionary

  void DoAnalysis(DS::DigitPMT *digitpmt, const std::vector<UShort_t> &digitWfm) override;

  /// Perform reverse sparse NNLS with iterative thresholding on a region submatrix
  TVectorD Thresholded_rsNNLS(const TMatrixD &W_region, const TVectorD &voltVec, const double threshold,
                              double &chi2ndf_out, int &iterations_out);

  /// Find threshold crossing regions in waveform for efficient processing
  std::vector<std::pair<int, int>> FindThresholdRegions(const std::vector<double> &voltWfm, double threshold,
                                                        int region_padding);

  /// Process a single threshold crossing region with rsNNLS
  void ProcessThresholdRegion(const TMatrixD &fW, const std::vector<double> &voltWfm, int start_sample, int end_sample,
                              DS::WaveformAnalysisResult *fit_result, double gain_calibration);

  /// Extract photoelectrons from significant weights in the region
  void ExtractPhotoelectrons(const TVectorD &region_weights, int dict_start, int dict_cols, int start_sample,
                             int end_sample, double chi2ndf, int iterations_ran, DS::WaveformAnalysisResult *fit_result,
                             double gain_calibration);

  /// Merge nearby weights within a time window to prevent PE overcounting
  /// Returns vector of (time, merged_weight) pairs
  std::vector<std::pair<double, double>> MergeNearbyWeights(const TVectorD &region_weights, int dict_start,
                                                            int dict_cols, double merge_window);
};

}  // namespace RAT

#endif
