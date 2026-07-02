#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>

#include <RAT/DS/RunStore.hh>
#include <RAT/Log.hh>
#include <RAT/NPEEstimator.hh>
#include <RAT/WaveformAnalysisRAVEN.hh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/NNLS.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {

void WaveformAnalysisRAVEN::Configure(const std::string& config_name) {
  debug << "WaveformAnalysisRAVEN: Configure called with config_name " << config_name << newline;
  // Load analysis parameters from DIGITIZER_ANALYSIS database
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);

    // Threshold crossing region configuration
    process_threshold_crossing = (fDigit->GetI("process_threshold_crossing") != 0);  // 0=no, 1=yes
    if (process_threshold_crossing) {
      voltage_threshold = fDigit->GetD("voltage_threshold");      // Voltage threshold in mV
      threshold_region_padding = fDigit->GetI("region_padding");  // Padding samples around threshold crossing
    }

    // Template type configuration
    template_type = fDigit->GetI("raven_template_type");  // 0=lognormal, 1=gaussian

    // Single photoelectron waveform parameters
    if (template_type == 0) {                             // lognormal
      lognormal_scale = fDigit->GetD("lognormal_scale");  // LogNormal 'm' parameter
      lognormal_shape = fDigit->GetD("lognormal_shape");  // LogNormal 'sigma' parameter
    } else if (template_type == 1) {                      // gaussian
      gaussian_width = fDigit->GetD("gaussian_width");    // Gaussian 'sigma' parameter
      // Optional per-PMT-type template widths (paired arrays); unlisted PMT
      // types fall back to gaussian_width.
      gaussian_width_types.clear();
      gaussian_width_values.clear();
      try {
        gaussian_width_types = fDigit->GetIArray("gaussian_width_pmt_types");
        gaussian_width_values = fDigit->GetDArray("gaussian_width_pmt_widths");
      } catch (DBNotFoundError) {
      }
      if (gaussian_width_types.size() != gaussian_width_values.size()) {
        RAT::Log::Die("WaveformAnalysisRAVEN: gaussian_width_pmt_types/widths must have equal length.");
      }
    } else {
      RAT::Log::Die("WaveformAnalysisRAVEN: Invalid template_type " + std::to_string(template_type) +
                    ". Must be 0 (lognormal) or 1 (gaussian).");
    }

    vpe_charge = fDigit->GetD("vpe_charge");  // Nominal PE charge in pC

    // Algorithm configuration
    max_iterations = fDigit->GetI("max_iterations");      // Max thresholding iterations
    weight_threshold = fDigit->GetD("weight_threshold");  // Component significance threshold
    upsample_factor = fDigit->GetD("upsampling_factor");  // Dictionary upsampling factor
    epsilon = fDigit->GetD("nnls_tolerance");             // NNLS convergence tolerance

    // NPE estimation configuration
    npe_estimate = fDigit->GetZ("npe_estimate");
    npe_estimate_charge_width = fDigit->GetD("npe_estimate_charge_width");
    npe_estimate_max_pes = fDigit->GetI("npe_estimate_max_pes");

    // Weight merging configuration
    weight_merge_window = fDigit->GetD("weight_merge_window");  // Time window for merging nearby weights (ns)

    // Optional parameters (absent from older tables): noise-scaled NNLS
    // stopping level and post-pruning position refinement.
    noise_sigma = 0.0;
    nnls_noise_nsigma = 3.0;
    refine_positions = false;
    try {
      noise_sigma = fDigit->GetD("noise_sigma");
    } catch (DBNotFoundError) {
    }
    try {
      nnls_noise_nsigma = fDigit->GetD("nnls_noise_nsigma");
    } catch (DBNotFoundError) {
    }
    try {
      refine_positions = fDigit->GetZ("refine_positions");
    } catch (DBNotFoundError) {
    }

    // Validate critical parameters
    if (upsample_factor <= 0) {
      RAT::Log::Die("WaveformAnalysisRAVEN: Invalid upsampling factor.");
    }

    // Initialize dictionary cache
    fWCache.clear();
    cached_nsamples = -1;            // Invalid initial value to force dictionary build on first use
    cached_digitizer_period = -1.0;  // Invalid initial value to force dictionary build on first use

  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Unable to find analysis parameters.");
  }
}

void WaveformAnalysisRAVEN::SetD(std::string param, double value) {
  if (param == "lognormal_scale") {
    lognormal_scale = value;
    fWCache.clear();
  } else if (param == "lognormal_shape") {
    lognormal_shape = value;
    fWCache.clear();
  } else if (param == "gaussian_width") {
    gaussian_width = value;
  } else if (param == "vpe_charge") {
    vpe_charge = value;
    fWCache.clear();
  } else if (param == "upsampling_factor") {
    upsample_factor = value;
    fWCache.clear();
  } else if (param == "weight_threshold") {
    weight_threshold = value;
  } else if (param == "voltage_threshold") {
    voltage_threshold = value;
  } else if (param == "nnls_tolerance") {
    epsilon = value;
  } else if (param == "npe_estimate_charge_width") {
    npe_estimate_charge_width = value;
  } else if (param == "weight_merge_window") {
    weight_merge_window = value;
  } else if (param == "noise_sigma") {
    noise_sigma = value;
  } else if (param == "nnls_noise_nsigma") {
    nnls_noise_nsigma = value;
  } else {
    WaveformAnalyzerBase::SetD(param, value);
  }
}

void WaveformAnalysisRAVEN::SetI(std::string param, int value) {
  if (param == "process_threshold_crossing") {
    process_threshold_crossing = (value != 0);
  } else if (param == "max_iterations") {
    max_iterations = value;
  } else if (param == "raven_template_type") {
    template_type = value;
    if (template_type != 0 && template_type != 1) {
      RAT::Log::Die("WaveformAnalysisRAVEN: Invalid raven_template_type " + std::to_string(value) +
                    ". Must be 0 (lognormal) or 1 (gaussian).");
    }
    fWCache.clear();
  } else if (param == "npe_estimate") {
    npe_estimate = (value != 0);
  } else if (param == "npe_estimate_max_pes") {
    npe_estimate_max_pes = static_cast<size_t>(value);
  } else if (param == "refine_positions") {
    refine_positions = (value != 0);
  } else {
    throw Processor::ParamUnknown(param);
  }
}

void WaveformAnalysisRAVEN::BuildDictionaryMatrix(int nsamples, double digitizer_period, double width,
                                                  TMatrixD& W_out) {
  debug << "WaveformAnalysisRAVEN: Building dictionary matrix - nsamples: " << nsamples
        << ", period: " << digitizer_period << newline;
  debug << "WaveformAnalysisRAVEN: Using raven_template_type: " << template_type << " ("
        << (template_type == 0 ? "lognormal" : "gaussian") << "), width " << width << newline;
  debug << "WaveformAnalysisRAVEN: Dictionary size: " << nsamples << " x "
        << static_cast<int>(nsamples * upsample_factor) << newline;

  const int dict_size = static_cast<int>(nsamples * upsample_factor);
  W_out.ResizeTo(nsamples, dict_size);
  W_out.Zero();

  const double mag_factor = vpe_charge * fTermOhms;

  // Generate dictionary with time-shifted templates
  for (int col = 0; col < dict_size; ++col) {
    double delay = col * digitizer_period / upsample_factor;

    for (int row = 0; row < nsamples; ++row) {
      double sample_time = row * digitizer_period;
      double template_val = 0.0;

      if (template_type == 0) {  // lognormal
        double lognormal_shift = delay - lognormal_scale;
        if (sample_time > lognormal_shift) {
          template_val = mag_factor * TMath::LogNormal(sample_time, lognormal_shape, lognormal_shift, lognormal_scale);
        }
      } else if (template_type == 1) {  // gaussian
        template_val = mag_factor * TMath::Gaus(sample_time, delay, width, kTRUE);
      }

      W_out(row, col) = -template_val;
    }
  }
}

void WaveformAnalysisRAVEN::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  // Invalidate the dictionary cache when digitizer parameters change
  const double period_tolerance = 1e-9;  // 1 ps tolerance for digitizer period comparison
  if (cached_nsamples != static_cast<int>(digitWfm.size()) ||
      std::abs(cached_digitizer_period - fTimeStep) > period_tolerance) {
    fWCache.clear();
    cached_nsamples = static_cast<int>(digitWfm.size());
    cached_digitizer_period = fTimeStep;
  }

  // Pick the template width for this PMT's type (gaussian template only) and
  // fetch/build the matching dictionary.
  double width = gaussian_width;
  if (template_type == 1 && !gaussian_width_types.empty()) {
    const int pmt_type = DS::RunStore::GetCurrentRun()->GetPMTInfo()->GetType(digitpmt->GetID());
    for (size_t i = 0; i < gaussian_width_types.size(); ++i) {
      if (gaussian_width_types[i] == pmt_type) {
        width = gaussian_width_values[i];
        break;
      }
    }
  }
  const int cache_key = (template_type == 0) ? -1 : static_cast<int>(std::lround(width * 1000.0));
  auto cache_it = fWCache.find(cache_key);
  if (cache_it == fWCache.end()) {
    cache_it = fWCache.emplace(cache_key, TMatrixD()).first;
    BuildDictionaryMatrix(cached_nsamples, cached_digitizer_period, width, cache_it->second);
  }
  const TMatrixD& fW = cache_it->second;

  double pedestal = digitpmt->GetPedestal();
  if (pedestal == -9999) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Pedestal is invalid! Did you run WaveformPrep first?");
  }

  // Get per-PMT gain calibration for consistent charge calculation (same as LucyDDM)
  double gain_calibration = DS::RunStore::GetCurrentRun()->GetChannelStatus()->GetChargeScaleByPMTID(digitpmt->GetID());

  // Verify waveform size matches dictionary matrix
  if (static_cast<int>(digitWfm.size()) != fW.GetNrows()) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Waveform size mismatch with dictionary matrix.");
  }

  std::vector<double> voltWfm = WaveformUtil::ADCtoVoltage(digitWfm, fVoltageRes, pedestal);

  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult(GetAnalyzerName());

  if (process_threshold_crossing) {
    // Find threshold crossing regions
    std::vector<std::pair<int, int>> crossing_regions =
        FindThresholdRegions(voltWfm, voltage_threshold, threshold_region_padding);

    if (crossing_regions.empty()) {
      // No signal above threshold - return empty result
      return;
    }

    // Process each threshold crossing region independently

    for (const auto& region : crossing_regions) {
      int start_sample = region.first;
      int end_sample = region.second;

      // Perform rsNNLS on this region
      ProcessThresholdRegion(fW, voltWfm, start_sample, end_sample, fit_result, gain_calibration);
    }
  } else {
    int start_sample = 0;
    int end_sample = static_cast<int>(voltWfm.size()) - 1;

    // Perform rsNNLS on the entire waveform
    ProcessThresholdRegion(fW, voltWfm, start_sample, end_sample, fit_result, gain_calibration);
  }
}

TVectorD WaveformAnalysisRAVEN::Thresholded_rsNNLS(const TMatrixD& W_region, const TVectorD& voltVec,
                                                   const double threshold, double& chi2ndf_out, int& iterations_out) {
  const int D = voltVec.GetNrows();
  const int K = W_region.GetNcols();

  if (W_region.GetNrows() != D) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Dictionary region row dimension mismatch.");
  }

  // NNLS stopping tolerance. The optimality test compares the gradient
  // max_j A_j^T r against this value; on a pure-noise residual that gradient
  // fluctuates at the level noise_sigma * ||A_j||, so stopping below it just
  // fits noise with extra low-weight components (and costs many extra solver
  // iterations). When noise_sigma is configured, stop at nnls_noise_nsigma
  // noise sigmas; otherwise fall back to the fixed nnls_tolerance.
  double tol = epsilon;
  if (noise_sigma > 0.0 && nnls_noise_nsigma > 0.0) {
    double max_col_norm2 = 0.0;
    for (int j = 0; j < K; ++j) {
      double norm2 = 0.0;
      for (int i = 0; i < D; ++i) norm2 += W_region(i, j) * W_region(i, j);
      max_col_norm2 = std::max(max_col_norm2, norm2);
    }
    tol = std::max(tol, nnls_noise_nsigma * noise_sigma * std::sqrt(max_col_norm2));
  }

  // Initial NNLS solve
  TVectorD h_full(K);
  h_full.Zero();
  h_full = Math::NNLS_LawsonHanson(W_region, voltVec, tol, 0, 0);

  // Build initial active set
  std::vector<int> P;
  P.reserve(K);
  for (int j = 0; j < K; ++j) {
    if (h_full(j) > 0.0) P.push_back(j);
  }

  // Helper lambda to extract dictionary submatrix for active components
  auto subCols = [](const TMatrixD& W, const std::vector<int>& cols) {
    TMatrixD S(W.GetNrows(), cols.size());
    for (size_t jj = 0; jj < cols.size(); ++jj) {
      int c = cols[jj];
      for (int i = 0; i < W.GetNrows(); ++i) S(i, jj) = W(i, c);
    }
    return S;
  };

  // Iterative thresholding
  int local_iterations_ran = 0;

  for (size_t iter = 0; iter < max_iterations && !P.empty(); ++iter) {
    local_iterations_ran = static_cast<int>(iter + 1);

    // Find component with minimum weight
    std::vector<int>::iterator minIt =
        std::min_element(P.begin(), P.end(), [&h_full](int a, int b) { return h_full(a) < h_full(b); });
    size_t minPos = std::distance(P.begin(), minIt);
    double minVal = h_full(*minIt);

    if (minVal >= threshold) break;

    // Never prune the last remaining component — always fit at least one PE per threshold crossing
    if (P.size() == 1) break;

    // Remove component with smallest weight
    h_full(P[minPos]) = 0.0;
    P.erase(P.begin() + minPos);

    // Re-solve on reduced active set
    TMatrixD W_P = subCols(W_region, P);
    TVectorD h_reduced(P.size());
    h_reduced.Zero();
    h_reduced = Math::NNLS_LawsonHanson(W_P, voltVec, tol, 0, 0);

    // Update full weight vector
    h_full.Zero();
    for (size_t k = 0; k < P.size(); ++k) {
      h_full(P[k]) = h_reduced(k);
    }
  }

  // Position refinement. The reverse pursuit above can only remove components,
  // so a component the initial solve misplaced (typically ~1 sample early, on
  // the steep leading edge of a pulse) stays misplaced and shows up as an early
  // ghost PE. For each remaining component, score nearby free columns with the
  // others' weights held fixed (a cheap 1-D projection), and if a neighbor
  // fits better, move the component there and re-solve; keep the move if the
  // total residual decreases.
  if (refine_positions && !P.empty()) {
    const int max_shift = std::max(1, static_cast<int>(std::lround(upsample_factor)));  // +- 1 sample
    auto residualOf = [&](const TVectorD& h) {
      TVectorD r = voltVec;
      for (int j = 0; j < K; ++j) {
        if (h(j) == 0.0) continue;
        for (int i = 0; i < D; ++i) r(i) -= W_region(i, j) * h(j);
      }
      return r;
    };
    TVectorD r_full = residualOf(h_full);
    double cur_rss = r_full * r_full;
    std::vector<char> occupied(K, 0);
    for (int c : P) occupied[c] = 1;

    for (int sweep = 0; sweep < 2; ++sweep) {
      bool improved = false;
      for (size_t idx = 0; idx < P.size(); ++idx) {
        const int c = P[idx];
        // Residual with this component removed, other weights fixed.
        TVectorD r_wo = r_full;
        for (int i = 0; i < D; ++i) r_wo(i) += W_region(i, c) * h_full(c);
        // 1-D score of each nearby free column: rss after optimally (re)fitting
        // a single nonnegative weight on that column.
        int best_col = c;
        double best_1d = std::numeric_limits<double>::max();
        for (int dc = -max_shift; dc <= max_shift; ++dc) {
          const int cp = c + dc;
          if (cp < 0 || cp >= K) continue;
          if (cp != c && occupied[cp]) continue;
          double dot = 0.0, norm2 = 0.0;
          for (int i = 0; i < D; ++i) {
            dot += W_region(i, cp) * r_wo(i);
            norm2 += W_region(i, cp) * W_region(i, cp);
          }
          const double gain = (dot > 0.0 && norm2 > 0.0) ? dot * dot / norm2 : 0.0;
          const double rss_1d = (r_wo * r_wo) - gain;
          if (rss_1d < best_1d - 1e-12) {
            best_1d = rss_1d;
            best_col = cp;
          }
        }
        if (best_col == c) continue;

        // Full re-solve with the component moved; accept only on improvement.
        std::vector<int> P_trial = P;
        P_trial[idx] = best_col;
        TMatrixD W_P = subCols(W_region, P_trial);
        TVectorD h_trial = Math::NNLS_LawsonHanson(W_P, voltVec, tol, 0, 0);
        TVectorD h_new(K);
        h_new.Zero();
        for (size_t k = 0; k < P_trial.size(); ++k) h_new(P_trial[k]) = h_trial(static_cast<int>(k));
        TVectorD r_new = residualOf(h_new);
        const double new_rss = r_new * r_new;
        if (new_rss < cur_rss - 1e-9) {
          occupied[c] = 0;
          occupied[best_col] = 1;
          P[idx] = best_col;
          h_full = h_new;
          r_full = r_new;
          cur_rss = new_rss;
          improved = true;
        }
      }
      if (!improved) break;
    }
  }

  // Ensure numerical stability
  for (int j = 0; j < K; ++j) {
    if (h_full(j) < 0.0) h_full(j) = 0.0;
  }

  // Calculate chi-squared goodness of fit
  TVectorD fitted = W_region * h_full;
  double chi2_sum = 0.0;
  for (int i = 0; i < D; ++i) {
    double residual = voltVec(i) - fitted(i);
    chi2_sum += residual * residual;
  }

  int active_components = 0;
  for (int j = 0; j < K; ++j) {
    if (h_full(j) > 0.0) active_components++;
  }
  int dof = std::max(1, D - active_components);
  chi2ndf_out = chi2_sum / dof;
  iterations_out = local_iterations_ran;

  return h_full;
}

std::vector<std::pair<int, int>> WaveformAnalysisRAVEN::FindThresholdRegions(const std::vector<double>& voltWfm,
                                                                             double threshold, int region_padding) {
  std::vector<std::pair<int, int>> regions;
  bool in_region = false;
  int region_start = -1;

  for (size_t i = 0; i < voltWfm.size(); ++i) {
    // Falling edge detection - signal goes below threshold
    if (voltWfm[i] < threshold && !in_region) {
      region_start = std::max(0, static_cast<int>(i) - region_padding);
      in_region = true;
    }
    // Rising edge detection - signal goes above threshold
    else if (voltWfm[i] >= threshold && in_region) {
      int region_end = std::min(static_cast<int>(voltWfm.size()) - 1, static_cast<int>(i) + region_padding - 1);
      regions.emplace_back(region_start, region_end);
      in_region = false;
    }
  }

  if (in_region) {
    regions.emplace_back(region_start, static_cast<int>(voltWfm.size()) - 1);
  }

  // Merge close regions
  if (regions.size() > 1) {
    std::vector<std::pair<int, int>> merged_regions;
    merged_regions.push_back(regions[0]);

    for (size_t i = 1; i < regions.size(); ++i) {
      if (regions[i].first <= merged_regions.back().second + region_padding) {
        merged_regions.back().second = regions[i].second;
      } else {
        merged_regions.push_back(regions[i]);
      }
    }
    std::swap(regions, merged_regions);
  }

  return regions;
}

void WaveformAnalysisRAVEN::ProcessThresholdRegion(const TMatrixD& fW, const std::vector<double>& voltWfm,
                                                   int start_sample, int end_sample,
                                                   DS::WaveformAnalysisResult* fit_result, double gain_calibration) {
  const int region_length = end_sample - start_sample + 1;

  // Use iterators to avoid copying waveform segment
  std::vector<double>::const_iterator region_begin = voltWfm.begin() + start_sample;

  // Calculate dictionary column range directly from sample indices
  // Dictionary column j corresponds to sample time j/upsample_factor
  // We want columns that correspond to this region's sample range

  const int dict_start = std::max(0, static_cast<int>(start_sample * upsample_factor));
  const int dict_end = std::min(fW.GetNcols() - 1, static_cast<int>(end_sample * upsample_factor));
  const int dict_cols = dict_end - dict_start + 1;

  if (dict_cols <= 0) {
    return;
  }

  // Extract relevant dictionary submatrix
  TMatrixD W_region(region_length, dict_cols);
  W_region.Zero();

  for (int row = 0; row < region_length; ++row) {
    int global_row = start_sample + row;
    if (global_row >= fW.GetNrows()) continue;

    for (int col = 0; col < dict_cols; ++col) {
      int global_col = dict_start + col;
      if (global_col >= 0 && global_col < fW.GetNcols()) {
        W_region(row, col) = fW(global_row, global_col);
      }
    }
  }

  // Convert region waveform to TVectorD using iterators
  TVectorD region_vec(region_length);
  std::vector<double>::const_iterator it = region_begin;
  for (int i = 0; i < region_length; ++i, ++it) {
    region_vec(i) = *it;
  }

  // Perform rsNNLS on this region
  double chi2ndf;
  int iterations_ran;
  TVectorD region_weights = Thresholded_rsNNLS(W_region, region_vec, weight_threshold, chi2ndf, iterations_ran);

  // Extract PEs from significant weights
  ExtractPhotoelectrons(region_weights, dict_start, dict_cols, start_sample, end_sample, chi2ndf, iterations_ran,
                        fit_result, gain_calibration);
}

std::vector<std::pair<double, double>> WaveformAnalysisRAVEN::MergeNearbyWeights(const TVectorD& region_weights,
                                                                                 int dict_start, int dict_cols,
                                                                                 double merge_window) {
  // Collect non-zero weights with their times
  std::vector<std::pair<double, double>> time_weight_pairs;  // (time, weight)
  for (int i = 0; i < dict_cols; ++i) {
    if (region_weights(i) > 0.0) {
      int global_dict_index = dict_start + i;
      double time = global_dict_index * fTimeStep / upsample_factor;
      time_weight_pairs.emplace_back(time, region_weights(i));
    }
  }

  if (time_weight_pairs.empty() || merge_window <= 0.0) {
    return time_weight_pairs;
  }

  // Dominant-atom merge: seed clusters at the largest remaining weight and
  // absorb all unassigned weights within merge_window of the *seed* time. The
  // previous scheme seeded clusters at the earliest weight, so a small
  // leading-edge ghost component anchored the cluster and dragged the merged
  // time early; anchoring at the dominant component folds such ghosts into the
  // main pulse instead.
  std::vector<size_t> order(time_weight_pairs.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return time_weight_pairs[a].second > time_weight_pairs[b].second; });

  std::vector<char> assigned(time_weight_pairs.size(), 0);
  std::vector<std::pair<double, double>> merged_weights;
  merged_weights.reserve(time_weight_pairs.size());

  for (size_t seed : order) {
    if (assigned[seed]) continue;
    const double seed_time = time_weight_pairs[seed].first;
    double cluster_weight_sum = 0.0;
    double cluster_time_weighted_sum = 0.0;
    for (size_t j = 0; j < time_weight_pairs.size(); ++j) {
      if (assigned[j]) continue;
      if (std::abs(time_weight_pairs[j].first - seed_time) <= merge_window) {
        assigned[j] = 1;
        cluster_weight_sum += time_weight_pairs[j].second;
        cluster_time_weighted_sum += time_weight_pairs[j].first * time_weight_pairs[j].second;
      }
    }
    merged_weights.emplace_back(cluster_time_weighted_sum / cluster_weight_sum, cluster_weight_sum);
  }

  // Restore time ordering for downstream consumers.
  std::sort(merged_weights.begin(), merged_weights.end());

  return merged_weights;
}

void WaveformAnalysisRAVEN::ExtractPhotoelectrons(const TVectorD& region_weights, int dict_start, int dict_cols,
                                                  int start_sample, int end_sample, double chi2ndf, int iterations_ran,
                                                  DS::WaveformAnalysisResult* fit_result, double gain_calibration) {
  // Merge nearby weights to prevent PE overcounting from weight splitting
  std::vector<std::pair<double, double>> merged_weights =
      MergeNearbyWeights(region_weights, dict_start, dict_cols, weight_merge_window);

  // Sanity check parameters
  double template_scale = (template_type == 0) ? lognormal_scale : gaussian_width;
  double region_start_time = start_sample * fTimeStep;
  double region_end_time = end_sample * fTimeStep;

  // Extract PEs from merged weights
  for (const auto& [delay, weight] : merged_weights) {
    // Sanity check - ensure PE time is within expected range
    if (delay < region_start_time - 3.0 * template_scale || delay > region_end_time + 3.0 * template_scale) {
      warn << "WaveformAnalysisRAVEN: PE time " << delay << " ns outside expected range ["
           << (region_start_time - 3.0 * template_scale) << ", " << (region_end_time + 3.0 * template_scale)
           << "] for region [" << start_sample << ", " << end_sample << "]" << newline;
      continue;
    }

    // Charge calculation: apply gain calibration
    double pe_charge = weight * vpe_charge * gain_calibration;  // Charge in pC

    // Estimate number of PEs using likelihood method
    // Use calibrated vpe_charge so expected charge per PE matches the scale of pe_charge
    double calibrated_vpe_charge = vpe_charge * gain_calibration;
    size_t npe = npe_estimate
                     ? EstimateNPE(pe_charge, calibrated_vpe_charge, npe_estimate_charge_width, npe_estimate_max_pes)
                     : 1;

    // Add each estimated PE with divided charge
    for (size_t ipe = 0; ipe < npe; ++ipe) {
      fit_result->AddPE(delay, pe_charge / npe,
                        {
                            {"chi2ndf", chi2ndf},
                            {"iterations_ran", static_cast<double>(iterations_ran)},
                            {"weight", weight / npe},
                            {"estimated_npe", static_cast<double>(npe)},
                        });
    }
  }
}
}  // namespace RAT