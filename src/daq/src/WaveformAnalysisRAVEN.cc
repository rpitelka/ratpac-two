#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>

#include <RAT/Config.hh>
#include <RAT/DS/RunStore.hh>
#include <RAT/Log.hh>
#include <RAT/NPEEstimator.hh>
#include <RAT/WaveformAnalysisRAVEN.hh>
#include <algorithm>
#include <cmath>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/NNLS.hh"
#include "RAT/WaveformUtil.hh"

#if NLOPT_Enabled
#include <nlopt.hpp>
#endif

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

    // Validate critical parameters
    if (upsample_factor <= 0) {
      RAT::Log::Die("WaveformAnalysisRAVEN: Invalid upsampling factor.");
    }

    // Global fit configuration
    global_fit_enabled = fDigit->GetZ("global_fit_enabled");
    if (global_fit_enabled) {
      global_fit_algo = fDigit->GetS("global_fit_algo");
      global_fit_time_window = fDigit->GetD("global_fit_time_window");
      global_fit_weight_max_scale = fDigit->GetD("global_fit_weight_max_scale");
      global_fit_max_evals = fDigit->GetI("global_fit_max_evals");
      global_fit_tolerance = fDigit->GetD("global_fit_tolerance");
      global_fit_float_npe = fDigit->GetZ("global_fit_float_npe");
#if !NLOPT_Enabled
      warn << "WaveformAnalysisRAVEN: global_fit_enabled=true but NLOPT not available at build time. "
              "Global fit disabled."
           << newline;
      global_fit_enabled = false;
#endif
    }

    // Initialize dictionary flags
    dictionary_built = false;
    cached_nsamples = -1;            // Invalid initial value to force dictionary build on first use
    cached_digitizer_period = -1.0;  // Invalid initial value to force dictionary build on first use

  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Unable to find analysis parameters.");
  }
}

void WaveformAnalysisRAVEN::SetD(std::string param, double value) {
  if (param == "lognormal_scale") {
    lognormal_scale = value;
  } else if (param == "lognormal_shape") {
    lognormal_shape = value;
  } else if (param == "gaussian_width") {
    gaussian_width = value;
  } else if (param == "vpe_charge") {
    vpe_charge = value;
  } else if (param == "upsampling_factor") {
    upsample_factor = value;
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
  } else if (param == "global_fit_time_window") {
    global_fit_time_window = value;
  } else if (param == "global_fit_weight_max_scale") {
    global_fit_weight_max_scale = value;
  } else if (param == "global_fit_tolerance") {
    global_fit_tolerance = value;
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
  } else if (param == "npe_estimate") {
    npe_estimate = (value != 0);
  } else if (param == "npe_estimate_max_pes") {
    npe_estimate_max_pes = static_cast<size_t>(value);
  } else if (param == "global_fit_enabled") {
    global_fit_enabled = (value != 0);
  } else if (param == "global_fit_max_evals") {
    global_fit_max_evals = value;
  } else if (param == "global_fit_float_npe") {
    global_fit_float_npe = (value != 0);
  } else {
    throw Processor::ParamUnknown(param);
  }
}

void WaveformAnalysisRAVEN::BuildDictionaryMatrix(int nsamples, double digitizer_period) {
  debug << "WaveformAnalysisRAVEN: Building dictionary matrix" << newline;
  debug << "WaveformAnalysisRAVEN: Dictionary state - built: " << dictionary_built
        << ", cached_nsamples: " << cached_nsamples << ", cached_period: " << cached_digitizer_period << newline;
  debug << "WaveformAnalysisRAVEN: Current params - nsamples: " << nsamples << ", period: " << digitizer_period
        << newline;
  debug << "WaveformAnalysisRAVEN: Using raven_template_type: " << template_type << " ("
        << (template_type == 0 ? "lognormal" : "gaussian") << ")" << newline;
  debug << "WaveformAnalysisRAVEN: Dictionary size: " << nsamples << " x "
        << static_cast<int>(nsamples * upsample_factor) << newline;

  const int dict_size = static_cast<int>(nsamples * upsample_factor);
  fW.ResizeTo(nsamples, dict_size);
  fW.Zero();

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
        template_val = mag_factor * TMath::Gaus(sample_time, delay, gaussian_width, kTRUE);
      }

      fW(row, col) = -template_val;
    }
  }
}

double WaveformAnalysisRAVEN::EvaluateTemplate(double sample_time_ns, double delay_ns) const {
  const double mag_factor = vpe_charge * fTermOhms;
  double template_val = 0.0;
  if (template_type == 0) {  // lognormal
    double shift = delay_ns - lognormal_scale;
    if (sample_time_ns > shift) {
      template_val = mag_factor * TMath::LogNormal(sample_time_ns, lognormal_shape, shift, lognormal_scale);
    }
  } else {  // gaussian
    template_val = mag_factor * TMath::Gaus(sample_time_ns, delay_ns, gaussian_width, kTRUE);
  }
  return -template_val;  // negative sign matches dictionary convention
}

double WaveformAnalysisRAVEN::ComputeModelChi2(const TVectorD& region_vec,
                                               const std::vector<std::pair<double, double>>& weights,
                                               int start_sample) const {
  const int n = region_vec.GetNrows();
  double chi2 = 0.0;
  for (int s = 0; s < n; ++s) {
    const double sample_time = (start_sample + s) * fTimeStep;
    double model = 0.0;
    for (const auto& [t, w] : weights) {
      model += w * EvaluateTemplate(sample_time, t);
    }
    const double r = region_vec(s) - model;
    chi2 += r * r;
  }
  const int dof = std::max(1, n - static_cast<int>(weights.size()));
  return chi2 / dof;
}

std::vector<std::pair<double, double>> WaveformAnalysisRAVEN::GlobalFitWeights(
    const TVectorD& region_vec, const std::vector<std::pair<double, double>>& initial_weights, int start_sample,
    double& chi2ndf_out, int& npe_delta_out) {
#if NLOPT_Enabled
  const int N = static_cast<int>(initial_weights.size());
  if (N == 0) return initial_weights;

  // Helper: run NLOPT on a candidate set of (time, weight) pairs.
  // Returns the optimised pairs, or the input unchanged on failure.
  auto run_nlopt = [&](const std::vector<std::pair<double, double>>& cands) -> std::vector<std::pair<double, double>> {
    const int M = static_cast<int>(cands.size());
    if (M == 0) return cands;

    // Parameter layout: x[0..M-1] = times, x[M..2M-1] = weights
    std::vector<double> x(2 * M), lb(2 * M), ub(2 * M);
    for (int i = 0; i < M; ++i) {
      x[i] = cands[i].first;
      x[M + i] = cands[i].second;
      lb[i] = cands[i].first - global_fit_time_window;
      ub[i] = cands[i].first + global_fit_time_window;
      lb[M + i] = 0.0;
      ub[M + i] = cands[i].second * global_fit_weight_max_scale;
    }

    try {
      nlopt::opt opt(global_fit_algo.c_str(), 2 * M);
      opt.set_lower_bounds(lb);
      opt.set_upper_bounds(ub);
      opt.set_maxeval(global_fit_max_evals);
      opt.set_xtol_rel(global_fit_tolerance);

      auto objective = [&](unsigned /*n*/, const double* xv, double* /*grad*/) -> double {
        std::vector<std::pair<double, double>> tmp(M);
        for (int i = 0; i < M; ++i) tmp[i] = {xv[i], xv[M + i]};
        return ComputeModelChi2(region_vec, tmp, start_sample);
      };
      opt.set_min_objective(objective);

      double min_val;
      opt.optimize(x, min_val);
    } catch (const std::exception& e) {
      debug << "WaveformAnalysisRAVEN: GlobalFit NLOPT exception: " << e.what() << newline;
      return cands;  // return unchanged on failure
    }

    std::vector<std::pair<double, double>> result(M);
    for (int i = 0; i < M; ++i) result[i] = {x[i], x[M + i]};
    return result;
  };

  // ── 1. Continuous optimisation of the N-PE solution ─────────────────────
  std::vector<std::pair<double, double>> best = run_nlopt(initial_weights);
  double best_chi2 = ComputeModelChi2(region_vec, best, start_sample);
  npe_delta_out = 0;

  if (!global_fit_float_npe) {
    chi2ndf_out = best_chi2;
    return best;
  }

  // ── 2. Try N-1 (remove the weakest PE) ──────────────────────────────────
  if (N > 1) {
    // Find PE with minimum weight in the current best solution
    int weakest = 0;
    for (int i = 1; i < N; ++i) {
      if (best[i].second < best[weakest].second) weakest = i;
    }
    std::vector<std::pair<double, double>> cands_nm1;
    cands_nm1.reserve(N - 1);
    for (int i = 0; i < N; ++i) {
      if (i != weakest) cands_nm1.push_back(best[i]);
    }
    std::vector<std::pair<double, double>> opt_nm1 = run_nlopt(cands_nm1);
    double chi2_nm1 = ComputeModelChi2(region_vec, opt_nm1, start_sample);
    if (chi2_nm1 < best_chi2) {
      best_chi2 = chi2_nm1;
      best = opt_nm1;
      npe_delta_out = -1;
    }
  }

  // ── 3. Try N+1 (add a PE at the largest residual) ───────────────────────
  {
    // Compute residual of current best solution
    const int n_samples = region_vec.GetNrows();
    int peak_sample = 0;
    double peak_abs_residual = 0.0;
    for (int s = 0; s < n_samples; ++s) {
      const double sample_time = (start_sample + s) * fTimeStep;
      double model = 0.0;
      for (const auto& [t, w] : best) model += w * EvaluateTemplate(sample_time, t);
      const double abs_res = std::abs(region_vec(s) - model);
      if (abs_res > peak_abs_residual) {
        peak_abs_residual = abs_res;
        peak_sample = s;
      }
    }
    const double new_time = (start_sample + peak_sample) * fTimeStep;
    // Seed the new PE weight from the residual amplitude divided by template peak.
    // Fall back to weight_threshold if the template evaluates to zero.
    const double tmpl_peak = std::abs(EvaluateTemplate(new_time, new_time));
    const double new_weight = (tmpl_peak > 0.0) ? (peak_abs_residual / tmpl_peak) : weight_threshold;

    std::vector<std::pair<double, double>> cands_np1 = best;
    cands_np1.emplace_back(new_time, new_weight);
    std::vector<std::pair<double, double>> opt_np1 = run_nlopt(cands_np1);
    double chi2_np1 = ComputeModelChi2(region_vec, opt_np1, start_sample);
    if (chi2_np1 < best_chi2) {
      best_chi2 = chi2_np1;
      best = opt_np1;
      npe_delta_out = +1;
    }
  }

  chi2ndf_out = best_chi2;
  return best;

#else
  // NLOPT not available — return unchanged
  npe_delta_out = 0;
  return initial_weights;
#endif
}

void WaveformAnalysisRAVEN::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  // Build dictionary on first call or when digitizer parameters change
  const double period_tolerance = 1e-9;  // 1 ps tolerance for digitizer period comparison
  if (!dictionary_built || cached_nsamples != static_cast<int>(digitWfm.size()) ||
      std::abs(cached_digitizer_period - fTimeStep) > period_tolerance) {
    // Use current digitizer information from the waveform and base class
    int nsamples = static_cast<int>(digitWfm.size());
    double digitizer_period = fTimeStep;

    BuildDictionaryMatrix(nsamples, digitizer_period);

    cached_nsamples = nsamples;
    cached_digitizer_period = digitizer_period;
    dictionary_built = true;
  }

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

  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult("RAVEN");

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
      ProcessThresholdRegion(voltWfm, start_sample, end_sample, fit_result, gain_calibration);
    }
  } else {
    int start_sample = 0;
    int end_sample = static_cast<int>(voltWfm.size()) - 1;

    // Perform rsNNLS on the entire waveform
    ProcessThresholdRegion(voltWfm, start_sample, end_sample, fit_result, gain_calibration);
  }
}

TVectorD WaveformAnalysisRAVEN::Thresholded_rsNNLS(const TMatrixD& W_region, const TVectorD& voltVec,
                                                   const double threshold, double& chi2ndf_out, int& iterations_out) {
  const int D = voltVec.GetNrows();
  const int K = W_region.GetNcols();

  if (W_region.GetNrows() != D) {
    RAT::Log::Die("WaveformAnalysisRAVEN: Dictionary region row dimension mismatch.");
  }

  // Initial NNLS solve
  TVectorD h_full(K);
  h_full.Zero();
  h_full = Math::NNLS_LawsonHanson(W_region, voltVec, epsilon, 0, 0);

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

    // Remove component with smallest weight
    h_full(P[minPos]) = 0.0;
    P.erase(P.begin() + minPos);
    if (P.empty()) {
      h_full.Zero();
      return h_full;
    }

    // Re-solve on reduced active set
    TMatrixD W_P = subCols(W_region, P);
    TVectorD h_reduced(P.size());
    h_reduced.Zero();
    h_reduced = Math::NNLS_LawsonHanson(W_P, voltVec, epsilon, 0, 0);

    // Update full weight vector
    h_full.Zero();
    for (size_t k = 0; k < P.size(); ++k) {
      h_full(P[k]) = h_reduced(k);
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

void WaveformAnalysisRAVEN::ProcessThresholdRegion(const std::vector<double>& voltWfm, int start_sample, int end_sample,
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

  // Merge nearby weights (previously done inside ExtractPhotoelectrons)
  std::vector<std::pair<double, double>> merged =
      MergeNearbyWeights(region_weights, dict_start, dict_cols, weight_merge_window);

  // Optional global fit: refine times and charges, optionally vary PE count by ±1
  int npe_delta = 0;
  if (global_fit_enabled && !merged.empty()) {
    merged = GlobalFitWeights(region_vec, merged, start_sample, chi2ndf, npe_delta);
  }

  // Extract PEs from the (possibly refined) merged weights
  ExtractPhotoelectronsFromMerged(merged, start_sample, end_sample, chi2ndf, iterations_ran, npe_delta, fit_result,
                                  gain_calibration);
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

  // Merge weights within the time window
  std::vector<std::pair<double, double>> merged_weights;
  merged_weights.reserve(time_weight_pairs.size());

  size_t i = 0;
  while (i < time_weight_pairs.size()) {
    // Start a new cluster
    double cluster_weight_sum = time_weight_pairs[i].second;
    double cluster_time_weighted_sum = time_weight_pairs[i].first * time_weight_pairs[i].second;
    size_t cluster_end = i + 1;

    // Extend cluster to include all weights within merge_window of any cluster member
    while (cluster_end < time_weight_pairs.size()) {
      // Check if next weight is within merge_window of the cluster's weighted mean time
      double cluster_mean_time = cluster_time_weighted_sum / cluster_weight_sum;
      if (time_weight_pairs[cluster_end].first - cluster_mean_time <= merge_window) {
        // Add to cluster
        cluster_weight_sum += time_weight_pairs[cluster_end].second;
        cluster_time_weighted_sum += time_weight_pairs[cluster_end].first * time_weight_pairs[cluster_end].second;
        cluster_end++;
      } else {
        break;
      }
    }

    // Store merged weight with charge-weighted mean time
    double merged_time = cluster_time_weighted_sum / cluster_weight_sum;
    merged_weights.emplace_back(merged_time, cluster_weight_sum);

    i = cluster_end;
  }

  return merged_weights;
}

void WaveformAnalysisRAVEN::ExtractPhotoelectrons(const TVectorD& region_weights, int dict_start, int dict_cols,
                                                  int start_sample, int end_sample, double chi2ndf, int iterations_ran,
                                                  DS::WaveformAnalysisResult* fit_result, double gain_calibration) {
  std::vector<std::pair<double, double>> merged_weights =
      MergeNearbyWeights(region_weights, dict_start, dict_cols, weight_merge_window);
  ExtractPhotoelectronsFromMerged(merged_weights, start_sample, end_sample, chi2ndf, iterations_ran, 0, fit_result,
                                  gain_calibration);
}

void WaveformAnalysisRAVEN::ExtractPhotoelectronsFromMerged(
    const std::vector<std::pair<double, double>>& merged_weights, int start_sample, int end_sample, double chi2ndf,
    int iterations_ran, int global_fit_npe_delta, DS::WaveformAnalysisResult* fit_result, double gain_calibration) {
  const double template_scale = (template_type == 0) ? lognormal_scale : gaussian_width;
  const double region_start_time = start_sample * fTimeStep;
  const double region_end_time = end_sample * fTimeStep;

  for (const auto& [delay, weight] : merged_weights) {
    // Sanity check - ensure PE time is within expected range
    if (delay < region_start_time - 3.0 * template_scale || delay > region_end_time + 3.0 * template_scale) {
      warn << "WaveformAnalysisRAVEN: PE time " << delay << " ns outside expected range ["
           << (region_start_time - 3.0 * template_scale) << ", " << (region_end_time + 3.0 * template_scale)
           << "] for region [" << start_sample << ", " << end_sample << "]" << newline;
      continue;
    }

    // Charge calculation: apply gain calibration
    const double pe_charge = weight * vpe_charge * gain_calibration;  // Charge in pC

    // Estimate number of PEs using likelihood method
    const double calibrated_vpe_charge = vpe_charge * gain_calibration;
    const size_t npe =
        npe_estimate ? EstimateNPE(pe_charge, calibrated_vpe_charge, npe_estimate_charge_width, npe_estimate_max_pes)
                     : 1;

    // Build FOM map — include global fit entries only when the global fit ran
    std::map<std::string, Double_t> foms = {
        {"chi2ndf", chi2ndf},
        {"iterations_ran", static_cast<double>(iterations_ran)},
        {"weight", weight / npe},
        {"estimated_npe", static_cast<double>(npe)},
    };
    if (global_fit_enabled) {
      foms["global_fit_chi2ndf"] = chi2ndf;
      if (global_fit_float_npe) {
        foms["global_fit_npe_delta"] = static_cast<double>(global_fit_npe_delta);
      }
    }

    // Add each estimated PE with divided charge
    for (size_t ipe = 0; ipe < npe; ++ipe) {
      fit_result->AddPE(delay, pe_charge / npe, foms);
    }
  }
}
}  // namespace RAT