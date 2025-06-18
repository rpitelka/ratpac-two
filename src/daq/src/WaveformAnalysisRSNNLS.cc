#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>

#include <RAT/Log.hh>
#include <RAT/WaveformAnalysisRSNNLS.hh>
#include <unsupported/Eigen/NNLS>
#include <utility>

#include "Eigen/Core"
#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/RunStore.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/PMTPulse.hh"
#include "RAT/Processor.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {
void WaveformAnalysisRSNNLS::Configure(const std::string& config_name) {
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);
    fPulseWidthMean = fDigit->GetD("pulse_width_mean");
    fChargeMean = fDigit->GetD("charge_mean");
    fUpsampleFactor = fDigit->GetI("upsample_factor");
    fWeightThreshold = fDigit->GetD("weight_threshold");
  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: Unable to find analysis parameters.");
  }
}

void WaveformAnalysisRSNNLS::SetD(std::string param, double value) {
  if (param == "pulse_width_mean") {
    fPulseWidthMean = value;
  } else if (param == "charge_mean") {
    fChargeMean = value;
  } else if (param == "weight_threshold") {
    fWeightThreshold = value;
  } else {
    throw Processor::ParamUnknown(param);
  }
}

void WaveformAnalysisRSNNLS::SetI(std::string param, int value) {
  if (param == "upsample_factor") {
    fUpsampleFactor = value;
  } else {
    throw Processor::ParamUnknown(param);
  }
}

void WaveformAnalysisRSNNLS::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  double pedestal = digitpmt->GetPedestal();
  if (pedestal == -9999) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: Pedestal is invalid! Did you run WaveformPrep first?");
  }
  // Convert from ADC to mV
  std::vector<double> voltWfm = WaveformUtil::ADCtoVoltage(digitWfm, fVoltageRes, pedestal);
  int signal_length = voltWfm.size();
  PMTPulse pulse_template = MakeTemplate(digitpmt);
  Eigen::MatrixXd S_matrix = MakeTemplateMatrix(&pulse_template, signal_length, fUpsampleFactor);
  Eigen::VectorXd voltWfm_eigen(signal_length);
  for (int i = 0; i < signal_length; ++i) {
    voltWfm_eigen(i) = voltWfm[i];
  }

  std::vector<std::pair<int, int>> crossings = digitpmt->GetCrossingSamples();
  if (crossings.empty()) {
    debug << "WaveformAnalysisRSNNLS: No crossings found in waveform." << newline;
    return;
  }

  std::vector<double> times;
  std::vector<double> nonzero_weights;

  for (const std::pair<int, int>& crossing : crossings) {
    int start_sample = crossing.first;
    int end_sample = crossing.second;
    // Perform the thresholded reverse-sparse NNLS
    std::pair<std::vector<int>, Eigen::VectorXd> result =
        Thresholded_rsNNLS(voltWfm_eigen(Eigen::seq(start_sample, end_sample)),
                           S_matrix(Eigen::seq(start_sample, end_sample),
                                    Eigen::seq(start_sample * fUpsampleFactor, end_sample * fUpsampleFactor)),
                           fWeightThreshold);
    const std::vector<int>& indices = result.first;
    const Eigen::VectorXd& w = result.second;
    for (size_t i = 0; i < indices.size(); ++i) {
      times.push_back(start_sample * fTimeStep + indices[i] * fTimeStep / fUpsampleFactor);
      nonzero_weights.push_back(w[indices[i]]);
    }
  }

  if (times.empty()) {
    debug << "WaveformAnalysisRSNNLS: No non-zero weights found in waveform." << newline;
    return;
  }

  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult("rsNNLS");
  // Convert to times
  for (size_t i = 0; i < times.size(); ++i) {
    fit_result->AddPE(times[i], nonzero_weights[i]);
  }
}

PMTPulse WaveformAnalysisRSNNLS::MakeTemplate(DS::DigitPMT* digitpmt) {
  int pmtid = digitpmt->GetID();
  double pulse_width_scale = DS::RunStore::GetCurrentRun()->GetChannelStatus()->GetPulseWidthScaleByPMTID(pmtid);
  double charge_scale = DS::RunStore::GetCurrentRun()->GetChannelStatus()->GetChargeScaleByPMTID(pmtid);
  // Create a PMT pulse based on the configured parameters
  PMTPulse pulse("analytic", "gaussian");
  pulse.SetPulseCharge(fChargeMean * fTermOhms * charge_scale);
  pulse.SetGausPulseWidth(fPulseWidthMean * pulse_width_scale);
  pulse.SetPulseStartTime(0.0);
  pulse.SetPulseOffset(0.0);
  pulse.SetPulseMin(0.0);
  pulse.SetPulsePolarity(true);  // Negative polarity

  return pulse;
}

Eigen::MatrixXd WaveformAnalysisRSNNLS::MakeTemplateMatrix(PMTPulse* pulse_template, int signal_length,
                                                           int upsample_factor) {
  // Create a template matrix for the PMT pulse
  int n_bins = signal_length * upsample_factor;
  Eigen::MatrixXd template_matrix(signal_length, n_bins);
  for (int i = 0; i < n_bins; ++i) {
    double delay = i * fTimeStep / upsample_factor;
    pulse_template->SetPulseOffset(delay);
    for (int j = 0; j < signal_length; ++j) {
      // Calculate the time for each bin in the template
      double t = j * fTimeStep;
      // Get the value of the pulse template at that time
      template_matrix(j, i) = pulse_template->GetPulseHeight(t);
    }
  }
  return template_matrix;
}

// Thresholded reverse-sparse NNLS using Eigen's NNLS
std::pair<std::vector<int>, Eigen::VectorXd> WaveformAnalysisRSNNLS::Thresholded_rsNNLS(const Eigen::VectorXd& m,
                                                                                        const Eigen::MatrixXd& S,
                                                                                        double threshold) {
  int D = m.size();
  int D2 = S.rows();
  int K = S.cols();
  if (D2 != D) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: S.rows() must equal m.size().");
  }

  // 1) Solve the full-support NNLS: min_{w >= 0} || m - S w ||_2^2
  Eigen::NNLS<Eigen::MatrixXd> nnls(S);
  nnls.solve(m);
  Eigen::VectorXd w = nnls.x();

  // Build initial support S_reduced = [ j | w[j] > 0 ]
  std::vector<int> S_reduced;
  for (int j = 0; j < K; ++j) {
    if (w[j] > 0.0) S_reduced.push_back(j);
  }

  // 2) While there is some j in S_reduced with w[j] < threshold, prune:
  while (!S_reduced.empty()) {
    // Extract the coefficients on the current support
    std::vector<double> w_vals;
    for (int idx : S_reduced) w_vals.push_back(w[idx]);
    auto min_it = std::min_element(w_vals.begin(), w_vals.end());
    int j_min_idx = std::distance(w_vals.begin(), min_it);
    int j_star = S_reduced[j_min_idx];

    // If the smallest coefficient >= threshold, we are done
    if (w[j_star] >= threshold) break;

    // Otherwise, prune j_star: set it to zero and remove from S_reduced
    w[j_star] = 0.0;
    S_reduced.erase(S_reduced.begin() + j_min_idx);

    // If no atoms remain, return all-zeros
    if (S_reduced.empty()) return std::make_pair(std::vector<int>(), Eigen::VectorXd::Zero(K));

    // Re-solve NNLS on the reduced dictionary S[:, S_reduced]
    Eigen::MatrixXd S_P(D, S_reduced.size());
    for (size_t i = 0; i < S_reduced.size(); ++i) {
      S_P.col(i) = S.col(S_reduced[i]);
    }
    Eigen::NNLS<Eigen::MatrixXd> nnls_P(S_P);
    nnls_P.solve(m);
    Eigen::VectorXd w_P = nnls_P.x();

    // Zero out everything, then refill only the indices in S_reduced
    w.setZero();
    for (size_t i = 0; i < S_reduced.size(); ++i) {
      w[S_reduced[i]] = w_P[i];
    }
    // Loop until all remaining w[S_reduced] >= threshold or S_reduced is empty
  }
  // At this point, every nonzero entry of w is >= threshold (or w is zero)
  return std::make_pair(S_reduced, w);
}

}  // namespace RAT
