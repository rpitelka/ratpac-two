#include <OsqpEigen/OsqpEigen.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>

#include <RAT/Log.hh>
#include <RAT/WaveformAnalysisRSNNLS.hh>
#include <algorithm>  // for std::remove, std::remove_if

#include "Eigen/Dense"
#include "Eigen/Sparse"
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
    warn << "WaveformAnalysisRSNNLS: No crossings found in waveform." << newline;
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
      nonzero_weights.push_back(w[i]);  // w is now indexed by support position, not original index
    }
  }

  if (times.empty()) {
    warn << "WaveformAnalysisRSNNLS: No non-zero weights found in waveform." << newline;
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
    pulse_template->SetPulseTimeOffset(delay);
    for (int j = 0; j < signal_length; ++j) {
      // Calculate the time for each bin in the template
      double t = j * fTimeStep;
      // Get the value of the pulse template at that time
      template_matrix(j, i) = pulse_template->GetPulseHeight(t);
    }
  }
  return template_matrix;
}

// Implements thresholded reverse-sparse NNLS using OsqpEigen
std::pair<std::vector<int>, Eigen::VectorXd> WaveformAnalysisRSNNLS::Thresholded_rsNNLS(const Eigen::VectorXd& m,
                                                                                        const Eigen::MatrixXd& S,
                                                                                        double threshold) {
  const int D = m.size();
  const int K = S.cols();
  if (S.rows() != D) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: S.rows() must equal m.size().");
  }

  // Numerical tolerance for zero detection
  const double eps = 1e-12;
  const int max_iterations = K;  // Safety limit to prevent infinite loops

  // Precompute QP data: P = S^T S, q = -S^T m
  Eigen::SparseMatrix<double> P = (S.transpose() * S).sparseView();
  // OSQP expects upper-triangular Hessian
  P = P.triangularView<Eigen::Upper>();
  Eigen::VectorXd q = -S.transpose() * m;

  // Constraint matrix A = I_K  (0 <= w <= u)
  Eigen::SparseMatrix<double> A(K, K);
  A.setIdentity();

  // Lower/upper bounds
  Eigen::VectorXd lower = Eigen::VectorXd::Zero(K);
  Eigen::VectorXd upper = Eigen::VectorXd::Constant(K, OsqpEigen::INFTY);

  // Setup OSQP solver wrapper
  OsqpEigen::Solver solver;
  solver.settings()->setWarmStart(true);
  solver.settings()->setVerbosity(false);
  solver.data()->setNumberOfVariables(K);
  solver.data()->setNumberOfConstraints(K);
  solver.data()->setHessianMatrix(P);
  solver.data()->setGradient(q);
  solver.data()->setLinearConstraintsMatrix(A);
  solver.data()->setLowerBound(lower);
  solver.data()->setUpperBound(upper);

  if (!solver.initSolver()) {
    RAT::Log::Die("OSQP initialization failed");
  }

  // 1) Solve full-support NNLS
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: Initial OSQP solve failed");
  }
  Eigen::VectorXd w = solver.getSolution();

  // Build initial support using numerical tolerance
  std::vector<int> support;
  support.reserve(K);
  for (int j = 0; j < K; ++j) {
    if (w[j] > eps) support.push_back(j);
  }

  // 2) Threshold-prune loop with safety counter
  int iteration = 0;
  while (!support.empty() && iteration < max_iterations) {
    // Find index with smallest coefficient among support
    int j_star = support[0];
    double min_val = w[j_star];
    for (int idx : support) {
      if (w[idx] < min_val) {
        min_val = w[idx];
        j_star = idx;
      }
    }

    // Stop if all remaining >= threshold
    if (min_val >= threshold) break;

    // Otherwise prune: fix w[j_star] = 0 by setting its upper bound to 0
    upper[j_star] = 0.0;
    solver.updateBounds(lower, upper);

    // Warm-start from previous solution
    solver.updateGradient(q);
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
      RAT::Log::Die("WaveformAnalysisRSNNLS: OSQP solve failed during pruning");
    }
    w = solver.getSolution();

    // Efficiently rebuild support by removing the pruned element
    support.erase(std::remove(support.begin(), support.end(), j_star), support.end());

    // Also remove any elements that became effectively zero
    support.erase(std::remove_if(support.begin(), support.end(), [&w, eps](int idx) { return w[idx] <= eps; }),
                  support.end());

    ++iteration;
  }

  if (iteration >= max_iterations) {
    warn << "WaveformAnalysisRSNNLS: Maximum iterations reached in threshold pruning" << newline;
  }

  // Return only the non-zero coefficients for efficiency
  Eigen::VectorXd support_weights(support.size());
  for (size_t i = 0; i < support.size(); ++i) {
    support_weights[i] = w[support[i]];
  }

  return {support, support_weights};
}

}  // namespace RAT
