#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>

#include <RAT/Log.hh>
#include <RAT/WaveformAnalysisRSNNLS.hh>
#include <limits>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/NNLS.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {

void WaveformAnalysisRSNNLS::Configure(const std::string& config_name) {
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);
    vpe_scale = fDigit->GetD("vpe_scale");
    vpe_shape = fDigit->GetD("vpe_shape");
    vpe_charge = fDigit->GetD("vpe_charge");
    ds = fDigit->GetD("internal_sampling_period");
    vpe_nsamples = fDigit->GetI("vpe_nsamples");
    max_iterations = fDigit->GetI("max_iterations");
    weight_threshold = fDigit->GetD("weight_threshold");
    charge_threshold = fDigit->GetD("charge_threshold");
  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: Unable to find analysis parameters.");
  }

  // vpe_norm.resize(vpe_nsamples);
  // vpe_norm_flipped.resize(vpe_nsamples);
  // for (size_t idx = 0; idx < vpe_nsamples; ++idx) {
  //   double curr_time = idx * ds;
  //   vpe_norm.at(idx) = TMath::LogNormal(curr_time, vpe_shape, 0, vpe_scale);
  //   vpe_norm_flipped.at(vpe_nsamples - 1 - idx) = vpe_norm.at(idx);
  // }

  // Build dictionary matrix
  BuildDictionaryMatrix();
}

void WaveformAnalysisRSNNLS::BuildDictionaryMatrix() {
  const int nsamples = vpe_nsamples;  // Using existing waveform size
  const double time_step = 2.0;
  const double up_res = 4.0;  // Upsampling resolution factor (configurable)
  const int dict_size = static_cast<int>(nsamples * up_res);

  fW.ResizeTo(nsamples, dict_size);
  fW.Zero();

  // Generate delays array
  for (int col = 0; col < dict_size; ++col) {
    double delay = col * time_step / up_res;

    // Fill column with negative LogNormal template
    for (int row = 0; row < nsamples; ++row) {
      double sample_time = row * time_step;
      double template_val = 0.0;

      // LogNormal is only defined for sample_time > delay
      if (sample_time > delay) {
        template_val = TMath::LogNormal(sample_time, vpe_shape, delay, vpe_scale);
      }

      fW(row, col) = -template_val;
    }
  }
}

void WaveformAnalysisRSNNLS::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  double pedestal = digitpmt->GetPedestal();
  if (pedestal == -9999) {
    RAT::Log::Die("WaveformAnalysisRSNNLS: Pedestal is invalid! Did you run WaveformPrep first?");
  }
  // Convert from ADC to mV
  std::vector<double> voltWfm = WaveformUtil::ADCtoVoltage(digitWfm, fVoltageRes, pedestal);

  TVectorD weights = Thresholded_rsNNLS(voltWfm, weight_threshold);

  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult("LucyDDM");
  for (size_t ipe = 0; ipe < reco_times.size(); ++ipe) {
    fit_result->AddPE(reco_times[ipe], reco_charges[ipe], {{"chi2ndf", chi2ndf}, {"iterations_ran", iterations_ran}});
  }
}

TVectorD WaveformAnalysisRSNNLS::Thresholded_rsNNLS(const std::vector<double>& voltWfm, const double threshold) {
  /*
  Perform reverse sparse non-negative least squares fitting on the waveform,
  applying a threshold to identify significant pulses.
  Returns TVectorD of size K with nonzero entries >= threshold (or all zeros).
  */

  const int D = static_cast<int>(voltWfm.size());
  if (fW.GetNrows() != D) {
    RAT::Log::Die("Thresholded_rsNNLS: Dictionary row dimension mismatch.");
  }
  const int K = fW.GetNcols();

  TVectorD x(D);
  for (int i = 0; i < D; ++i) x(i) = voltWfm[i];

  TVectorD h_full(K);
  h_full.Zero();
  h_full = Math::NNLS_LawsonHanson(fW, x, -1.0, 0);

  std::vector<int> P;
  P.reserve(K);
  for (int j = 0; j < K; ++j)
    if (h_full(j) > 0.0) P.push_back(j);

  auto subCols = [](const TMatrixD& W, const std::vector<int>& cols) {
    TMatrixD S(W.GetNrows(), cols.size());
    for (size_t jj = 0; jj < cols.size(); ++jj) {
      int c = cols[jj];
      for (int i = 0; i < W.GetNrows(); ++i) S(i, jj) = W(i, c);
    }
    return S;
  };

  while (!P.empty()) {
    double minVal = std::numeric_limits<double>::infinity();
    size_t minPos = 0;
    for (size_t k = 0; k < P.size(); ++k) {
      double v = h_full(P[k]);
      if (v < minVal) {
        minVal = v;
        minPos = k;
      }
    }
    if (minVal >= threshold) break;

    h_full(P[minPos]) = 0.0;
    P.erase(P.begin() + minPos);
    if (P.empty()) {
      h_full.Zero();
      return h_full;
    }

    TMatrixD W_P = subCols(fW, P);
    TVectorD h_reduced(P.size());
    h_reduced.Zero();
    h_reduced = Math::NNLS_LawsonHanson(W_P, x, -1.0, 0);
    h_full.Zero();
    for (size_t k = 0; k < P.size(); ++k) h_full(P[k]) = h_reduced(k);
  }

  // Numerical safety: clip tiny negatives
  for (int j = 0; j < K; ++j)
    if (h_full(j) < 0.0) h_full(j) = 0.0;

  return h_full;
}

}  // namespace RAT
