#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>

#include <RAT/Log.hh>
#include <RAT/WaveformAnalysisMultiLognormal.hh>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {
void WaveformAnalysisMultiLognormal::Configure(const std::string& config_name) {
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);
    fFitShape = fDigit->GetD("lognormal_shape");
    fFitScale = fDigit->GetD("lognormal_scale");
  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisMultiLognormal: Unable to find analysis parameters.");
  }
}

void WaveformAnalysisMultiLognormal::SetD(std::string param, double value) {
  if (param == "lognormal_shape") {
    fFitShape = value;
  } else if (param == "lognormal_scale") {
    fFitScale = value;
  } else {
    throw Processor::ParamUnknown(param);
  }
}

void WaveformAnalysisMultiLognormal::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  double pedestal = digitpmt->GetPedestal();
  if (pedestal == -9999) {
    RAT::Log::Die("WaveformAnalysisMultiLognormal: Pedestal is invalid! Did you run WaveformPrep first?");
  }
  // Convert from ADC to mV
  std::vector<double> voltWfm = WaveformUtil::ADCtoVoltage(digitWfm, fVoltageRes, pedestal = pedestal);
  fDigitTimeInWindow = digitpmt->GetDigitizedTimeNoOffset();
  // Fit waveform to lognormal
  fNPE = digitpmt->GetReconNPEs();
  if (fNPE != 0) {
    FitWaveform(voltWfm);

    DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult("MultiLognormal");

    // Add each individual PE from the multi-lognormal fit
    for (size_t i = 0; i < fFittedTimes.size(); ++i) {
      fit_result->AddPE(fFittedTimes[i], fFittedCharges[i], {{"baseline", fFittedBaseline}, {"chi2ndf", fChi2NDF}});
    }
  } else {
    debug << "WaveformAnalysisMultiLognormal: NPE is zero, skipping fit." << newline;
  }
}

static double MultiLognormal(double* x, double* par, int nlognormals) {
  /*
  Sum of nlognormals lognormal distributions.
  Each lognormal uses 5 parameters: mag, theta, baseline, m, s.
  Parameters are packed as [mag1, theta1, baseline1, m1, s1, mag2, theta2, ...]
  */
  // Check that we have the right number of parameters (5 per lognormal)
  int expected_params = 5 * nlognormals;
  if (expected_params <= 0) {
    RAT::Log::Die("MultiLognormalN: Invalid number of lognormals specified");
    return 0.0;
  }

  double sum = 0.0;
  for (int i = 0; i < nlognormals; ++i) {
    double mag = par[5 * i + 0];
    double theta = par[5 * i + 1];
    double baseline = par[5 * i + 2];
    double m = par[5 * i + 3];
    double s = par[5 * i + 4];
    if (x[0] > theta) {
      sum += baseline - mag * TMath::LogNormal(x[0], s, theta, m);
    } else {
      sum += baseline;
    }
  }
  return sum;
}

void WaveformAnalysisMultiLognormal::FitWaveform(const std::vector<double>& voltWfm) {
  /*
  Fit the PMT pulse to a multi-lognormal distribution
  */
  TH1D* wfm = new TH1D("wfm", "wfm", voltWfm.size(), 0, voltWfm.size() * fTimeStep);
  for (UShort_t i = 0; i < voltWfm.size(); i++) {
    wfm->SetBinContent(i, voltWfm[i]);
    // Arb. choice, TODO
    wfm->SetBinError(i, fVoltageRes * 2.0);
  }

  // Fit the entire waveform range
  double bf = 0;
  double tf = voltWfm.size() * fTimeStep;

  // Set timing ranges for parameter limits based on full waveform
  double thigh = tf;
  double tmed = fDigitTimeInWindow;
  double tlow = 0;

  // Use MultiLognormal with fNPE lognormals (5 parameters per lognormal)
  const int nlognormals = static_cast<int>(fNPE);
  const int ndf = 5 * nlognormals;

  // Create lambda wrapper to pass nlognormals to MultiLognormal
  auto multiLognormalWrapper = [nlognormals](double* x, double* par) -> double {
    return MultiLognormal(x, par, nlognormals);
  };

  TF1* ln_fit = new TF1("ln_fit", multiLognormalWrapper, bf, tf, ndf);

  // Set parameters for each lognormal
  for (int i = 0; i < nlognormals; ++i) {
    // Magnitude
    ln_fit->SetParameter(5 * i + 0, 40.0);
    ln_fit->SetParLimits(5 * i + 0, 1.0, 400.0);
    // Theta (timing offset for each PE)
    double theta_guess = tmed + i * 2.0;  // Spread PEs slightly in time
    ln_fit->SetParameter(5 * i + 1, theta_guess);
    ln_fit->SetParLimits(5 * i + 1, tlow, thigh);
    // Baseline (only fit baseline for first lognormal)
    if (i == 0) {
      ln_fit->SetParameter(5 * i + 2, 0.0);
      ln_fit->SetParLimits(5 * i + 2, -1.0, 1.0);
    } else {
      ln_fit->FixParameter(5 * i + 2, 0.0);
    }
    // Scale parameter
    ln_fit->SetParameter(5 * i + 3, fFitScale);
    ln_fit->SetParLimits(5 * i + 3, fFitScale - 5.0, fFitScale + 5.0);
    // Shape parameter (fixed)
    ln_fit->FixParameter(5 * i + 4, fFitShape);
  }

  wfm->Fit("ln_fit", "0QR", "", bf, tf);

  // Extract fitted parameters from all lognormals
  fChi2NDF = ln_fit->GetChisquare() / ln_fit->GetNDF();
  fFittedBaseline = ln_fit->GetParameter(2);  // Baseline from first lognormal

  // Store individual PE times and charges
  fFittedTimes.clear();
  fFittedCharges.clear();

  for (int i = 0; i < nlognormals; ++i) {
    double mag = ln_fit->GetParameter(5 * i + 0);
    double theta = ln_fit->GetParameter(5 * i + 1);
    double m = ln_fit->GetParameter(5 * i + 3);

    double charge_i = mag / fTermOhms;
    double time_i = theta + m;  // Peak time for this lognormal

    fFittedTimes.push_back(time_i);
    fFittedCharges.push_back(charge_i);
  }

  delete wfm;
  delete ln_fit;
}

}  // namespace RAT
