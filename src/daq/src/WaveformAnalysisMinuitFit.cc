#include <Math/Factory.h>
#include <Math/Functor.h>
#include <TMath.h>

#include <RAT/DS/PMTInfo.hh>
#include <RAT/DS/RunStore.hh>
#include <RAT/Log.hh>
#include <RAT/WaveformAnalysisMinuitFit.hh>
#include <algorithm>
#include <cmath>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {

namespace {
// PMTPULSE stores the gaussian width as a piecewise-linear PDF that the waveform generator
// samples by inverse-CDF. MinuitFit starts from one template per PMT, so collapse the PDF to its
// median by inverting the same trapezoidal CDF. The median is better for skewed width distributions.
double PDFMedian(const std::vector<double>& x, const std::vector<double>& prob) {
  if (x.size() != prob.size() || x.size() < 2) {
    RAT::Log::Die(
        "WaveformAnalysisMinuitFit: PMTPULSE width PDF needs at least two matching width and probability points.");
  }

  std::vector<double> cumu(x.size(), 0.0);
  for (size_t i = 0; i + 1 < x.size(); ++i) {
    cumu[i + 1] = cumu[i] + (x[i + 1] - x[i]) * (prob[i] + prob[i + 1]) / 2.0;
  }

  if (cumu.back() <= 0.0) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: PMTPULSE width PDF has non-positive normalization.");
  }

  for (size_t i = 1; i < x.size(); ++i) {
    const double c = cumu[i] / cumu.back();
    if (0.5 <= c) {
      const double c_prev = cumu[i - 1] / cumu.back();
      return (0.5 - c_prev) * (x[i] - x[i - 1]) / (c - c_prev) + x[i - 1];
    }
  }
  return x.back();
}

// Parameter layout of a cluster fit: the parameters shared by the cluster, then a
// (time, charge) pair per PE.
constexpr int kBaseline = 0;
constexpr int kWidthScale = 1;
constexpr int kFirstPE = 2;
}  // namespace

void WaveformAnalysisMinuitFit::Configure(const std::string& config_name) {
  debug << "WaveformAnalysisMinuitFit: Configure called with config_name " << config_name << newline;
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);

    seed_analyzer = fDigit->GetS("seed_analyzer");
    seed_missing_warned = false;

    // Template type configuration
    template_type = fDigit->GetI("template_type");  // 0=lognormal, 1=gaussian

    // Single photoelectron waveform parameters
    if (template_type == 0) {                             // lognormal
      lognormal_scale = fDigit->GetD("lognormal_scale");  // LogNormal 'm' parameter
      lognormal_shape = fDigit->GetD("lognormal_shape");  // LogNormal 'sigma' parameter
    } else if (template_type == 1) {                      // gaussian
      gaussian_width = fDigit->GetD("gaussian_width");    // Gaussian 'sigma' parameter
    } else {
      RAT::Log::Die("WaveformAnalysisMinuitFit: Invalid template_type " + std::to_string(template_type) +
                    ". Must be 0 (lognormal) or 1 (gaussian).");
    }
    width_from_pmtpulse = fDigit->GetZ("width_from_pmtpulse");

    // Fit configuration
    noise_rms = fDigit->GetD("noise_rms");
    time_window = fDigit->GetD("time_window");
    duplicate_spread = fDigit->GetD("duplicate_spread");
    charge_max_factor = fDigit->GetD("charge_max_factor");
    fit_window_pre = fDigit->GetD("fit_window_pre");
    fit_window_post = fDigit->GetD("fit_window_post");
    max_cluster_pes = fDigit->GetI("max_cluster_pes");

    float_width = fDigit->GetZ("float_width");
    width_scale_min = fDigit->GetD("width_scale_min");
    width_scale_max = fDigit->GetD("width_scale_max");

    float_baseline = fDigit->GetZ("float_baseline");
    baseline_limit = fDigit->GetD("baseline_limit");

    // Minimizer configuration
    minuit_strategy = fDigit->GetI("minuit_strategy");
    minuit_tolerance = fDigit->GetD("minuit_tolerance");
    minuit_max_calls = fDigit->GetI("minuit_max_calls");
  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: Unable to find analysis parameters.");
  }

  if (noise_rms <= 0.0) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: noise_rms must be positive.");
  }
  if (time_window <= 0.0) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: time_window must be positive.");
  }
  // Every fit starts from a width scale of 1, so the limits must contain it.
  if (float_width && !(0.0 < width_scale_min && width_scale_min <= 1.0 && 1.0 <= width_scale_max)) {
    RAT::Log::Die(
        "WaveformAnalysisMinuitFit: Width scale limits must satisfy 0 < width_scale_min <= 1 <= "
        "width_scale_max.");
  }

  fModelShapeCache.clear();

  fMinimizer.reset(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
  if (!fMinimizer) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: Unable to create the Minuit2 minimizer.");
  }
  ConfigureMinimizer();
}

void WaveformAnalysisMinuitFit::ConfigureMinimizer() {
  fMinimizer->SetStrategy(minuit_strategy);
  fMinimizer->SetTolerance(minuit_tolerance);
  fMinimizer->SetMaxFunctionCalls(minuit_max_calls);
  fMinimizer->SetErrorDef(1.0);  // chi-square
  fMinimizer->SetPrintLevel(0);
}

void WaveformAnalysisMinuitFit::SetS(std::string param, std::string value) {
  if (param == "seed_analyzer") {
    seed_analyzer = value;
    seed_missing_warned = false;
  } else {
    WaveformAnalyzerBase::SetS(param, value);
  }
}

// Template parameters feed the fallback shape, which the model shape cache may hold.
void WaveformAnalysisMinuitFit::SetD(std::string param, double value) {
  if (param == "lognormal_scale") {
    lognormal_scale = value;
    fModelShapeCache.clear();
  } else if (param == "lognormal_shape") {
    lognormal_shape = value;
    fModelShapeCache.clear();
  } else if (param == "gaussian_width") {
    gaussian_width = value;
    fModelShapeCache.clear();
  } else if (param == "noise_rms") {
    noise_rms = value;
  } else if (param == "time_window") {
    time_window = value;
  } else if (param == "duplicate_spread") {
    duplicate_spread = value;
  } else if (param == "charge_max_factor") {
    charge_max_factor = value;
  } else if (param == "fit_window_pre") {
    fit_window_pre = value;
  } else if (param == "fit_window_post") {
    fit_window_post = value;
  } else if (param == "width_scale_min") {
    width_scale_min = value;
  } else if (param == "width_scale_max") {
    width_scale_max = value;
  } else if (param == "baseline_limit") {
    baseline_limit = value;
  } else if (param == "minuit_tolerance") {
    minuit_tolerance = value;
    ConfigureMinimizer();
  } else {
    WaveformAnalyzerBase::SetD(param, value);
  }
}

void WaveformAnalysisMinuitFit::SetI(std::string param, int value) {
  if (param == "template_type") {
    template_type = value;
    if (template_type != 0 && template_type != 1) {
      RAT::Log::Die("WaveformAnalysisMinuitFit: Invalid template_type " + std::to_string(value) +
                    ". Must be 0 (lognormal) or 1 (gaussian).");
    }
    fModelShapeCache.clear();
  } else if (param == "width_from_pmtpulse") {
    width_from_pmtpulse = (value != 0);
  } else if (param == "max_cluster_pes") {
    max_cluster_pes = static_cast<size_t>(value);
  } else if (param == "float_width") {
    float_width = (value != 0);
  } else if (param == "float_baseline") {
    float_baseline = (value != 0);
  } else if (param == "minuit_strategy") {
    minuit_strategy = value;
    ConfigureMinimizer();
  } else if (param == "minuit_max_calls") {
    minuit_max_calls = value;
    ConfigureMinimizer();
  } else {
    WaveformAnalyzerBase::SetI(param, value);
  }
}

WaveformAnalysisMinuitFit::TemplateShape WaveformAnalysisMinuitFit::ConfiguredShape() const {
  // The gaussian template has no shape parameter, and Configure leaves lognormal_shape unset
  // whenever that template is selected, so the shape stays zero here: reading it would be an
  // uninitialized read.
  return (template_type == 0) ? TemplateShape(lognormal_shape, lognormal_scale) : TemplateShape(0.0, gaussian_width);
}

// The template is the waveform generator's pulse shape, so its per-model parameters come from
// the generator's own PMTPULSE table. There 'lognormal_width' and 'lognormal_mean' are the
// template's shape and scale, and the gaussian width is a PDF rather than a single value.
WaveformAnalysisMinuitFit::TemplateShape WaveformAnalysisMinuitFit::ShapeForModel(const std::string& model_name) {
  std::map<std::string, TemplateShape>::const_iterator cached = fModelShapeCache.find(model_name);
  if (cached != fModelShapeCache.end()) {
    return cached->second;
  }

  DBLinkPtr lpulse;
  try {
    lpulse = DB::Get()->GetLink("PMTPULSE", model_name);
    lpulse->GetS("index");
  } catch (DBNotFoundError&) {
    // The generator falls back to the default entry for an unlisted model; match it.
    try {
      lpulse = DB::Get()->GetLink("PMTPULSE", "");
      lpulse->GetS("index");
    } catch (DBNotFoundError&) {
      lpulse = nullptr;
    }
  }

  // Fall back to the configured parameters whenever PMTPULSE cannot describe this model with
  // the template MinuitFit is set up to fit.
  TemplateShape shape = ConfiguredShape();
  std::string reason;

  if (!lpulse) {
    reason = "no PMTPULSE entry";
  } else {
    try {
      const std::string pulse_type = lpulse->GetS("pulse_type");
      const std::string pulse_shape = lpulse->GetS("pulse_shape");
      const std::string wanted = (template_type == 0) ? "lognormal" : "gaussian";

      if (pulse_type != "analytic") {
        reason = "PMTPULSE pulse_type is " + pulse_type + ", not analytic";
      } else if (pulse_shape != wanted) {
        reason = "PMTPULSE pulse_shape is " + pulse_shape + ", but MinuitFit fits " + wanted;
      } else if (template_type == 0) {
        shape = TemplateShape(lpulse->GetD("lognormal_width"), lpulse->GetD("lognormal_mean"));
      } else {
        shape.second = PDFMedian(lpulse->GetDArray("gaussian_width"), lpulse->GetDArray("gaussian_width_prob"));
      }
    } catch (DBNotFoundError&) {
      reason = "PMTPULSE entry is missing template parameters";
    }
  }

  if (reason.empty()) {
    info << "WaveformAnalysisMinuitFit: Template for " << model_name
         << " from PMTPULSE: " << (template_type == 0 ? "lognormal m " : "gaussian sigma ") << shape.second << " ns"
         << newline;
  } else {
    warn << "WaveformAnalysisMinuitFit: " << reason << " for " << model_name
         << ". Using the configured template parameters." << newline;
  }

  fModelShapeCache[model_name] = shape;
  return shape;
}

WaveformAnalysisMinuitFit::TemplateShape WaveformAnalysisMinuitFit::ShapeForPMT(int pmtid) {
  DS::Run* run = DS::RunStore::GetCurrentRun();
  TemplateShape shape = ShapeForModel(run->GetPMTInfo()->GetModelNameByID(pmtid));
  // The generator scales the lognormal 'm' and the gaussian width per channel, leaving the
  // lognormal 'sigma' alone.
  shape.second *= run->GetChannelStatus()->GetPulseWidthScaleByPMTID(pmtid);
  return shape;
}

// The same unit-area pulse RAVEN builds its dictionary from: TMath::LogNormal(lag, sigma, -m, m)
// or TMath::Gaus(lag, 0, sigma, kTRUE), with the lognormal sigma taken from fFitShape.
double WaveformAnalysisMinuitFit::Template(double lag, double scale, double& dlag, double& dscale) const {
  if (template_type == 0) {  // lognormal
    const double sigma = fFitShape.first;
    const double u = lag + scale;
    if (u <= 0.0) {
      dlag = 0.0;
      dscale = 0.0;
      return 0.0;
    }
    const double l = std::log(u / scale);
    const double value = std::exp(-l * l / (2.0 * sigma * sigma)) / (u * sigma * std::sqrt(2.0 * TMath::Pi()));
    dlag = -value * (1.0 + l / (sigma * sigma)) / u;
    dscale = -value * (1.0 / u + l / (sigma * sigma) * (1.0 / u - 1.0 / scale));
    return value;
  }

  // gaussian
  const double z = lag / scale;
  const double value = std::exp(-z * z / 2.0) / (scale * std::sqrt(2.0 * TMath::Pi()));
  dlag = -value * z / scale;
  dscale = value * (z * z - 1.0) / scale;
  return value;
}

// The model is a baseline minus a negative-going template per PE, scaled from charge (pC) to
// voltage (mV) by the termination resistance.
double WaveformAnalysisMinuitFit::Chi2(const double* x) {
  const double scale = x[kWidthScale] * fFitShape.second;
  double dlag, dscale;
  double chi2 = 0.0;
  for (size_t i = 0; i < fFitTimes.size(); ++i) {
    double model = x[kBaseline];
    for (size_t k = 0; k < fFitNPE; ++k) {
      const double time = x[kFirstPE + 2 * k];
      const double charge = x[kFirstPE + 2 * k + 1];
      model -= charge * fTermOhms * Template(fFitTimes[i] - time, scale, dlag, dscale);
    }
    const double residual = fFitVolts[i] - model;
    chi2 += residual * residual;
  }
  return chi2 / (noise_rms * noise_rms);
}

void WaveformAnalysisMinuitFit::Chi2Gradient(const double* x, double* grad) {
  const double scale = x[kWidthScale] * fFitShape.second;
  std::fill(grad, grad + kFirstPE + 2 * fFitNPE, 0.0);

  std::vector<double> value(fFitNPE), dlag(fFitNPE), dscale(fFitNPE);
  for (size_t i = 0; i < fFitTimes.size(); ++i) {
    double model = x[kBaseline];
    for (size_t k = 0; k < fFitNPE; ++k) {
      value[k] = Template(fFitTimes[i] - x[kFirstPE + 2 * k], scale, dlag[k], dscale[k]);
      model -= x[kFirstPE + 2 * k + 1] * fTermOhms * value[k];
    }
    // d(chi2)/dp = -2 r dm/dp / noise^2 for residual r = v - m
    const double c = -2.0 * (fFitVolts[i] - model) / (noise_rms * noise_rms);
    grad[kBaseline] += c;
    for (size_t k = 0; k < fFitNPE; ++k) {
      const double charge = x[kFirstPE + 2 * k + 1];
      // The lag falls as the PE time rises, so dm/dtime = +charge * R * dT/dlag.
      grad[kFirstPE + 2 * k] += c * charge * fTermOhms * dlag[k];
      grad[kFirstPE + 2 * k + 1] -= c * fTermOhms * value[k];
      grad[kWidthScale] -= c * charge * fTermOhms * dscale[k] * fFitShape.second;
    }
  }
}

std::vector<WaveformAnalysisMinuitFit::SeedPE> WaveformAnalysisMinuitFit::ReadSeed(DS::DigitPMT* digitpmt,
                                                                                   double gain_calibration) {
  std::vector<SeedPE> pes;

  const std::vector<std::string> fitters = digitpmt->GetFitterNames();
  if (std::find(fitters.begin(), fitters.end(), seed_analyzer) == fitters.end()) {
    if (!seed_missing_warned) {
      warn << GetAnalyzerName() << ": no result named \"" << seed_analyzer
           << "\" to seed from. Does that analyzer run first in the macro? "
           << "Channels without a seed are left empty." << newline;
      seed_missing_warned = true;
    }
    return pes;
  }
  DS::WaveformAnalysisResult* seed = digitpmt->GetOrCreateWaveformAnalysisResult(seed_analyzer);
  const std::vector<Double_t>& seed_times = seed->getTimes();
  const std::vector<Double_t>& seed_charges = seed->getCharges();

  // AddPE() stored the seed times with the cable offset already subtracted, while the fit
  // works in raw digitizer time, so put the offset back.
  const double time_offset = seed->getTimeOffset();

  // The seed is sorted in time, so repeated times are adjacent. They come from an analyzer
  // splitting one hit into integer PEs, and would see identical gradients if they started
  // together, so spread them around the shared time.
  size_t i = 0;
  while (i < seed_times.size()) {
    size_t j = i + 1;
    while (j < seed_times.size() && seed_times[j] == seed_times[i]) ++j;
    if (seed_times[i] != WaveformUtil::INVALID) {
      const double seed_time = seed_times[i] + time_offset;
      const size_t n = j - i;
      for (size_t k = 0; k < n; ++k) {
        const double spread = (static_cast<double>(k) - (n - 1) / 2.0) * duplicate_spread;
        const double start_time = std::clamp(seed_time + spread, seed_time - time_window, seed_time + time_window);
        pes.push_back({seed_time, start_time, seed_charges[i + k] / gain_calibration});
      }
    }
    i = j;
  }
  return pes;
}

// Each PE needs the samples within its time window, padded by fit_window_pre/post. PEs whose
// sample ranges overlap are fit together; the seed is sorted, so a cluster closes as soon as a
// PE's range starts beyond it.
std::vector<WaveformAnalysisMinuitFit::Cluster> WaveformAnalysisMinuitFit::BuildClusters(const std::vector<SeedPE>& pes,
                                                                                         int nsamples) const {
  std::vector<Cluster> clusters;
  for (const SeedPE& pe : pes) {
    const int first =
        std::max(0, static_cast<int>(std::ceil((pe.seed_time - time_window - fit_window_pre) / fTimeStep)));
    const int last = std::min(nsamples - 1,
                              static_cast<int>(std::floor((pe.seed_time + time_window + fit_window_post) / fTimeStep)));
    if (clusters.empty() || first > clusters.back().last_sample) {
      clusters.push_back({{}, first, last});
    }
    clusters.back().pes.push_back(pe);
    clusters.back().last_sample = std::max(clusters.back().last_sample, last);
  }
  return clusters;
}

void WaveformAnalysisMinuitFit::FitCluster(const Cluster& cluster, const std::vector<double>& voltWfm,
                                           const TemplateShape& shape, DS::WaveformAnalysisResult* fit_result,
                                           double gain_calibration) {
  const size_t npe = cluster.pes.size();
  const int nsamples = cluster.last_sample - cluster.first_sample + 1;
  const int nfree = 2 * npe + (float_width ? 1 : 0) + (float_baseline ? 1 : 0);
  if (npe > max_cluster_pes || nsamples <= nfree) {
    EmitSeed(cluster, fit_result, gain_calibration, -1);
    return;
  }

  fFitTimes.resize(nsamples);
  fFitVolts.resize(nsamples);
  for (int i = 0; i < nsamples; ++i) {
    fFitTimes[i] = (cluster.first_sample + i) * fTimeStep;
    fFitVolts[i] = voltWfm[cluster.first_sample + i];
  }
  fFitShape = shape;
  fFitNPE = npe;

  const unsigned int npar = kFirstPE + 2 * npe;
  ROOT::Math::GradFunctor chi2(this, &WaveformAnalysisMinuitFit::Chi2, &WaveformAnalysisMinuitFit::Chi2Gradient, npar);
  fMinimizer->Clear();
  fMinimizer->SetFunction(chi2);

  if (float_baseline) {
    fMinimizer->SetLimitedVariable(kBaseline, "baseline", 0.0, 0.1 * noise_rms, -baseline_limit, baseline_limit);
  } else {
    fMinimizer->SetFixedVariable(kBaseline, "baseline", 0.0);
  }
  if (float_width) {
    fMinimizer->SetLimitedVariable(kWidthScale, "width_scale", 1.0, 0.05, width_scale_min, width_scale_max);
  } else {
    fMinimizer->SetFixedVariable(kWidthScale, "width_scale", 1.0);
  }
  for (size_t k = 0; k < npe; ++k) {
    const SeedPE& pe = cluster.pes[k];
    const std::string id = std::to_string(k);
    fMinimizer->SetLimitedVariable(kFirstPE + 2 * k, "time_" + id, pe.start_time, 0.1 * time_window,
                                   pe.seed_time - time_window, pe.seed_time + time_window);
    // A seed with no positive charge gives no scale for an upper limit.
    if (pe.charge > 0.0) {
      fMinimizer->SetLimitedVariable(kFirstPE + 2 * k + 1, "charge_" + id, pe.charge, 0.1 * pe.charge, 0.0,
                                     charge_max_factor * pe.charge);
    } else {
      fMinimizer->SetLowerLimitedVariable(kFirstPE + 2 * k + 1, "charge_" + id, 0.0, 0.1, 0.0);
    }
  }

  const bool valid = fMinimizer->Minimize();
  const int status = fMinimizer->Status();
  if (!valid || !std::isfinite(fMinimizer->MinValue())) {
    EmitSeed(cluster, fit_result, gain_calibration, status);
    return;
  }

  const double* x = fMinimizer->X();
  const double* errors = fMinimizer->Errors();
  const double chi2ndf = fMinimizer->MinValue() / (nsamples - nfree);
  const double edm = fMinimizer->Edm();
  // A time this close to its window edge was stopped by the limit, not by the waveform.
  const double limit_tolerance = 1e-3 * time_window;
  for (size_t k = 0; k < npe; ++k) {
    const SeedPE& pe = cluster.pes[k];
    const double time = x[kFirstPE + 2 * k];
    const double charge = x[kFirstPE + 2 * k + 1];
    const bool at_limit = std::abs(std::abs(time - pe.seed_time) - time_window) < limit_tolerance;
    fit_result->AddPE(time, charge * gain_calibration,
                      {
                          {"chi2ndf", chi2ndf},
                          {"fit_valid", 1.0},
                          {"fit_status", static_cast<double>(status)},
                          {"edm", edm},
                          {"cluster_npe", static_cast<double>(npe)},
                          {"time_error", errors[kFirstPE + 2 * k]},
                          {"charge_error", errors[kFirstPE + 2 * k + 1] * gain_calibration},
                          {"time_shift", time - pe.seed_time},
                          {"width_scale", x[kWidthScale]},
                          {"baseline", x[kBaseline]},
                          {"at_limit", at_limit ? 1.0 : 0.0},
                      });
  }
}

// Unfit PEs sit at their seed times, so their time shift is zero; everything else the fit
// would have measured is invalid.
void WaveformAnalysisMinuitFit::EmitSeed(const Cluster& cluster, DS::WaveformAnalysisResult* fit_result,
                                         double gain_calibration, int fit_status) {
  for (const SeedPE& pe : cluster.pes) {
    fit_result->AddPE(pe.seed_time, pe.charge * gain_calibration,
                      {
                          {"chi2ndf", WaveformUtil::INVALID},
                          {"fit_valid", 0.0},
                          {"fit_status", static_cast<double>(fit_status)},
                          {"edm", WaveformUtil::INVALID},
                          {"cluster_npe", static_cast<double>(cluster.pes.size())},
                          {"time_error", WaveformUtil::INVALID},
                          {"charge_error", WaveformUtil::INVALID},
                          {"time_shift", 0.0},
                          {"width_scale", WaveformUtil::INVALID},
                          {"baseline", WaveformUtil::INVALID},
                          {"at_limit", 0.0},
                      });
  }
}

void WaveformAnalysisMinuitFit::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  double pedestal = digitpmt->GetPedestal();
  if (pedestal == -9999) {
    RAT::Log::Die("WaveformAnalysisMinuitFit: Pedestal is invalid! Did you run WaveformPrep first?");
  }

  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult(GetAnalyzerName());

  const int pmtid = digitpmt->GetID();
  // Per-PMT gain calibration, applied the same way as the seed analyzers so charges stay comparable
  double gain_calibration = DS::RunStore::GetCurrentRun()->GetChannelStatus()->GetChargeScaleByPMTID(pmtid);

  std::vector<SeedPE> pes = ReadSeed(digitpmt, gain_calibration);
  if (pes.empty()) {
    return;
  }

  std::vector<double> voltWfm = WaveformUtil::ADCtoVoltage(digitWfm, fVoltageRes, pedestal);
  const TemplateShape shape = width_from_pmtpulse ? ShapeForPMT(pmtid) : ConfiguredShape();

  for (const Cluster& cluster : BuildClusters(pes, static_cast<int>(voltWfm.size()))) {
    FitCluster(cluster, voltWfm, shape, fit_result, gain_calibration);
  }
}

}  // namespace RAT
