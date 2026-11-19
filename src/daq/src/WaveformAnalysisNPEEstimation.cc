#include <Math/QuantFuncMathCore.h>

#include <RAT/DS/RunStore.hh>
#include <RAT/Log.hh>
#include <RAT/NPEEstimator.hh>
#include <RAT/PDFPMTCharge.hh>
#include <RAT/WaveformAnalysisNPEEstimation.hh>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "RAT/DS/DigitPMT.hh"
#include "RAT/DS/WaveformAnalysisResult.hh"
#include "RAT/WaveformUtil.hh"

namespace RAT {

void WaveformAnalysisNPEEstimation::Configure(const std::string& config_name) {
  try {
    fDigit = DB::Get()->GetLink("DIGITIZER_ANALYSIS", config_name);
    seed_analyzer = fDigit->GetS("seed_analyzer");
    output_name = fDigit->GetS("output_name");
    seed_missing_warned = false;
    spe_charge = fDigit->GetD("spe_charge");
    spe_charge_width = fDigit->GetD("spe_charge_width");
    spe_charge_from_pmtcharge = fDigit->GetZ("spe_charge_from_pmtcharge");
    max_pes = fDigit->GetI("max_pes");
    pe_time_spread = fDigit->GetD("pe_time_spread");
    pe_time_spread_fom = fDigit->GetS("pe_time_spread_fom");
    spread_fom_missing_warned = false;
  } catch (DBNotFoundError) {
    RAT::Log::Die("WaveformAnalysisNPEEstimation: Unable to find analysis parameters.");
  }
  if (pe_time_spread < 0.0) {
    RAT::Log::Die("WaveformAnalysisNPEEstimation: pe_time_spread must not be negative.");
  }
  fModelSPECache.clear();
  CheckNames();
}

void WaveformAnalysisNPEEstimation::CheckNames() const {
  if (seed_analyzer == output_name) {
    RAT::Log::Die("WaveformAnalysisNPEEstimation: output_name \"" + output_name +
                  "\" must differ from seed_analyzer, or the seed would be overwritten while it is read.");
  }
}

void WaveformAnalysisNPEEstimation::SetS(std::string param, std::string value) {
  if (param == "seed_analyzer") {
    seed_analyzer = value;
    seed_missing_warned = false;
    CheckNames();
  } else if (param == "output_name") {
    output_name = value;
    CheckNames();
  } else if (param == "pe_time_spread_fom") {
    pe_time_spread_fom = value;
    spread_fom_missing_warned = false;
  } else {
    WaveformAnalyzerBase::SetS(param, value);
  }
}

void WaveformAnalysisNPEEstimation::SetD(std::string param, double value) {
  if (param == "spe_charge") {
    spe_charge = value;
  } else if (param == "spe_charge_width") {
    spe_charge_width = value;
  } else if (param == "pe_time_spread") {
    if (value < 0.0) {
      RAT::Log::Die("WaveformAnalysisNPEEstimation: pe_time_spread must not be negative.");
    }
    pe_time_spread = value;
  } else {
    WaveformAnalyzerBase::SetD(param, value);
  }
}

void WaveformAnalysisNPEEstimation::SetI(std::string param, int value) {
  if (param == "max_pes") {
    max_pes = static_cast<size_t>(value);
  } else if (param == "spe_charge_from_pmtcharge") {
    spe_charge_from_pmtcharge = (value != 0);
  } else {
    WaveformAnalyzerBase::SetI(param, value);
  }
}

std::string WaveformAnalysisNPEEstimation::GetAnalyzerName() const { return output_name; }

std::pair<double, double> WaveformAnalysisNPEEstimation::SPEForModel(const std::string& model_name) {
  std::map<std::string, std::pair<double, double>>::const_iterator cached = fModelSPECache.find(model_name);
  if (cached != fModelSPECache.end()) {
    return cached->second;
  }

  // The simulation falls back to the default entry for an unlisted model; match it.
  std::unique_ptr<PDFPMTCharge> pdf;
  try {
    pdf = std::make_unique<PDFPMTCharge>(model_name);
  } catch (DBNotFoundError&) {
    try {
      pdf = std::make_unique<PDFPMTCharge>();
    } catch (DBNotFoundError&) {
      RAT::Log::Die("WaveformAnalysisNPEEstimation: No PMTCHARGE entry for " + model_name + " and no default entry.");
    }
  }
  const std::pair<double, double> spe(pdf->GetMean(), pdf->GetStdDev());
  info << "WaveformAnalysisNPEEstimation: Single-PE charge for " << model_name << " from PMTCHARGE: mean " << spe.first
       << " pC, width " << spe.second << " pC" << newline;

  fModelSPECache[model_name] = spe;
  return spe;
}

void WaveformAnalysisNPEEstimation::DoAnalysis(DS::DigitPMT* digitpmt, const std::vector<UShort_t>& digitWfm) {
  const std::vector<std::string> fitters = digitpmt->GetFitterNames();
  if (std::find(fitters.begin(), fitters.end(), seed_analyzer) == fitters.end()) {
    if (!seed_missing_warned) {
      warn << "WaveformAnalysisNPEEstimation: no result named \"" << seed_analyzer
           << "\" to read. Does that analyzer run first in the macro? Channels without one are left empty." << newline;
      seed_missing_warned = true;
    }
    return;
  }
  DS::WaveformAnalysisResult* seed = digitpmt->GetOrCreateWaveformAnalysisResult(seed_analyzer);
  DS::WaveformAnalysisResult* fit_result = digitpmt->GetOrCreateWaveformAnalysisResult(GetAnalyzerName());

  DS::Run* run = DS::RunStore::GetCurrentRun();
  std::pair<double, double> spe(spe_charge, spe_charge_width);
  if (spe_charge_from_pmtcharge) {
    spe = SPEForModel(run->GetPMTInfo()->GetModelNameByID(digitpmt->GetID()));
  }
  // Same per-channel calibration the seed analyzers apply to their charges
  double gain_calibration = run->GetChannelStatus()->GetChargeScaleByPMTID(digitpmt->GetID());
  const double calibrated_spe_charge = spe.first * gain_calibration;

  const std::vector<Double_t>& seed_times = seed->getTimes();
  const std::vector<Double_t>& seed_charges = seed->getCharges();
  // The seed times have the cable offset subtracted already, and AddPE() subtracts it again.
  const double time_offset = seed->getTimeOffset();

  // A seed without the figure of merit falls back to pe_time_spread.
  bool spread_from_fom = !pe_time_spread_fom.empty() && !seed_times.empty();
  if (spread_from_fom) {
    try {
      seed->getFOM(pe_time_spread_fom, 0);
    } catch (const std::out_of_range&) {
      spread_from_fom = false;
      if (!spread_fom_missing_warned) {
        warn << "WaveformAnalysisNPEEstimation: \"" << seed_analyzer << "\" has no figure of merit \""
             << pe_time_spread_fom << "\". Using pe_time_spread for every pulse." << newline;
        spread_fom_missing_warned = true;
      }
    }
  }

  // The seed is sorted in time, so the PEs of a pulse the seed already split are adjacent.
  size_t i = 0;
  while (i < seed_times.size()) {
    size_t j = i + 1;
    double pulse_charge = seed_charges[i];
    while (j < seed_times.size() && seed_times[j] == seed_times[i]) pulse_charge += seed_charges[j++];
    if (seed_times[i] != WaveformUtil::INVALID) {
      size_t npe = EstimateNPE(pulse_charge, calibrated_spe_charge, spe.second, max_pes);
      double spread = pe_time_spread;
      if (spread_from_fom) {
        // Pulses the seed analyzer could not measure carry an invalid figure of merit.
        const double fom = seed->getFOM(pe_time_spread_fom, i);
        if (fom != WaveformUtil::INVALID && std::isfinite(fom) && fom >= 0.0) spread = fom;
      }
      for (size_t ipe = 0; ipe < npe; ++ipe) {
        // The pulse time is the charge-weighted center of its PEs, so they are spread around it
        // rather than all placed on it.
        double time = seed_times[i] + time_offset;
        if (npe > 1 && spread > 0.0) {
          time += ROOT::Math::normal_quantile((ipe + 0.5) / npe, spread);
        }
        fit_result->AddPE(time, pulse_charge / npe, {{"estimated_npe", static_cast<double>(npe)}});
      }
    }
    i = j;
  }
}

}  // namespace RAT
