// fsutanto@umich.edu
// akindele1@llnl.gov
// March 9, 2020

// Beta decay spectrum is largely borrowed from geant4.10.4
// Only allowed transition is considered (constant shape function)

#include <CLHEP/Vector/LorentzVector.h>

#include <G4Electron.hh>
#include <G4Event.hh>
#include <G4Gamma.hh>
#include <G4Neutron.hh>
#include <G4ParticleDefinition.hh>
#include <G4PhysicalConstants.hh>
#include <G4PrimaryParticle.hh>
#include <G4PrimaryVertex.hh>
#include <G4ThreeVector.hh>
#include <G4UnitsTable.hh>
#include <RAT/Factory.hh>
#include <RAT/GLG4PosGen.hh>
#include <RAT/GLG4StringUtil.hh>
#include <RAT/GLG4TimeGen.hh>
#include <RAT/Log.hh>
#include <RAT/NGen.hh>
#include <Randomize.hh>
#include <numeric>
#include <string>

#undef DEBUG

/////////////////////////////////////////////////////////////////
// Global variables
/////////////////////////////////////////////////////////////////

// Branching ratio array and Q value array (end point energy)
double qArrN[4] = {2.741, 3.301, 3.595, 4.126};     // MeV
double brArrN[4] = {0.0735, 0.499, 0.0057, 0.375};  // branching ratio taking into account decay to produce neutrons
double cdfN[4] = {0., 0., 0., 0.};                  // will be filled later based on branching ratio

namespace RAT {

NGen::NGen() : stateStr(""), isotope(17), posGen(0) {
  // As in the combo generator, use a default time generator if the
  // user does not supply one.
  timeGen = new GLG4TimeGen_Poisson();

  // Initialize the decay particles.
  neutron = G4Neutron::Neutron();
  electron = G4Electron::Electron();

  // Charge of final state of nucleus
  Z = 7;

  // A of parent
  A = 17;

  // finestructureconstant*Z
  alphaZ = fine_structure_const * Z;

  // Nuclear radius in units of hbar/m_e/c
  Rnuc = 0.5 * fine_structure_const * std::pow(A, 0.33333);

  // Electron screening potential in units of electrom mass
  V0 = 1.13 * fine_structure_const * fine_structure_const * std::pow(std::abs(Z), 1.33333);

  // S
  gamma0 = std::sqrt(1. - alphaZ * alphaZ);

  // cdf (cumulativ distribution fnction) for branching ratio
  pdfNow = 0.;
  sumBr = accumulate(std::begin(brArrN), std::end(brArrN), 0., std::plus<double>());
  for (size_t j = 0; j < sizeof(brArrN) / sizeof(brArrN[0]); j++) {
    pdfNow += brArrN[j] / sumBr;
    cdfN[j] = pdfNow;
  }
}

NGen::~NGen() {
  delete timeGen;
  delete posGen;
}

void NGen::GenerateEvent(G4Event *event) {
  // Generate the position of the isotope.  Note that, for now, we
  // don't change the position of the isotope as it decays.
  G4ThreeVector position;
  posGen->GeneratePosition(position);

  // number of particles
  G4int numElectron = 1;
  G4int numNeutron = 1;

  /////////////////////////////////////////////////////////////////////
  // For beta particle
  /////////////////////////////////////////////////////////////////////

  for (int i = 0; i < numElectron; i++) {
    // time
    double t = 0.0;
    G4double time = NextTime() + t;

    // direction random
    G4double cost = 1. - 2. * G4UniformRand();  // cos theta
    G4double theta = acos(cost);
    G4double phi = 2. * CLHEP::pi * G4UniformRand();
    G4double px = sin(theta) * cos(phi);
    G4double py = sin(theta) * sin(phi);
    G4double pz = cost;

    // energy
    size_t idxNow = 0;
    G4double randNow = G4UniformRand();
    for (size_t j = 0; j < sizeof(brArrN) / sizeof(brArrN[0]); j++) {
      if (randNow < cdfN[j]) {
        idxNow = j;
        break;
      }
    }
    double e0 = qArrN[idxNow];  // end point energy of beta particle
    SetUpBetaSpectrumSampler(e0);

    double erg = qArrN[idxNow] * spectrumSampler->shoot(G4Random::getTheEngine()) * CLHEP::MeV;
    double restErg = electron->GetPDGMass() * CLHEP::MeV;
    erg = sqrt(erg * erg + erg * 2.0 * restErg);

    // create vertex
    G4PrimaryVertex *vertex = new G4PrimaryVertex(position, time);
    G4PrimaryParticle *particle = new G4PrimaryParticle(electron, px * erg, py * erg, pz * erg);

    // mass
    particle->SetMass(electron->GetPDGMass());

    // add particle to vertex
    vertex->SetPrimary(particle);

    // add vertex to event
    event->AddPrimaryVertex(vertex);
  }

  /////////////////////////////////////////////////////////////////////
  // For neutron
  /////////////////////////////////////////////////////////////////////

  for (int i = 0; i < numNeutron; i++) {
    // time
    double t = 0.0;
    G4double time = NextTime() + t;

    // direction random
    G4double cost = 1. - 2. * G4UniformRand();  // cos theta
    G4double theta = acos(cost);
    G4double phi = 2. * CLHEP::pi * G4UniformRand();
    G4double px = sin(theta) * cos(phi);
    G4double py = sin(theta) * sin(phi);
    G4double pz = cost;

    // energy
    double erg = 0.025 * CLHEP::eV;
    double restErg = neutron->GetPDGMass() * CLHEP::MeV;
    erg = sqrt(erg * erg + erg * 2.0 * restErg);

    // create vertex
    G4PrimaryVertex *vertex = new G4PrimaryVertex(position, time);
    G4PrimaryParticle *particle = new G4PrimaryParticle(neutron, px * erg, py * erg, pz * erg);

    // mass
    particle->SetMass(neutron->GetPDGMass());

    // add particle to vertex
    vertex->SetPrimary(particle);

    // add vertex to event
    event->AddPrimaryVertex(vertex);
  }
}

void NGen::ResetTime(double offset) {
  double eventTime = timeGen->GenerateEventTime();
  nextTime = eventTime + offset;
#ifdef DEBUG
  debug << "RAT::NGen::ResetTime:"
        << " eventTime=" << G4BestUnit(eventTime, "Time") << ", offset=" << G4BestUnit(offset, "Time")
        << ", nextTime=" << G4BestUnit(nextTime, "Time") << newline;
#endif
}

void NGen::SetState(G4String state) {
#ifdef DEBUG
  debug << "RAT::NGen::SetState called with state='" << state << "'" << newline;
#endif

  // Break the argument to the this generator into sub-std::strings
  // separated by ":".
  state = util_strip_default(state);
  std::vector<std::string> parts = util_split(state, ":");
  size_t nArgs = parts.size();

#ifdef DEBUG
  debug << "RAT::NGen::SetState: nArgs=" << nArgs << newline;
#endif

  try {
    if (nArgs >= 3) {
      // The last argument is an optional time generator
      delete timeGen;
      timeGen = 0;  // In case of exception in next line
      timeGen = GlobalFactory<GLG4TimeGen>::New(parts[2]);
    }

    if (nArgs >= 2) {
      // The first argument is the Californium isotope.  At
      // present, only N17is supported.
      isotope = util_to_int(parts[0]);

      if (isotope != 17) {
        warn << "RAT::NGen::SetState: Only N 17 is supported" << newline;
      }

      // The second argument is a position generator.
      delete posGen;
      posGen = 0;
      posGen = GlobalFactory<GLG4PosGen>::New(parts[1]);
    } else {
      G4Exception(__FILE__, "Invalid Parameter", FatalException,
                  ("NGen syntax error: '" + state + "' does not have a position generator").c_str());
    }

    stateStr = state;  // Save for later call to GetState()
  } catch (FactoryUnknownID &unknown) {
    warn << "Unknown generator \"" << unknown.id << "\"" << newline;
  }
}

G4String NGen::GetState() const { return stateStr; }

void NGen::SetTimeState(G4String state) {
  if (timeGen)
    timeGen->SetState(state);
  else
    warn << "NGen error: Cannot set time state, no time generator selected" << newline;
}

G4String NGen::GetTimeState() const {
  if (timeGen)
    return timeGen->GetState();
  else
    return G4String("NGen error: no time generator selected");
}

void NGen::SetPosState(G4String state) {
  if (posGen)
    posGen->SetState(state);
  else
    warn << "NGen error: Cannot set position state, no position generator "
            "selected"
         << newline;
}

G4String NGen::GetPosState() const {
  if (posGen)
    return posGen->GetState();
  else
    return G4String("NGen error: no position generator selected");
}

/////////////////////////////////////////////////////////////////////////////////////
// GEANT4 BETA DECAY MODEL
/////////////////////////////////////////////////////////////////////////////////////

void NGen::SetUpBetaSpectrumSampler(G4double &e0) {
  // Array to store spectrum pdf
  G4int npti = 100;
  spectrumSampler = 0;
  G4double *pdf = new G4double[npti];

  // end point energy in unit of electron mass
  // recall that qArrN is in MeV, so divide with e mass in MeV as well
  e0 /= 0.511;
  G4double e;  // Total electron energy in units of electron mass
  G4double p;  // Electron momentum in units of electron mass
  G4double f;  // Spectral shape function

  for (G4int ptn = 0; ptn < npti; ptn++) {
    // Calculate simple phase space
    e = 1. + e0 * (G4double(ptn) + 0.5) / G4double(npti);
    p = std::sqrt(e * e - 1.);
    f = p * e * (e0 - e + 1.) * (e0 - e + 1.);

    // Apply Fermi factor to get allowed shape
    f *= FermiFunction(e);

    // Set the pdf according to the branching ratio
    pdf[ptn] = f;
  }

  spectrumSampler = new G4RandGeneral(pdf, npti);
  delete[] pdf;
}

G4double NGen::FermiFunction(G4double &W) {
  // Calculate the relativistic Fermi function.  Argument W is the
  // total electron energy in units of electron mass.

  G4double Wprime;
  if (Z < 0) {
    Wprime = W + V0;
  } else {
    Wprime = W - V0;
    if (Wprime <= 1.00001) Wprime = 1.00001;
  }

  G4double p_e = std::sqrt(Wprime * Wprime - 1.);
  G4double eta = alphaZ * Wprime / p_e;
  G4double epieta = std::exp(pi * eta);
  G4double realGamma = tgamma(2. * gamma0 + 1);
  G4double mod2Gamma = ModSquared(gamma0, eta);

  // Fermi function
  G4double factor1 = 2 * (1 + gamma0) * mod2Gamma / realGamma / realGamma;
  G4double factor2 = epieta * std::pow(2 * p_e * Rnuc, 2 * (gamma0 - 1));

  // Electron screening factor
  G4double factor3 = (Wprime / W) * std::sqrt((Wprime * Wprime - 1.) / (W * W - 1.));

  return factor1 * factor2 * factor3;
}

G4double NGen::ModSquared(G4double &re, G4double &im) {
  // Calculate the squared modulus of the Gamma function
  // with complex argument (re, im) using approximation B
  // of Wilkinson, Nucl. Instr. & Meth. 82, 122 (1970).
  // Here, choose N = 1 in Wilkinson's notation for approximation B

  G4double factor1 = std::pow((1 + re) * (1 + re) + im * im, re + 0.5);
  G4double factor2 = std::exp(2 * im * std::atan(im / (1 + re)));
  G4double factor3 = std::exp(2 * (1 + re));
  G4double factor4 = 2. * pi;
  G4double factor5 = std::exp((1 + re) / ((1 + re) * (1 + re) + im * im) / 6);
  G4double factor6 = re * re + im * im;
  return factor1 * factor4 * factor5 / factor2 / factor3 / factor6;
}

}  // namespace RAT
