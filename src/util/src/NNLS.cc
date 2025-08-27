#include <RAT/NNLS.hh>
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "TDecompQRH.h"
#include "TDecompSVD.h"

namespace RAT {
namespace Math {

namespace {
// Build A[:, P] as a dense matrix with columns in 'Pidx'.
static TMatrixD ColSubset(const TMatrixD& A, const std::vector<int>& Pidx) {
  const int m = A.GetNrows();
  const int k = static_cast<int>(Pidx.size());
  TMatrixD AP(m, k);
  for (int j = 0; j < k; ++j) {
    const int col = Pidx[j];
    for (int i = 0; i < m; ++i) AP(i, j) = A(i, col);
  }
  return AP;
}

// Solve least squares AP * z ~= b using QR; fallback to SVD on failure.
static TVectorD SolveLeastSquares(const TMatrixD& AP, const TVectorD& b) {
  TVectorD z(AP.GetNcols());
  if (AP.GetNcols() == 0) {
    z.Zero();
    return z;
  }
  TDecompQRH qr(AP);
  Bool_t ok = kFALSE;
  z = qr.Solve(b, ok);
  if (!ok) {
    TDecompSVD svd(AP);
    z = svd.Solve(b, ok);
    if (!ok) throw std::runtime_error("NNLS: least-squares solve failed (QR & SVD).");
  }
  return z;
}

// Infinity norm helper
static double InfNorm(const TVectorD& v) {
  double a = 0.0;
  for (int i = 0; i < v.GetNrows(); ++i) a = std::max(a, std::abs(v[i]));
  return a;
}

}  // namespace

TVectorD NNLS_LawsonHanson(const TMatrixD& A, const TVectorD& b, double tol, int max_outer) {
  const int m = A.GetNrows();
  const int n = A.GetNcols();
  if (b.GetNrows() != m) throw std::invalid_argument("NNLS: A and b shape mismatch.");

  // Precompute A^T
  TMatrixD AT(TMatrixD::kTransposed, A);

  // State
  TVectorD x(n);
  x.Zero();
  TVectorD r = b;       // r = b - A*x, starts at b
  TVectorD w = AT * r;  // "dual" / gradient

  // Sets: P = passive (in basis), Z = remaining (complement). We track via booleans.
  std::vector<char> inP(n, 0);  // 0/1 flags
  std::vector<int> Pidx;        // indices currently in passive set

  // Tolerances & iteration caps
  if (tol < 0.0) tol = 1e-12 * std::max(1.0, InfNorm(w));
  if (max_outer <= 0) max_outer = 3 * std::max(1, n);

  int outer_iter = 0;

  // Outer loop: add most positive w_j to passive set until KKT satisfied.
  while (true) {
    // Find j in Z with max w_j
    int t = -1;
    double wmax = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < n; ++j)
      if (!inP[j]) {
        if (w[j] > wmax) {
          wmax = w[j];
          t = j;
        }
      }

    // KKT satisfied?
    if (t < 0 || wmax <= tol || outer_iter++ >= max_outer) break;

    // Move t from Z to P
    inP[t] = 1;
    Pidx.clear();
    Pidx.reserve(n);
    for (int j = 0; j < n; ++j)
      if (inP[j]) Pidx.push_back(j);

    // Inner loop: solve unconstrained LS on P, then enforce nonnegativity via step back if needed.
    while (true) {
      // Solve z_P = argmin ||A_P z - b||, with z_Z := 0
      TMatrixD AP = ColSubset(A, Pidx);
      TVectorD zP = SolveLeastSquares(AP, b);

      // Form z in full space
      TVectorD z(n);
      z.Zero();
      for (int k = 0; k < (int)Pidx.size(); ++k) z[Pidx[k]] = zP[k];

      // If all z_P >= 0, accept and update x
      bool allpos = true;
      for (int k = 0; k < (int)Pidx.size(); ++k)
        if (zP[k] <= 0.0) {
          allpos = false;
          break;
        }
      if (allpos) {
        x = z;
        break;
      }

      // Otherwise, step towards z until some component hits zero; remove any nonpositive from P.
      double alpha = 1.0;
      for (int j = 0; j < n; ++j) {
        if (inP[j] && z[j] <= 0.0) {
          const double denom = x[j] - z[j];
          if (denom > 0.0) alpha = std::min(alpha, x[j] / denom);
        }
      }
      // Update x <- x + alpha*(z - x)
      for (int j = 0; j < n; ++j) x[j] = x[j] + alpha * (z[j] - x[j]);

      // Drop any indices that are ~zero (numerically) from P
      const double tiny = 0.0;  // exact zero is fine; x should not go slightly negative
      bool removed = false;
      for (int j = 0; j < n; ++j) {
        if (inP[j] && x[j] <= tiny) {
          inP[j] = 0;
          removed = true;
        }
      }
      if (removed) {
        Pidx.clear();
        for (int j = 0; j < n; ++j)
          if (inP[j]) Pidx.push_back(j);
        if (Pidx.empty()) break;  // nothing left to solve on
      }
    }  // end inner loop

    // Update residual and w for next outer iteration
    r = b - (A * x);
    w = AT * r;
  }

  // Final cleanup: ensure nonnegativity
  for (int j = 0; j < n; ++j)
    if (x[j] < 0.0) x[j] = 0.0;
  return x;
}

double NNLS_ResidualNorm(const TMatrixD& A, const TVectorD& b, const TVectorD& x) {
  TVectorD r = b - (A * x);
  double sum = 0.0;
  for (int i = 0; i < r.GetNrows(); ++i) sum += r[i] * r[i];
  return std::sqrt(sum);
}

}  // namespace Math
}  // namespace RAT
