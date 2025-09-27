#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

// Include Wasserstein headers with relative path
#include "../Wasserstein_c/src/wasserstein/Wasserstein.hh"

// Use namespace for convenience
using namespace wasserstein;

// Simple wrapper function for EMD computation
double compute_emd_cpp(
    const std::vector<double>& weights1,
    const std::vector<double>& coords1,  // flattened [y1, phi1, y2, phi2, ...]
    const std::vector<double>& weights2,
    const std::vector<double>& coords2,  // flattened [y1, phi1, y2, phi2, ...]
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    // Check dimensions
    size_t n1 = weights1.size();
    size_t n2 = weights2.size();

    if (coords1.size() != n1 * 2) {
        throw std::runtime_error("coords1 size mismatch");
    }
    if (coords2.size() != n2 * 2) {
        throw std::runtime_error("coords2 size mismatch");
    }

    // Create EMD object with YPhiArrayDistance for proper y-phi metric
    EMDFloat64<VectorEvent, YPhiArrayDistance> emd_calculator;
    emd_calculator.set_R(R);
    emd_calculator.set_beta(beta);
    emd_calculator.set_norm(norm);

    // Create events - VectorEvent constructor takes (particles, weights)
    // coords1 and coords2 are already flat: [y1, phi1, y2, phi2, ...]
    VectorEvent<double> ev1(coords1, weights1);  // (particles, weights)
    VectorEvent<double> ev2(coords2, weights2);  // (particles, weights)

    // Compute EMD
    double emd_value = emd_calculator(ev1, ev2);

    return emd_value;
}

// Batch EMD computation for multiple events
std::vector<double> compute_emds_cpp(
    const std::vector<std::vector<double>>& weights_list,
    const std::vector<std::vector<double>>& coords_list,
    double R = 1.0,
    double beta = 1.0,
    bool norm = false,
    bool symmetric = false)
{
    size_t n_events = weights_list.size();
    std::vector<double> results;

    if (symmetric) {
        // Compute upper triangular matrix for symmetric case
        results.resize(n_events * n_events, 0.0);

        for (size_t i = 0; i < n_events; ++i) {
            for (size_t j = i + 1; j < n_events; ++j) {
                double emd_val = compute_emd_cpp(
                    weights_list[i], coords_list[i],
                    weights_list[j], coords_list[j],
                    R, beta, norm
                );
                results[i * n_events + j] = emd_val;
                results[j * n_events + i] = emd_val; // Symmetric
            }
        }
    } else {
        // Compute all pairs
        size_t n1 = weights_list.size();
        size_t n2 = coords_list.size();
        results.resize(n1 * n2, 0.0);

        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                results[i * n2 + j] = compute_emd_cpp(
                    weights_list[i], coords_list[i],
                    weights_list[j], coords_list[j],
                    R, beta, norm
                );
            }
        }
    }

    return results;
}

// Alternative interface that takes events as matrices [pT, y, phi]
double compute_emd_from_matrix(
    const std::vector<std::vector<double>>& event1,  // Each row is [pT, y, phi]
    const std::vector<std::vector<double>>& event2,
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    // Extract weights and coordinates
    std::vector<double> weights1, coords1, weights2, coords2;

    for (const auto& particle : event1) {
        if (particle.size() >= 3) {
            weights1.push_back(particle[0]);  // pT
            coords1.push_back(particle[1]);   // y
            coords1.push_back(particle[2]);   // phi
        }
    }

    for (const auto& particle : event2) {
        if (particle.size() >= 3) {
            weights2.push_back(particle[0]);  // pT
            coords2.push_back(particle[1]);   // y
            coords2.push_back(particle[2]);   // phi
        }
    }

    return compute_emd_cpp(weights1, coords1, weights2, coords2, R, beta, norm);
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    // Define main EMD function
    mod.method("compute_emd_cpp", &compute_emd_cpp);

    // Define batch EMD function
    mod.method("compute_emds_cpp", &compute_emds_cpp);

    // Define matrix interface
    mod.method("compute_emd_from_matrix", &compute_emd_from_matrix);

    // Add documentation
    mod.method("wasserstein_version", []() { return std::string("1.0.0"); });
}