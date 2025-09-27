#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Include Wasserstein headers with relative path
#include "../Wasserstein_c/src/wasserstein/Wasserstein.hh"

// Use namespace for convenience
using namespace wasserstein;

// Corrected EMD wrapper for y-phi coordinates
double compute_emd_yphi(
    const std::vector<double>& weights1,
    const std::vector<double>& coords1,  // flattened [y1, phi1, y2, phi2, ...]
    const std::vector<double>& weights2,
    const std::vector<double>& coords2,  // flattened [y1, phi1, y2, phi2, ...]
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    size_t n1 = weights1.size();
    size_t n2 = weights2.size();

    if (coords1.size() != n1 * 2) {
        throw std::runtime_error("coords1 size mismatch");
    }
    if (coords2.size() != n2 * 2) {
        throw std::runtime_error("coords2 size mismatch");
    }

    // Create EMD calculator using the predefined types
    // Use DefaultArray2Event which is ArrayEvent<Value, Array2ParticleCollection>
    EMD<double, DefaultArray2Event, YPhiArrayDistance> emd_calculator;
    emd_calculator.set_R(R);
    emd_calculator.set_beta(beta);
    emd_calculator.set_norm(norm);

    // Create events directly with arrays
    // ArrayEvent takes (weight_array, particle_array, size, stride)
    // Note: const_cast is safe since EMD won't modify the arrays
    DefaultArray2Event<double> ev1(
        const_cast<double*>(weights1.data()),
        const_cast<double*>(coords1.data()),
        n1, 2  // size, stride (2 for y,phi)
    );

    DefaultArray2Event<double> ev2(
        const_cast<double*>(weights2.data()),
        const_cast<double*>(coords2.data()),
        n2, 2
    );

    // Compute EMD
    double emd_value = emd_calculator(ev1, ev2);

    return emd_value;
}

// Keep the old function for backward compatibility but have it call the new one
double compute_emd_cpp(
    const std::vector<double>& weights1,
    const std::vector<double>& coords1,
    const std::vector<double>& weights2,
    const std::vector<double>& coords2,
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    return compute_emd_yphi(weights1, coords1, weights2, coords2, R, beta, norm);
}

// Alternative interface that takes events as matrices [pT, y, phi]
double compute_emd_from_matrix(
    const std::vector<double>& event1,  // flattened matrix, row-major: [pT1, y1, phi1, pT2, y2, phi2, ...]
    size_t n1,  // number of particles in event1
    const std::vector<double>& event2,  // flattened matrix
    size_t n2,  // number of particles in event2
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    // Extract weights and coordinates
    std::vector<double> weights1(n1), coords1(n1 * 2);
    std::vector<double> weights2(n2), coords2(n2 * 2);

    for (size_t i = 0; i < n1; ++i) {
        weights1[i] = event1[i * 3];      // pT
        coords1[i * 2] = event1[i * 3 + 1];     // y
        coords1[i * 2 + 1] = event1[i * 3 + 2]; // phi
    }

    for (size_t i = 0; i < n2; ++i) {
        weights2[i] = event2[i * 3];      // pT
        coords2[i * 2] = event2[i * 3 + 1];     // y
        coords2[i * 2 + 1] = event2[i * 3 + 2]; // phi
    }

    return compute_emd_yphi(weights1, coords1, weights2, coords2, R, beta, norm);
}

// Batch EMD computation - using PairwiseEMD with OpenMP support
std::vector<double> compute_emds_cpp(
    const std::vector<std::vector<double>>& events,  // Each event: [pT, y, phi] per particle
    double R = 1.0,
    double beta = 1.0,
    bool norm = false)
{
    size_t n_events = events.size();

    // Convert events to the format needed by PairwiseEMD
    std::vector<DefaultArray2Event<double>> wasserstein_events;
    wasserstein_events.reserve(n_events);

    // We need to store the arrays persistently
    std::vector<std::vector<double>> weights_storage(n_events);
    std::vector<std::vector<double>> coords_storage(n_events);

    for (size_t i = 0; i < n_events; ++i) {
        size_t n_particles = events[i].size() / 3;

        // Extract weights and coords for this event
        weights_storage[i].resize(n_particles);
        coords_storage[i].resize(n_particles * 2);

        for (size_t j = 0; j < n_particles; ++j) {
            weights_storage[i][j] = events[i][j * 3];          // pT
            coords_storage[i][j * 2] = events[i][j * 3 + 1];     // y
            coords_storage[i][j * 2 + 1] = events[i][j * 3 + 2]; // phi
        }
    }

    // Now create the events pointing to our stored data
    for (size_t i = 0; i < n_events; ++i) {
        size_t n_particles = weights_storage[i].size();
        wasserstein_events.emplace_back(
            weights_storage[i].data(),
            coords_storage[i].data(),
            n_particles, 2  // size, stride
        );
    }

    // Get number of threads
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = omp_get_max_threads();
    #endif

    // Create PairwiseEMD object
    PairwiseEMD<EMD<double, DefaultArray2Event, YPhiArrayDistance>> pairwise_emd(
        R, beta, norm,
        num_threads,    // Use all available threads
        -1,             // print_every (negative = auto)
        0,              // verbose = 0 (no output)
        false,          // request_mode
        true,           // store_sym_emds_raw
        false           // throw_on_error
    );

    // Compute all pairwise EMDs
    pairwise_emd.compute(wasserstein_events);

    // Get the results
    const std::vector<double>& emds = pairwise_emd.emds();

    // Convert from flattened symmetric to full matrix
    std::vector<double> results(n_events * n_events, 0.0);

    // The PairwiseEMD stores only unique pairs in flattened symmetric format
    // We need to expand it to full matrix
    size_t k = 0;
    for (size_t i = 0; i < n_events; ++i) {
        for (size_t j = i + 1; j < n_events; ++j) {
            results[i * n_events + j] = emds[k];
            results[j * n_events + i] = emds[k]; // Symmetric
            k++;
        }
    }

    return results;
}

// Version string
std::string wasserstein_version() {
    return "1.0.0";
}

// Module definition for Julia
JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.method("compute_emd_cpp", &compute_emd_cpp);
    mod.method("compute_emd_yphi", &compute_emd_yphi);
    mod.method("compute_emd_from_matrix", &compute_emd_from_matrix);
    mod.method("compute_emds_cpp", &compute_emds_cpp);
    mod.method("wasserstein_version", &wasserstein_version);
}