#pragma once
#include <string>
#include <memory>
#include <algorithm>
#include <immintrin.h>
#include <cstdint>
#include <utility>
#include <cfloat>
#include <cmath>
#include <climits>
#include <iostream>
#include "graph.h"
#include <unordered_set>

using namespace std;

#define RWRATIO 25
#include <fstream>

namespace fuser {
	// SplitMix32 (16/15/15 mixer), eight unsigned 32-bit generators.
	// https://github.com/bryc/code/blob/master/jshash/PRNGs.md#splitmix32
	inline __m256i splitmix32(__m256i& state) {
		state = _mm256_add_epi32(state, _mm256_set1_epi32(0x9e3779b9));
		__m256i z = _mm256_xor_si256(state, _mm256_srli_epi32(state, 16));
		z = _mm256_mullo_epi32(z, _mm256_set1_epi32(0x21f0aaad));
		z = _mm256_xor_si256(z, _mm256_srli_epi32(z, 15));
		z = _mm256_mullo_epi32(z, _mm256_set1_epi32(0x735a2d97));
		return _mm256_xor_si256(z, _mm256_srli_epi32(z, 15));
	}
	__m256i murmur(__m256i h) {
		auto h2 = _mm256_srli_epi32(h, 16);
		h = _mm256_xor_si256(h, h2);
		h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x85ebca6b));
		h2 = _mm256_srli_epi32(h, 13);
		h = _mm256_xor_si256(h ,h2);
		h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0xc2b2ae35));
		h2 = _mm256_srli_epi32(h, 16);
		h = _mm256_xor_si256(h, h2);
		return h;
	}
}
template <typename T>
struct EigenVecs {
	EigenVecs() {};
	EigenVecs(size_t n, size_t omega) :
		len(n), omega(omega) {
		values = new double[omega];
		vectors = new T[omega * n];
	}
	void load_txt(const string& path) {
		std::ifstream infile(path);
		if (!infile) throw std::runtime_error("cannot open eigenpairs: " + path);
		infile.sync_with_stdio(false);
		for (int i = 0; i < omega; ++i) {
			double val = 0;
			if (!(infile >> values[i]) || !std::isfinite(values[i]))
                throw std::runtime_error("invalid eigenvalue in " + path);
			for (size_t j = 0; j < len; j++) {
				if (!(infile >> vectors[(omega * j) + i]) || !std::isfinite(vectors[(omega * j) + i]))
                    throw std::runtime_error("invalid eigenvector in " + path);
			}
		}
	}
	T* begin(size_t i) {
		return &vectors[i * omega];
	}

	double& operator[](size_t i) {
		return values[i];
	}
	double* values;
	T* vectors;
	size_t len, omega;
	void free() {
		if (values != NULL)
			delete[] values;
		if (vectors != NULL)
			delete[] vectors;
	}
};

struct AESCGraph : fuser::graph_t<uint32_t, uint32_t> {
	EigenVecs<float> eigens;
	AESCGraph(const string& graph_path, const string& eigen_path, size_t omega) {
		this->load_txt(graph_path);
        if (omega < 2 || omega > this->n) throw std::runtime_error("omega must be between 2 and vertex count");
        for (size_t v = 0; v < this->n; ++v)
            if (this->degree(v) == 0) throw std::runtime_error("graph vertex IDs must be contiguous with no isolated vertices");
		this->eigens = EigenVecs<float>(this->n, omega);
		this->eigens.load_txt(eigen_path);
	}
	void free() {
		graph_t::free();
		eigens.free();
	}
};
struct AESCConfig {
	double epsilon, delta;
	string strFolder, strGraph;
	int omega, gamma;
};

union alignas(32) avx2_t {
	__m256 f;
	__m256i i;
	float fs[8];
	uint32_t is[8];
	avx2_t() = default;
	avx2_t(__m256i rhs) :i(rhs) {}
};

struct AESC {
	AESCGraph& g;
	AESCConfig& config;

	AESC(AESCGraph& g, AESCConfig& config) :
		g(g), config(config) {
	}
	float* tgtp() {
		float* pred_secs = new float[g.m];
		int* taus = calculate_taus(config.epsilon / 2.0);
		tgtp_baseline(pred_secs, taus);

		delete[] taus;
		return pred_secs;
	}

	int get_tau(size_t dv, size_t du, const float epsilon, const double lamba, const float delta, const float upsilon) {
		float eps_delta = epsilon - delta > 0.0 ? epsilon - delta : epsilon;
		double a = (std::log)((std::max)((1.0 / du + 1.0 / dv - 2.0 / du / dv - upsilon) / eps_delta / (1.0 - (lamba * lamba)), 1.0));
		double b = (std::log)((double)(1.0) / abs(lamba));
		int tau = (std::max)((std::ceil)((double)(a / b - 1)), (double)1.0);
		return tau % 2 == 0 ? int(tau + 1) : int(tau);
	}

	float get_delta(size_t u, size_t v, double* eigen_t_1) {
		float delta = 0;
		const int omega = g.eigens.omega;
		const auto* uf = g.eigens.begin(u);
		const auto* vf = g.eigens.begin(v);

		for (int i = 1; i < omega - 1; ++i) {
			const auto val = g.eigens[i];
			const auto diff = (uf[i] - vf[i]);
			const float eigen_t_1_i = eigen_t_1[i];
			delta += (diff * diff) * eigen_t_1_i / (1 - val);
		}
		return delta / g.m;
	}
	float get_upsilon(size_t u, size_t v) {
		float upsilon = 0;
		const int omega = g.eigens.omega;
		const auto* uf = g.eigens.begin(u);
		const auto* vf = g.eigens.begin(v);
		for (int i = 1; i < omega - 1; ++i) {
			const auto val = g.eigens[i];
			const auto diff = (uf[i] - vf[i]);
			upsilon += (diff * diff) * (1 + val);
		}
		return upsilon / g.m;
	}

	int* calculate_taus(const float epsilon) {

		int* taus = new int[g.m];
		std::fill(taus, taus + g.m, 0);
		const auto omega = g.eigens.omega;

#pragma omp parallel
		{
			double* __restrict eigen_t_1 = new double[omega];
			double* __restrict eigen_sq = new double[omega];
			for (size_t i = 0; i < omega; i++) {
				eigen_sq[i] = (g.eigens[i] * g.eigens[i]);
			}
#pragma omp for schedule(dynamic,100)
			for (long long u = 0; u < g.n; u++) {
				for (auto* it = g.begin(u); it < g.end(u); it++) {
					const auto v = *it;
					if (v < u) continue;
					float delta = 0;
					float upsilon = get_upsilon(u, v);

					double lamba = g.eigens[1];
					const size_t du = g.degree(u), dv = g.degree(v);
					// The initial bound uses the full spectral weight.
                    int tau = get_tau(du, dv, epsilon, lamba, 0, 0);
					int t = 1;
					std::copy(&eigen_sq[0], &eigen_sq[omega], eigen_t_1);
                    while (true) {
                        delta = get_delta(u, v, eigen_t_1);
                        lamba = g.eigens[g.eigens.omega - 1];
                        const int tauprime = get_tau(du, dv, epsilon, lamba, delta, upsilon);
                        if (t <= tauprime && tauprime < tau) {
                            tau = tauprime;
                            t += 2;
                        } else {
                            break;
                        }
                        for (size_t i = 0; i < omega; i++) {
                            eigen_t_1[i] *= eigen_sq[i];
                        }
                    }

					taus[(it - g.begin(0))] = tau;
					auto lower = std::lower_bound(g.begin(v), g.end(v), u);
					auto dist = distance(g.begin(0), lower);
					taus[dist] = tau;
				}
			}
			delete[] eigen_sq;
			delete[] eigen_t_1;
		}
		return taus;
	}

    // Upstream push_single semantics, retaining reusable CSR frontier buffers.
    void diffuse(size_t src, float* pvec, const size_t* q, size_t q_size,
                 size_t* q_next, size_t& q_next_size, float ds,
                 float* preds, char* visited) {
        for (size_t i = 0; i < q_size; ++i) {
            const auto v = q[i];
            const float residue = pvec[v];
            pvec[v] = 0;
            for (auto* ptr = g.begin(v); ptr < g.end(v); ++ptr) {
                const auto u = *ptr;
                if (!visited[u]) {
                    q_next[q_next_size++] = u;
                    visited[u] = true;
                }
                pvec[u] += residue / float(g.degree(u));
            }
        }
        for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; ++pos) {
            preds[pos] += (pvec[src] - pvec[g.adj[pos]]) / ds;
        }
        for (size_t i = 0; i < q_next_size; ++i) visited[q_next[i]] = false;
    }

	float randomwalk_2way_avx2(
			uint32_t src,
			uint32_t v,
			const size_t num_walk,
			const uint32_t len_walk,
			float* pvec) {
		const int BLOCKSIZE = 8;
		const uint32_t grid_step = UINT32_MAX / num_walk;

		avx2_t x;
		x.f = _mm256_setzero_ps();

		for (size_t i = 0; i < num_walk; i += BLOCKSIZE) {
			avx2_t cur, cur2;
			cur.i = _mm256_set1_epi32(src);
			cur2.i = _mm256_set1_epi32(v);

			// UniformGrid(j, k) = (j + 1) * floor((2^32 - 1) / k).
			__m256i secret = _mm256_mullo_epi32(
				_mm256_add_epi32(
					_mm256_set1_epi32(static_cast<int>(i)),
					_mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1)
				),
				_mm256_set1_epi32(static_cast<int>(grid_step))
			);
			__m256i secret2 = secret;
			// Preserve the original distinct grid seeds for regular sampling.
			__m256i random_state = secret;

			uint32_t len = 0;
			for (; len < len_walk; len++) {
				const __m256i zero = _mm256_setzero_si256();
				const __m256i cmp = _mm256_cmpeq_epi32(secret, zero);
				const __m256i cmp2 = _mm256_cmpeq_epi32(secret2, zero);
				const int mask = _mm256_movemask_epi8(cmp);
				const int mask2 = _mm256_movemask_epi8(cmp2);
				if (mask != 0 || mask2 != 0) [[unlikely]] break;
				avx2_t rnd, ds, rnd2, ds2;

				rnd = _mm256_xor_si256(secret, fuser::murmur(cur.i));
				rnd2 = _mm256_xor_si256(secret2, fuser::murmur(cur2.i));
				ds = _mm256_set_epi32(
					g.degree(cur.is[7]), g.degree(cur.is[6]),
					g.degree(cur.is[5]), g.degree(cur.is[4]),
					g.degree(cur.is[3]), g.degree(cur.is[2]),
					g.degree(cur.is[1]), g.degree(cur.is[0]));
				ds2 = _mm256_set_epi32(
					g.degree(cur2.is[7]), g.degree(cur2.is[6]),
					g.degree(cur2.is[5]), g.degree(cur2.is[4]),
					g.degree(cur2.is[3]), g.degree(cur2.is[2]),
					g.degree(cur2.is[1]), g.degree(cur2.is[0]));

				avx2_t k, k2;
				// k = high32(rnd * degree): contiguous buckets without division.
				const __m256i even = _mm256_mul_epu32(rnd.i, ds.i);
				const __m256i odd = _mm256_mul_epu32(
					_mm256_srli_epi64(rnd.i, 32), _mm256_srli_epi64(ds.i, 32));
				const __m256i even2 = _mm256_mul_epu32(rnd2.i, ds2.i);
				const __m256i odd2 = _mm256_mul_epu32(
					_mm256_srli_epi64(rnd2.i, 32), _mm256_srli_epi64(ds2.i, 32));
				k.i = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xaa);
				k2.i = _mm256_blend_epi32(_mm256_srli_epi64(even2, 32), odd2, 0xaa);

				cur = _mm256_set_epi32(
					g.begin(cur.is[7])[k.is[7]], g.begin(cur.is[6])[k.is[6]],
					g.begin(cur.is[5])[k.is[5]], g.begin(cur.is[4])[k.is[4]],
					g.begin(cur.is[3])[k.is[3]], g.begin(cur.is[2])[k.is[2]],
					g.begin(cur.is[1])[k.is[1]], g.begin(cur.is[0])[k.is[0]]
				);
				cur2 = _mm256_set_epi32(
					g.begin(cur2.is[7])[k2.is[7]], g.begin(cur2.is[6])[k2.is[6]],
					g.begin(cur2.is[5])[k2.is[5]], g.begin(cur2.is[4])[k2.is[4]],
					g.begin(cur2.is[3])[k2.is[3]], g.begin(cur2.is[2])[k2.is[2]],
					g.begin(cur2.is[1])[k2.is[1]], g.begin(cur2.is[0])[k2.is[0]]
				);
				secret = _mm256_mullo_epi32(secret, ds.i);
				secret2 = _mm256_mullo_epi32(secret2, ds2.i);
				x.f = _mm256_add_ps(x.f,
					_mm256_sub_ps(
						_mm256_set_ps(
							pvec[cur.is[7]], pvec[cur.is[6]],
							pvec[cur.is[5]], pvec[cur.is[4]],
							pvec[cur.is[3]], pvec[cur.is[2]],
							pvec[cur.is[1]], pvec[cur.is[0]]
						),
						_mm256_set_ps(
							pvec[cur2.is[7]], pvec[cur2.is[6]],
							pvec[cur2.is[5]], pvec[cur2.is[4]],
							pvec[cur2.is[3]], pvec[cur2.is[2]],
							pvec[cur2.is[1]], pvec[cur2.is[0]]
						)
					)
				);
			}
			// Continue the whole bouquet at len; no further zero checks.
			for (; len < len_walk; len++) {
				avx2_t rnd, rnd2, ds, ds2;
				// Successive draws from each lane's persistent SplitMix32 state.
				rnd.i = fuser::splitmix32(random_state);
				rnd2.i = fuser::splitmix32(random_state);
				ds = _mm256_set_epi32(
					g.degree(cur.is[7]), g.degree(cur.is[6]),
					g.degree(cur.is[5]), g.degree(cur.is[4]),
					g.degree(cur.is[3]), g.degree(cur.is[2]),
					g.degree(cur.is[1]), g.degree(cur.is[0]));
				ds2 = _mm256_set_epi32(
					g.degree(cur2.is[7]), g.degree(cur2.is[6]),
					g.degree(cur2.is[5]), g.degree(cur2.is[4]),
					g.degree(cur2.is[3]), g.degree(cur2.is[2]),
					g.degree(cur2.is[1]), g.degree(cur2.is[0]));

				avx2_t k, k2;
				k.i = _mm256_rem_epu32(rnd.i, ds.i);
				k2.i = _mm256_rem_epu32(rnd2.i, ds2.i);

				cur = _mm256_set_epi32(
					g.begin(cur.is[7])[k.is[7]], g.begin(cur.is[6])[k.is[6]],
					g.begin(cur.is[5])[k.is[5]], g.begin(cur.is[4])[k.is[4]],
					g.begin(cur.is[3])[k.is[3]], g.begin(cur.is[2])[k.is[2]],
					g.begin(cur.is[1])[k.is[1]], g.begin(cur.is[0])[k.is[0]]
				);
				cur2 = _mm256_set_epi32(
					g.begin(cur2.is[7])[k2.is[7]], g.begin(cur2.is[6])[k2.is[6]],
					g.begin(cur2.is[5])[k2.is[5]], g.begin(cur2.is[4])[k2.is[4]],
					g.begin(cur2.is[3])[k2.is[3]], g.begin(cur2.is[2])[k2.is[2]],
					g.begin(cur2.is[1])[k2.is[1]], g.begin(cur2.is[0])[k2.is[0]]
				);
				x.f = _mm256_add_ps(x.f,
					_mm256_sub_ps(
						_mm256_set_ps(
							pvec[cur.is[7]], pvec[cur.is[6]],
							pvec[cur.is[5]], pvec[cur.is[4]],
							pvec[cur.is[3]], pvec[cur.is[2]],
							pvec[cur.is[1]], pvec[cur.is[0]]
						),
						_mm256_set_ps(
							pvec[cur2.is[7]], pvec[cur2.is[6]],
							pvec[cur2.is[5]], pvec[cur2.is[4]],
							pvec[cur2.is[3]], pvec[cur2.is[2]],
							pvec[cur2.is[1]], pvec[cur2.is[0]]
						)
					)
				);
			}
		}
		float sum = 0;
		for (int b = 0; b < BLOCKSIZE; b++) {
			sum += x.fs[b];
		}
		return sum;
	}
	float get_chi(const size_t vi, const size_t vj, float* pvec, int len_walk, float& global_min, float& global_max, float& edge_max) {

		float src_local_max = 0;
		float src_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vi); ptr < g.end(vi); ptr++) {
			const auto val = *ptr;
			src_local_max = (std::max)(src_local_max, pvec[val]);
			src_local_min = (std::min)(src_local_min, pvec[val]);
		}

		float tgt_local_max = 0;
		float tgt_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vj); ptr < g.end(vj); ptr++) {
			const auto val = *ptr;
			tgt_local_max = (std::max)(tgt_local_max, pvec[val]);
			tgt_local_min = (std::min)(tgt_local_min, pvec[val]);
		}

		float chi = global_max + (src_local_max + tgt_local_max) / 2.0 + (len_walk - 1) * edge_max - src_local_min - tgt_local_min - 2 * (len_walk - 1) * global_min;
		return chi;
	}

	float get_edge_max(size_t* q, size_t q_size, float* pvec, const float& global_max, int gamma) {
		int nnz_size = q_size;
		int real_gamma = min(gamma, nnz_size);
		nth_element(q, q + real_gamma - 1, q + q_size, [&](const size_t A, const size_t B) -> bool {
			return pvec[A] > pvec[B]; });

		float gamma_max = 1;
		unordered_set<size_t> set;
		for (int i = 0; i < real_gamma; ++i) {
			auto& node = q[i];
			auto& val = pvec[node];
			gamma_max = (std::min)(gamma_max, val);
			set.insert(node);
		}

		gamma_max = gamma < nnz_size ? gamma_max : 0;
		float edge_max = global_max + gamma_max;

		for (int i = 0; i < real_gamma; ++i) {
			auto& u = q[i];
			for (auto* ptr = g.begin(u); ptr < g.end(u); ptr++) {
				const auto v = *ptr;
				if (!set.contains(v))
					continue;
				edge_max = (std::max)(edge_max, pvec[u] + pvec[v]);
			}
		}
		return edge_max;
	}

	void tgtp_baseline(float* preds, int* taus) {
		std::fill(preds, preds + g.m, 0);
		const int BLOCKSIZE = 8;

#pragma omp parallel
		{
			char* visited = new char[g.n];
			std::fill(visited, visited + g.n, 0);
			size_t* q = new size_t[g.n];
			size_t q_size = 0;
			size_t* q_next = new size_t[g.n];
			size_t q_next_size = 0;
#pragma omp for schedule(dynamic)
			for (int64_t src = 0; src < g.n; src++) {
				auto pvec_ptr = std::make_unique<float[]>(g.n);
				float* pvec = pvec_ptr.get();
				int ell = 0;
				pvec[src] = 1.0;

				auto ds = (float)g.degree(src);

				q_size = 0;
				q_next_size = 0;
				q[q_size++] = src;
				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					preds[pos] = 1.0 / ds;
				}
                float global_min, global_max;

				while (true) {
					global_min = 1;
					global_max = 0;
					double picost = 0, rwcost = 0;

					diffuse(src, pvec, q, q_size, q_next, q_next_size, ds, preds, visited);
					swap(q, q_next);
					q_size = q_next_size;
					q_next_size = 0;

					ell++;
					for (size_t i = 0; i < q_size; i++) {
						const auto v = q[i];
						picost += (double)g.degree(v);
						auto& val = pvec[v];
						global_max = max(global_max, val);
						global_min = min(global_min, val);
					}
					global_min = q_size < g.n ? 0 : global_min;

					for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
						int rest_ell = taus[pos] - ell;
						if (rest_ell <= 0)
							continue;
						double chi = 2.0 * rest_ell * ((double)global_max - global_min);
						double nr = max(ceil(8 * pow(chi, 2) * log(g.m / config.delta) / pow(ds * config.epsilon, 2)), 1.0);
						rwcost += nr;
					}
					if (picost >= RWRATIO * rwcost)
						break;
				}
				float edge_max = get_edge_max(q, q_size, pvec, global_max, config.gamma);
				if (config.gamma == 1)
					edge_max = 2 * global_max;

				for (int64_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					const auto v = g.adj[pos];

					int64_t rest_ell = taus[pos] - ell;
					if (rest_ell <= 0)
						continue;

					float chi = get_chi(src, v, pvec, rest_ell, global_min, global_max, edge_max);
					if (config.gamma <= 0) {
						chi = 2 * rest_ell * global_max;
					}
					const float dse = ds * config.epsilon;
					volatile size_t nr = max(ceil(8 * (chi * chi) * log(g.m / config.delta) / (dse * dse)), 1.0);
					nr += (BLOCKSIZE - (nr % BLOCKSIZE)); //EXPLICITLY COMPLEMENTING TO THE NEXT MULTIPLE OF BLOCKSIZE

					float xij = randomwalk_2way_avx2(src, v, nr, rest_ell, pvec);
#pragma omp atomic
					preds[pos] += xij / (ds * nr);
				}
			}
			delete[] q;
			delete[] q_next;
			delete[] visited;
		}
		float* rev_preds = new float[g.m];
		#pragma omp parallel for
		for(int64_t i=0; i<g.m; i++){
			rev_preds[i]=0;
		}
		#pragma omp parallel for
		for (size_t u = 0; u < g.n; u++) {
			for (size_t pos = g.idx[u]; pos < g.idx[u + 1]; pos++) {
				const auto v = g.adj[pos];
				auto* vu = std::lower_bound(g.begin(v), g.end(v), u);
				size_t vu_pos = distance(g.begin(0), vu);
				rev_preds[vu_pos] = preds[pos];
			}
		}
		#pragma omp parallel for
		for (size_t i = 0; i < g.m; i++)
			preds[i] += rev_preds[i];

		delete[] rev_preds;
	}

};
