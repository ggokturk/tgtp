#include "graph.h"
#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

// Exact all-edge spanning centrality via the matrix-tree identity
// SC(e) = effective resistance between the endpoints (unit-weight edges).
// The grounded Laplacian B (vertex 0 removed) is SPD; with G = B^{-1},
// R_eff(0,v) = G[v,v] and R_eff(u,v) = G[u,u] + G[v,v] - 2 G[u,v] for u,v > 0.
// Output: headerless native-endian floats in CSR adjacency order (both
// directions of every undirected edge), matching the tgtp executable.
int main(int argc,char** argv) {
    if(argc!=3) {
        std::cerr<<"Usage: exact_sc graph.txt outfile.bin\n";return 1;
    }
    try {
        fuser::graph_t<> graph;
        graph.load_txt(argv[1]);
        if(graph.n<2) throw std::invalid_argument("graph needs at least two vertices");
        for(uint32_t v=0;v<graph.n;++v) if(!graph.degree(v)) throw std::invalid_argument("isolated vertex");
        const Eigen::Index k=static_cast<Eigen::Index>(graph.n-1);
        Eigen::MatrixXd B=Eigen::MatrixXd::Zero(k,k);
        for(uint32_t u=1;u<graph.n;++u) {
            B(u-1,u-1)+=static_cast<double>(graph.degree(u));
            for(auto* it=graph.begin(u);it<graph.end(u);++it) {
                const uint32_t v=*it;
                if(v) B(u-1,v-1)-=1.0;
            }
        }
        const Eigen::LLT<Eigen::MatrixXd> llt(B);
        if(llt.info()!=Eigen::Success) throw std::runtime_error("grounded Laplacian is singular; graph must be connected");
        const Eigen::MatrixXd G=llt.solve(Eigen::MatrixXd::Identity(k,k));
        if(!G.allFinite()) throw std::runtime_error("nonfinite pseudoinverse");
        std::ofstream out(argv[2],std::ios::binary);
        if(!out) throw std::runtime_error("cannot open output file: "+std::string(argv[2]));
        for(uint32_t u=0;u<graph.n;++u) {
            for(auto* it=graph.begin(u);it<graph.end(u);++it) {
                const uint32_t v=*it;
                double r;
                if(u==0) r=G(v-1,v-1);
                else if(v==0) r=G(u-1,u-1);
                else r=G(u-1,u-1)+G(v-1,v-1)-2.0*G(u-1,v-1);
                if(!(r>=0.0) || r>1.0+1e-9) throw std::runtime_error("effective resistance out of range");
                const float value=static_cast<float>(r);
                out.write(reinterpret_cast<const char*>(&value),sizeof(value));
            }
        }
        out.close();
        if(!out) throw std::runtime_error("failed to write output file");
        graph.free();
    } catch(const std::exception& error) { std::cerr<<"exact_sc: "<<error.what()<<'\n'; return 1; }
}
