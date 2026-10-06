#include "aesc.h"
#include "benchmark_options.h"
#include <filesystem>
#include <iomanip>
#include <memory>
#include <numeric>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc<2 || argc>9) {
        std::cerr<<"Usage: benchmark dataset [eps_csv=0.05,0.01,0.005] [threads_csv=1,2,4,8,16]"
                 <<" [repeats=3] [data_root=datasets] [omega=128] [gamma=10] [warmups=1]\n";
        return 1;
    }
    try {
        const auto eps=bench::split(argc>2?argv[2]:"0.05,0.01,0.005");
        const auto threads=bench::split(argc>3?argv[3]:"1,2,4,8,16");
        const auto repeats=bench::positive(argc>4?argv[4]:"3");
        const double warmup_arg=bench::number(argc>8?argv[8]:"1");
        if(warmup_arg<0 || warmup_arg>INT32_MAX || warmup_arg!=std::floor(warmup_arg)) throw std::invalid_argument("invalid warmups");
        const unsigned warmups=static_cast<unsigned>(warmup_arg);
        AESCConfig config{};
        config.strFolder=argc>5?argv[5]:"datasets";
        config.strGraph=argv[1];
        config.omega=bench::positive(argc>6?argv[6]:"128");
        config.gamma=bench::positive(argc>7?argv[7]:"10");
        for(const auto& e:eps) if(!(bench::number(e)>0 && bench::number(e)<1)) throw std::invalid_argument("epsilon must be between zero and one");
        for(const auto& t:threads) bench::positive(t);
        const auto directory=std::filesystem::path(config.strFolder)/config.strGraph;
        AESCGraph graph((directory/"graph.txt").string(),(directory/("sorted_eigens_"+std::to_string(config.omega)+".txt")).string(),config.omega);
        config.delta=1.0/graph.n;
        omp_set_dynamic(0);
        std::cout<<std::setprecision(10)<<std::unitbuf;
        std::cout<<"sampler\tdataset\tepsilon\tthreads\trepeat\tseconds\tchecksum\n";
        for(const auto& e:eps) for(const auto& t:threads) {
            config.epsilon=bench::number(e);
            omp_set_num_threads(bench::positive(t));
            for(uint64_t run=0;run<uint64_t(warmups)+repeats;++run) {
                AESC algorithm(graph,config);
                const double start=omp_get_wtime();
                std::unique_ptr<float[]> scores(algorithm.tgtp());
                const double elapsed=omp_get_wtime()-start;
                double checksum=0;
                for(size_t j=0;j<graph.m;++j) {
                    if(!std::isfinite(scores[j])) throw std::runtime_error("nonfinite score");
                    checksum+=scores[j];
                }
                if(run>=warmups) std::cout<<"seed32-grid-multiply-high"<<'\t'<<config.strGraph<<'\t'<<config.epsilon<<'\t'<<t<<'\t'<<run-warmups<<'\t'<<elapsed<<'\t'<<checksum<<'\n';
            }
        }
        graph.free();
    } catch(const std::exception& error) { std::cerr<<"benchmark: "<<error.what()<<'\n'; return 1; }
}
