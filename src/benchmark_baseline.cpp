#include "aesc_baseline.h"
#include "benchmark_options.h"
#include <filesystem>
#include <iomanip>
#include <memory>
#include <numeric>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc<2 || argc>9) {
        std::cerr<<"Usage: benchmark_baseline dataset [eps_csv=0.05,0.01,0.005] [threads=1]"
                 <<" [repeats=3] [data_root=datasets] [omega=128] [gamma=10] [warmups=1]\n";
        return 1;
    }
    try {
        const auto eps=bench::split(argc>2?argv[2]:"0.05,0.01,0.005");
        if (bench::positive(argc>3?argv[3]:"1") != 1)
            throw std::invalid_argument("baseline requires threads=1");
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
        omp_set_dynamic(0);
        omp_set_num_threads(1);
        const auto directory=std::filesystem::path(config.strFolder)/config.strGraph;
        AESCGraph graph((directory/"graph.txt").string(),(directory/("sorted_eigens_"+std::to_string(config.omega)+".txt")).string(),config.omega);
        config.delta=1.0/graph.n;
        std::cout<<std::setprecision(10)<<std::unitbuf;
        std::cout<<"sampler\tdataset\tepsilon\tthreads\trepeat\tseconds\tchecksum\n";
        for(const auto& e:eps) {
            config.epsilon=bench::number(e);
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
                if(run>=warmups) std::cout<<"mc2way-baseline"<<'\t'<<config.strGraph<<'\t'<<config.epsilon<<'\t'<<1<<'\t'<<run-warmups<<'\t'<<elapsed<<'\t'<<checksum<<'\n';
            }
        }
        graph.free();
    } catch(const std::exception& error) { std::cerr<<"benchmark: "<<error.what()<<'\n'; return 1; }
}
