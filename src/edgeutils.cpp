#include "dataset_download.h"
#include "eigenprep.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
using namespace std;

int usage() {
	cerr << "usage:\n"
	     << "  edgeutils -d <urls.txt> [output root]\n"
	     << "  edgeutils -c <dataset folder> <omega>\n"
	     << "  edgeutils -call <datasets root> <omega>\n";
	return 1;
}

int main(int argc, char** argv) {
	if (argc < 2)
		return usage();
	const string method = string(argv[1]);
	try {
		if (method == "-d") {
			if (argc < 3 || argc > 4)
				return usage();
			dataset::download_datasets(string(argv[2]),
			                           filesystem::path(argc > 3 ? argv[3] : "."));
		}
		else if (method == "-c") {
			if (argc != 4)
				return usage();
			dataset::eigenprep_folder(filesystem::path(argv[2]),
			                          (size_t)atol(argv[3]));
		}
		else if (method == "-call") {
			if (argc != 4)
				return usage();
			dataset::eigenprep_batch(filesystem::path(argv[2]), (size_t)atol(argv[3]));
		}
		else {
			return usage();
		}
	} catch (const std::exception& e) {
		cerr << "error: " << e.what() << endl;
		return 1;
	}
	return 0;
}
