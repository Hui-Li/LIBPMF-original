#include "util.h"
#include "pmf.h"
#include <cstring>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace std;

bool with_weights;

void exit_with_help()
{
	printf(
	"Usage: omp-pmf-train [options] data_dir [model_filename]\n"
	"options:\n"
	"    -s type : set type of solver (default 0)\n"    
	"    	 0 -- CCDR1 with fundec stopping condition\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -p do_predict: do prediction or not (default 0)\n"    
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n"
	);
	exit(1);
}

template < typename T > std::string to_string( const T& n )
{
    std::ostringstream stm;
    stm << n ;
    return stm.str() ;
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	parameter param;   // default values have been set by the constructor 
	with_weights = false;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 'r':
				param.rho = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				param.eta0 = atof(argv[i]);
				break;

			case 'B':
				param.num_blocks = atoi(argv[i]);
				break;

			case 'm':
				param.lrate_method = atoi(argv[i]);
				break;

			case 'u':
				param.betaup = atof(argv[i]);
				break;

			case 'd':
				param.betadown = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0) 
		param.verbose = 1;

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') 
			*p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	return param;
}


void run_ccdr1(parameter &param, const char* input_file_name, const char* model_file_name=NULL){
	smat_t R;
	mat_t W,H;
	testset_t T;

	FILE *model_fp = NULL;

	/*
	if(model_file_name) {
		model_fp = fopen(model_file_name, "wb");
		if(model_fp == NULL)
		{
			fprintf(stderr,"can't open output file %s\n",model_file_name);
			exit(1);
		}
	}*/

	load(input_file_name,R,T, with_weights);
	// W, H  here are k*m, k*n
	initial_col(W, param.k, R.rows);
	initial_col(H, param.k, R.cols);

	string w = "q";
 	string h = "p";

#ifdef WITH_AVG
	printf("global mean %g\n", R.get_global_mean());
	// printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));
	w += "-avg.txt";
	h += "-avg.txt";
#else
	w += ".txt";
	h += ".txt";
#endif
	
	puts("starts!");
	double time = omp_get_wtime();
	ccdr1(R, W, H, T, param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	/*if(model_fp) {
		FILE *model_fp_W = fopen("W.txt", "wb");
		save_mat_t(W,model_fp_W,false);
		fclose(model_fp_W);

		FILE *model_fp_H = fopen("H.txt", "wb");
		save_mat_t(H,model_fp_H,false);
		fclose(model_fp_H);
	}*/
	


	ofstream file1(w.c_str());

        for (int qIndex = 0; qIndex < W[0].size(); qIndex++) {
            for (int kIndex = 0; kIndex < W.size(); kIndex++) {

                file1 << to_string(W[kIndex][qIndex]);
                if(kIndex!=W.size()-1){
                	file1 << ",";
                }else{
                	file1 << endl;
                }
            }
        }
        file1.close();

	ofstream file2(h.c_str());

        for (int pIndex = 0; pIndex < H[0].size(); pIndex++) {
            for (int kIndex = 0; kIndex < H.size(); kIndex++) {
            	file2 << to_string(H[kIndex][pIndex]);
                if(kIndex!=H.size()-1){
                	file2 << ",";
                }else{
                	file2 << endl;
                }
            }
        }
        file2.close();

	return ;
}

#define WITH_AVG

int main(int argc, char* argv[]){
	char input_file_name[1024];
	char model_file_name[1024];
	parameter param = parse_command_line(argc, argv, input_file_name, model_file_name); 

	switch (param.solver_type){
		case CCDR1:
			run_ccdr1(param, input_file_name, model_file_name);
			break;
		default:
			fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
			break;
	}
	return 0;
}

