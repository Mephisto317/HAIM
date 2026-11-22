#include <iostream>
#include <fstream>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<limits.h>
#include<float.h>
#include <assert.h>
#include<memory.h>
#include<math.h>
#include<set>
#include<algorithm>
#include<utility>
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAXA 12

// The improved build jointly evaluates remove/add pairs during CCA search.
// Compile with -DHAIM_LEGACY_SWAP to reproduce the original two-stage move.
#ifndef HAIM_LEGACY_SWAP
#ifndef JOINT_SWAP_CANDIDATE_LIMIT
#define JOINT_SWAP_CANDIDATE_LIMIT 32
#ifndef HAIM_FIXED_SEARCH_PARAMETERS
#define HAIM_ADAPTIVE_JOINT_REMOVE
#endif
#endif
#if JOINT_SWAP_CANDIDATE_LIMIT < 1
#error "JOINT_SWAP_CANDIDATE_LIMIT must be at least 1"
#endif
#ifdef HAIM_ADAPTIVE_JOINT_REMOVE
#define HAIM_JOINT_REMOVE_CAPACITY 64
#else
#define HAIM_JOINT_REMOVE_CAPACITY JOINT_SWAP_CANDIDATE_LIMIT
#endif
#endif

// Restricted exact 2-for-2 exchange. It is an intensification step on top of
// joint 1-swap, so legacy builds keep the original behavior automatically.
#if !defined(HAIM_LEGACY_SWAP) && !defined(HAIM_DISABLE_TWO_SWAP)
#define HAIM_TWO_SWAP_ENABLED
#ifndef HAIM_TWO_SWAP_REMOVE_LIMIT
#define HAIM_TWO_SWAP_REMOVE_LIMIT 12
#ifndef HAIM_FIXED_SEARCH_PARAMETERS
#define HAIM_ADAPTIVE_TWO_SWAP_REMOVE
#endif
#endif
#ifndef HAIM_TWO_SWAP_ADD_LIMIT
#define HAIM_TWO_SWAP_ADD_LIMIT 24
#ifndef HAIM_FIXED_SEARCH_PARAMETERS
#define HAIM_ADAPTIVE_TWO_SWAP_ADD
#endif
#endif
#ifdef HAIM_ADAPTIVE_TWO_SWAP_REMOVE
#define HAIM_TWO_SWAP_REMOVE_CAPACITY 16
#else
#define HAIM_TWO_SWAP_REMOVE_CAPACITY HAIM_TWO_SWAP_REMOVE_LIMIT
#endif
#ifdef HAIM_ADAPTIVE_TWO_SWAP_ADD
#define HAIM_TWO_SWAP_ADD_CAPACITY 32
#else
#define HAIM_TWO_SWAP_ADD_CAPACITY HAIM_TWO_SWAP_ADD_LIMIT
#endif
#ifndef HAIM_TWO_SWAP_TRIGGER
#define HAIM_TWO_SWAP_TRIGGER 250
#endif
#ifndef HAIM_TWO_SWAP_MAX_USES
#define HAIM_TWO_SWAP_MAX_USES 3
#endif
#if HAIM_TWO_SWAP_REMOVE_LIMIT < 2 || HAIM_TWO_SWAP_ADD_LIMIT < 2
#error "2-swap candidate limits must both be at least 2"
#endif
#if HAIM_TWO_SWAP_TRIGGER < 1 || HAIM_TWO_SWAP_MAX_USES < 1
#error "2-swap trigger and maximum uses must both be at least 1"
#endif
#endif

// Adaptive large-neighborhood diversification. Compile with
// -DHAIM_DISABLE_LNS to keep the joint 1-swap build without destroy-repair.
#ifndef HAIM_DISABLE_LNS
#ifndef HAIM_LNS_MIN_DESTROY_PERCENT
#define HAIM_LNS_MIN_DESTROY_PERCENT 2
#endif
#ifndef HAIM_LNS_MAX_DESTROY_PERCENT
#define HAIM_LNS_MAX_DESTROY_PERCENT 5
#endif
#ifndef HAIM_LNS_RCL_SIZE
#define HAIM_LNS_RCL_SIZE 8
#endif
#ifndef HAIM_LNS_MAX_DESTROY_VERTICES
#define HAIM_LNS_MAX_DESTROY_VERTICES 64
#endif
#ifdef HAIM_LNS_PERTURBATIONS_PER_SEARCH
#define HAIM_EXPLICIT_LNS_BUDGET
#else
#define HAIM_LNS_PERTURBATIONS_PER_SEARCH 3
#endif
#ifndef HAIM_LNS_MAX_PERTURBATIONS
#define HAIM_LNS_MAX_PERTURBATIONS 12
#endif
#ifndef HAIM_LNS_MIN_TRIGGER
#define HAIM_LNS_MIN_TRIGGER 100
#endif
#ifndef HAIM_LNS_MAX_TRIGGER
#define HAIM_LNS_MAX_TRIGGER 1000
#endif
#if HAIM_LNS_MIN_DESTROY_PERCENT < 1 || HAIM_LNS_MAX_DESTROY_PERCENT > 100 || \
	HAIM_LNS_MIN_DESTROY_PERCENT > HAIM_LNS_MAX_DESTROY_PERCENT
#error "LNS destroy percentages must satisfy 1 <= min <= max <= 100"
#endif
#if HAIM_LNS_RCL_SIZE < 1 || HAIM_LNS_MAX_DESTROY_VERTICES < 1 || \
	HAIM_LNS_PERTURBATIONS_PER_SEARCH < 1
#error "LNS RCL size, destroy cap, and perturbations per search must be at least 1"
#endif
#if HAIM_LNS_MIN_TRIGGER < 1 || HAIM_LNS_MIN_TRIGGER > HAIM_LNS_MAX_TRIGGER
#error "LNS trigger bounds must satisfy 1 <= min <= max"
#endif
#if HAIM_LNS_MAX_PERTURBATIONS < 3
#error "HAIM_LNS_MAX_PERTURBATIONS must be at least 3"
#endif
#endif

// Periodic search counts every base-search call, including an identity move.
// Set to 0 to reproduce the finite-budget scheduler. Fixed/legacy comparison
// builds retain their previous scheduling automatically.
#ifndef HAIM_PERIODIC_SEARCH
#define HAIM_PERIODIC_SEARCH 1
#endif
#if HAIM_PERIODIC_SEARCH != 0 && HAIM_PERIODIC_SEARCH != 1
#error "HAIM_PERIODIC_SEARCH must be 0 or 1"
#endif
#if HAIM_PERIODIC_SEARCH && !defined(HAIM_FIXED_SEARCH_PARAMETERS) && !defined(HAIM_LEGACY_SWAP)
#define HAIM_PERIODIC_SEARCH_ENABLED 1
#else
#define HAIM_PERIODIC_SEARCH_ENABLED 0
#endif
#ifndef HAIM_LNS_BASE_INTERVAL
#define HAIM_LNS_BASE_INTERVAL 1000
#endif
#if HAIM_LNS_BASE_INTERVAL < 1
#error "HAIM_LNS_BASE_INTERVAL must be at least 1"
#endif
#ifndef HAIM_TWO_SWAP_BASE_INTERVAL
#define HAIM_TWO_SWAP_BASE_INTERVAL 250
#endif
#if HAIM_TWO_SWAP_BASE_INTERVAL < 1
#error "HAIM_TWO_SWAP_BASE_INTERVAL must be at least 1"
#endif

int greedy_limit = 100000;
int cca_limit = 50000;
int smooth = 0;

using namespace std;

double time_limit;

class TimeTest
{
	clock_t start, finish;
public:
	void Init()
	{
		start = clock();
	}
	double GetTime()
	{
		finish = clock();
		return 1.0 * (finish - start) / CLOCKS_PER_SEC;
	}
	bool NeedToStop()
	{
		double nowTime = GetTime();
		return nowTime > time_limit;
	}
} time_test;

typedef struct Column
{
	int score;
	bool config;
	bool is_in_c;
} Column;

typedef struct Row
{
	int weight;
} Row;

Column **cs;//set
Row **rs;//element
int m, n;
int element_size, set_size;
int seed;
int density;
int **uncover;
int **index_uncover;
int **subscore;
int **covered;
int *uncov_num;
int *Remember;
int *remember;
int *rtemp;
int *incumbent_solution;
int *rem_temp;
int *rem_half;
int *col_temp;
int *nncol;
double *tau;
double *eta;
double *te_beta;
int step;
int best_value;
int **ori_row, **ori_col;
int *ori_ncol, *ori_nrow;
int ***row, ***col;
int **ncol, **nrow;
int *best_sol, *k;
double start_time;
int max_k;
int **set_age;
int tabu[MAXA];
int uncover_num;
double real_time;
int happy;
int sad;
int best_temp;
int greedytag = 0;
double omiga = 0.95;
int randstep = 0;
int cntants = 0;
int tpp = 0;
int tag_init = 1;
int randant = 0;
int valuek = 0;
int randnum = 0;
int beta_parma = 1;
double epslion = 0.005;
int ants = 10;
double delta_tau = 0.;
double taumax, taumin;
int k_half = 0;		//k/2
int remove_score = 0;
bool initial_solution_saved = false;
bool incumbent_valid = false;
int incumbent_value = -1;
double incumbent_time = 0.0;
#if defined(HAIM_LNS_STATS) || defined(HAIM_TWO_SWAP_STATS)
long long base_search_calls = 0;
#endif
#ifdef HAIM_LNS_STATS
long long lns_calls = 0;
long long lns_destroyed_vertices = 0;
long long lns_raw_delta_sum = 0;
long long lns_best_improvements = 0;
long long lns_followup_best_improvements = 0;
double lns_cpu_seconds = 0.0;
#endif
#if defined(HAIM_TWO_SWAP_ENABLED) && defined(HAIM_TWO_SWAP_STATS)
long long two_swap_calls = 0;
long long two_swap_combinations = 0;
long long two_swap_moves = 0;
long long two_swap_raw_gain = 0;
long long two_swap_best_improvements = 0;
long long two_swap_after_lns_calls = 0;
double two_swap_cpu_seconds = 0.0;
#endif
#ifdef HAIM_TWO_SWAP_ENABLED
bool two_swap_graph_compatible = true;
#endif



void init();
void find_worst_in_c(int);
#ifndef HAIM_LEGACY_SWAP
bool find_best_joint_swap(int, long long &);
#endif
#ifdef HAIM_TWO_SWAP_ENABLED
bool try_restricted_two_swap(int, bool);
#endif
void remove(int, int);
void find_random_in_vout(int);
void add(int, int);
void reload_weight_score(int);
void reload_age(int);
void CCA_local_search(int);
void CCA_local_searchw(int);
#ifndef HAIM_DISABLE_LNS
void lns_destroy_repair(int);
#endif
void init_aco();
void init_eta_tebeta();
void add_aoc(int, int);
void reload_alltau();
void reload_alltaupro(int);
void reload_bt(int);
void reload_bs(int);
void maxmin();
void ANT_1();
void ANT_2();
void ANT_3();
void ANT_4();
void ANT_5();
void ANT_6();
void ANT_7();
void ANT_8();
void ANT_greedy();
void ANT_greedy();
void ant_empty_greedy(int t);
void remove_greedy(int t, int c);
void add_aoc_greedy(int t, int c);
void add_greedy(int t,int c);
void ant_search_greedy(int t);
void greedy_ls(int t);
void greedy_worst(int t);
void greedy_in_vout(int t);

void ant_empty(int);
void ant_build(int, int);
void ant_search(int);
void ant_searchw(int);
void init_search(int);
void init_searchw(int);
void MMAS_pro();
void NuAnts();
void update_best_sol();
void consider_global_incumbent(int, double);
void restore_global_incumbent();
void free_all();
void free_ori();
void NuAnts_density();
void MMAS_pro_density();




void free_ori()
{
	free(ori_row);
	free(ori_col);
	free(ori_ncol);
	free(ori_nrow);
}

void free_all()
{
	free(row);
	free(col);
	free(ncol);
	free(nrow);
	free(best_sol);
	free(set_age);
	for (int i = 0; i < MAXA; i++)
	{
		free(cs[i]);
		free(rs[i]);

		//        free(uncover[i]);
		//        free(index_uncover[i]);
		delete[] uncover[i];
		delete[] index_uncover[i];

		free(subscore[i]);
		//free(covered[i]);
		//free(Proba[i]);
	}
	//free(best_array);

	free(uncov_num);
	free(Remember);
	free(remember);
	free(rtemp);
	free(incumbent_solution);
	free(rem_temp);
	free(rem_half);
	free(col_temp);
	free(nncol);
	free(tau);
	free(eta);
	free(te_beta);
}


int get_next_int()
{
    int x;
    char buf[32];
    while (true)
    {
        if (scanf("%d", &x) == 1)
        {
            return x;
        }
        if (scanf("%31s", buf) != 1)
        {
            printf("Input reading error in get_next_int!\n");
            exit(1);
        }
        // skip token�� "p"��"edge"��"e"��
    }
}


void build_instance(char *file)
{
	int i, j, h, t;
	if (freopen(file, "r", stdin) == NULL)
	{
		fprintf(stderr, "Cannot open instance: %s\n", file);
		exit(EXIT_FAILURE);
	}

	char first[32] = {0};
	if (scanf("%31s", first) != 1)
	{
		fprintf(stderr, "Cannot read instance header: %s\n", file);
		exit(EXIT_FAILURE);
	}
	if (strcmp(first, "p") == 0)
	{
		char edge_str[32];
		if (scanf("%31s", edge_str) != 1)
		{
			fprintf(stderr, "Cannot read instance type: %s\n", file);
			exit(EXIT_FAILURE);
		}
		n = get_next_int();            
		m = get_next_int();            
//		printf("n=%d, m=%d\n", n, m);
	}
	else
	{
		m = atoi(first);
		n = get_next_int();
//		printf("m=%d, n=%d\n", m, n);
	}
	if (n <= 0 || m < 0)
	{
		fprintf(stderr, "Invalid instance dimensions: n=%d, m=%d\n", n, m);
		exit(EXIT_FAILURE);
	}

	element_size = m + 1;
	set_size = n + 1;

	subscore = (int **)malloc(MAXA * sizeof(int *));
	covered = (int **)malloc(MAXA * sizeof(int *));
	for (i = 0; i < MAXA; i++) subscore[i] = (int *)malloc(set_size * sizeof(int));

	for (i = 0; i < MAXA; i++) covered[i] = (int *)malloc(element_size * sizeof(int));

	for (i = 0; i < MAXA; i++)
	{
		for (j = 0; j < set_size; j++)
		{
			subscore[i][j] = 0;
		}
	}

	for (i = 0; i < MAXA; i++)
	{
		for (j = 0; j < element_size; j++)
		{
			covered[i][j] = 0;
		}
	}

	uncov_num = (int *)malloc(MAXA * sizeof(int));
	for (i = 0; i < MAXA; i++) uncov_num[i] = 0;

	Remember = (int *)malloc(set_size * sizeof(int));
	remember = (int *)malloc(set_size * sizeof(int));
	rtemp = (int *)malloc(set_size * sizeof(int));
	incumbent_solution = (int *)malloc(set_size * sizeof(int));
	// These arrays store vertices, so size them by n rather than by m.
	rem_temp = (int *)malloc(set_size * sizeof(int));
	rem_half = (int *)malloc(set_size * sizeof(int));
	col_temp = (int *)malloc(set_size * sizeof(int));
	nncol = (int *)malloc(element_size * sizeof(int));
	
	for (i = 0; i < element_size; i++)
	{
		nncol[i] = 0;
	}
	for (i = 0; i < set_size; i++)
	{
		col_temp[i] = 0;
	}

	tau = (double *)malloc(set_size * sizeof(double));
	eta = (double *)malloc(set_size * sizeof(double));
	te_beta = (double *)malloc(set_size * sizeof(double));

	cs = (Column **)malloc(MAXA * sizeof(Column *));
	rs = (Row **)malloc(MAXA * sizeof(Row *));
	for (i = 0; i < MAXA; i++)
	{
		cs[i] = (Column *)malloc(set_size * sizeof(Column));
		rs[i] = (Row *)malloc(element_size * sizeof(Row));
	}

	uncover = new int*[MAXA];
	index_uncover = new int*[MAXA];
	for (i = 0; i < MAXA; i++)
	{
		uncover[i] = new int[set_size];
		index_uncover[i] = new int[element_size];
	}


	ori_row = (int **)malloc(m * sizeof(int *));
	ori_ncol = (int *)malloc(m * sizeof(int));
	ori_col = (int **)malloc(n * sizeof(int *));
	ori_nrow = (int *)malloc(n * sizeof(int));
	row = (int ***)malloc(MAXA * sizeof(int **));
	ncol = (int **)malloc(MAXA * sizeof(int *));
	col = (int ***)malloc(MAXA * sizeof(int **));
	nrow = (int **)malloc(MAXA * sizeof(int *));
	set_age = (int **)malloc(MAXA * sizeof(int *));
	k = (int *)malloc(n * sizeof(int));
	best_sol = (int *)malloc(n * sizeof(int));
	for (t = 0; t < MAXA; t++)
	{
		tabu[t] = -1;
		set_age[t] = (int *)malloc(n * sizeof(int));
		for (j = 0; j < n; j++)
			set_age[t][j] = 1;
	}

	for (t = 0; t < MAXA; t++) {
		for (j = 0; j < n; j++)
		{
			cs[t][j].config = 1;//////////////////////////////////// 
			cs[t][j].is_in_c = 0;
			cs[t][j].score = 0;
			best_sol[j] = 0;
		}
	}

	for (i = 0; i < m; i++) 
	{
		ori_ncol[i]=2;
		ori_row[i] = (int *)malloc(ori_ncol[i] * sizeof(int));
		for (h = 0; h < ori_ncol[i]; h++)
		{
			int val = get_next_int();        
			ori_row[i][h] = val - 1;           
		}
#ifdef HAIM_TWO_SWAP_ENABLED
		if (ori_row[i][0] == ori_row[i][1])
			two_swap_graph_compatible = false;
#endif
	}

	for (j = 0; j < n; j++)
		ori_nrow[j] = 0;
	for (i = 0; i < m; i++)
	{
		for (h = 0; h < ori_ncol[i]; h++)
			ori_nrow[ori_row[i][h]]++;
	}
	for (j = 0; j < n; j++)
	{
		ori_col[j] = (int *)malloc(ori_nrow[j] * sizeof(int));
		k[j] = 0;
	}
	for (i = 0; i < m; i++)
	{
		for (h = 0; h < ori_ncol[i]; h++)
		{
			ori_col[ori_row[i][h]][k[ori_row[i][h]]] = i;
			k[ori_row[i][h]]++;
		}
	}


	for (t = 0; t < MAXA; t++)
	{

		ncol[t] = (int *)malloc(m * sizeof(int));
		nrow[t] = (int *)malloc(n * sizeof(int));
		col[t] = (int **)malloc(n * sizeof(int *));
		row[t] = (int **)malloc(m * sizeof(int *));

		for (i = 0; i < m; i++) {
			rs[t][i].weight = 1;/////////////////////////////////////////// 
			ncol[t][i] = ori_ncol[i];
			row[t][i] = (int *)malloc(ori_ncol[i] * sizeof(int));
			for (h = 0; h < ori_ncol[i]; h++)
			{
				row[t][i][h] = ori_row[i][h];
			}
		}

		for (j = 0; j < n; j++)
		{
			cs[t][j].score = ori_nrow[j];
			nrow[t][j] = ori_nrow[j];

			col[t][j] = (int *)malloc(ori_nrow[j] * sizeof(int));
			for (h = 0; h < ori_nrow[j]; h++)
			{
				col[t][j][h] = ori_col[j][h];
			}
		}
	}
	long long x = 1LL * m * n;
	long long y = 0;
	for (j = 0; j < n; j++)
	{
		y += ori_nrow[j];
	}
	double z = x > 0 ? 1.0 * y / x : 0.0;
	printf("m: %d; n: %d; m*n: %lld; density: %f \n",m,n,x,z);
	if(z < 0.01)
	{
		density = 1;
		cca_limit = 1000;
		printf("density < 0.01! \n");
	}
	else
		density = 0;
	free(k);
	free_ori();
}


void init()
{
	int i, j;
	uncover_num = m;
	for (i = 0; i < MAXA; i++) {
		uncov_num[i] = m;
		for (j = 0; j < m; j++)
			index_uncover[i][j] = 1;
	}
}

void Four_stage_solution_structure(int t)
{
	int frcount = 0;
	int i, j;
	int max = -1;
	for (j = 0; j < n; j++)
	{
		if (cs[t][j].is_in_c == 1)
			continue;
		if (cs[t][j].score > max)
		{
			max = cs[t][j].score;
			//cmax = j;
			for (i = 0; i < frcount; i++)
			{
				col_temp[i] = 0;
			}
			frcount = 0;
		}
		if (cs[t][j].score == max)
		{
			col_temp[frcount] = j;
			frcount = frcount + 1;
		}
	}
	if (frcount == 1)
	{
		add_aoc(t, col_temp[0]);
		//SD ++;
	}
	if (frcount > 1)
	{
		int the_set;
		int crcount = 0;
		for (i = 0; i < frcount; i++)
		{
			int max_cnt = 0;
			for (j = 0; j < nrow[t][col_temp[i]]; j++)
			{
				if (index_uncover[t][col[t][col_temp[i]][j]] == -1)
					continue;
				if (ncol[t][col[t][col_temp[i]][j]] == 1)
					max_cnt++;
			}
			if (max_cnt > crcount)
			{
				crcount = max_cnt;
				the_set = col_temp[i];
			}
		}
		if (crcount > 0)
		{
			add_aoc(t, the_set);
		}
		for (i = 0; i < m; i++) {
			nncol[i] = 0;
		}
		if (crcount == 0) {

			int trcount = 0;
			int pse_set;
			for (i = 0; i < frcount; i++) {
				for (j = 0; j < nrow[t][col_temp[i]]; j++) {
					if (index_uncover[t][col[t][col_temp[i]][j]] == -1)
						continue;
					nncol[col[t][col_temp[i]][j]] ++;
				}
			}
			for (i = 0; i < frcount; i++) {
				int pse_cnt = 0;
				for (j = 0; j < nrow[t][col_temp[i]]; j++) {
					if (index_uncover[t][col[t][col_temp[i]][j]] == -1)
						continue;
					if (nncol[col[t][col_temp[i]][j]] == 1)//nncol[]
						pse_cnt++;
				}
				if (pse_cnt > trcount)
				{
					trcount = pse_cnt;
					pse_set = col_temp[i];
				}
			}
			if (trcount > 0) {
				add_aoc(t, pse_set);
			}
			if (trcount == 0) {
				int max_set = 0;
				int max_all = 0;
				for (i = 0; i < frcount; i++) {
					if (max_set < nrow[t][col_temp[i]]) {
						max_set = nrow[t][col_temp[i]];
						max_all = col_temp[i];
					}
				}
				add_aoc(t, max_all);
			}
		}
	}
}

void FourlevelGreedy(int t)
{
	int add_count = 0;
	while (add_count < max_k)
	{
		if (uncover_num > 0)
			Four_stage_solution_structure(t);
		else
		{
			// A maximum-k solution still needs exactly k vertices even when a
			// full cover is reached early.
			int vertex = 0;
			while (vertex < n && cs[t][vertex].is_in_c == 1)
				vertex++;
			if (vertex == n)
				break;
			add_aoc(t, vertex);
		}
		add_count++;
	}
	int current_value = m - uncover_num;
	if (!initial_solution_saved || current_value > best_value)
	{
		int saved = 0;
		for (int vertex = 0; vertex < n && saved < max_k; vertex++)
		{
			if (cs[t][vertex].is_in_c == 1)
				Remember[saved++] = vertex;
		}
		if (!initial_solution_saved)
		{
			best_value = current_value;
			real_time = time_test.GetTime();
			initial_solution_saved = true;
			consider_global_incumbent(current_value, real_time);
		}
		else
			update_best_sol();
	}
	init_search(t);
}

void FourlevelGreedyw(int t)
{
	int add_count = 0;
	while (add_count < max_k)
	{
		if (uncover_num > 0)
			Four_stage_solution_structure(t);
		else
		{
			int vertex = 0;
			while (vertex < n && cs[t][vertex].is_in_c == 1)
				vertex++;
			if (vertex == n)
				break;
			add_aoc(t, vertex);
		}
		add_count++;
	}
	int current_value = m - uncover_num;
	if (!initial_solution_saved || current_value > best_value)
	{
		int saved = 0;
		for (int vertex = 0; vertex < n && saved < max_k; vertex++)
		{
			if (cs[t][vertex].is_in_c == 1)
				Remember[saved++] = vertex;
		}
		if (!initial_solution_saved)
		{
			best_value = current_value;
			real_time = time_test.GetTime();
			initial_solution_saved = true;
			consider_global_incumbent(current_value, real_time);
		}
		else
			update_best_sol();
	}
	init_searchw(t);
}


void find_worst_in_c(int t)
{
	int maxc = 0, maxx = INT_MIN;
	int k_temp = max_k;
	int tempk = 0;
	for (int j = 0; j < k_half; j++)
	{
		randnum = rand() % k_temp;
		rem_half[j] = rem_temp[randnum];
		k_temp--;
		tempk = rem_temp[randnum];
		rem_temp[randnum] = rem_temp[k_temp];
		rem_temp[k_temp] = tempk;
	}
	randstep = rand() % k_half;
	for (int j = randstep; j < k_half; j++)
	{
		if (rem_half[j] == tabu[t])
			continue;
		if (maxx < cs[t][rem_half[j]].score)
		{
			maxc = rem_half[j];
			maxx = cs[t][rem_half[j]].score;
		}
		else if (maxx == cs[t][rem_half[j]].score)
		{
			if (subscore[t][maxc] > subscore[t][rem_half[j]])
				maxc = rem_half[j];
			else if (subscore[t][maxc] == subscore[t][rem_half[j]])
			{
				if (set_age[t][maxc] < set_age[t][rem_half[j]])
					maxc = rem_half[j];
			}
		}
	}
	for (int j = 0; j < randstep; j++)
	{

		if (rem_half[j] == tabu[t])
			continue;
		if (maxx < cs[t][rem_half[j]].score)
		{
			maxc = rem_half[j];
			maxx = cs[t][rem_half[j]].score;
		}
		else if (maxx == cs[t][rem_half[j]].score)
		{
			if (subscore[t][maxc] > subscore[t][rem_half[j]])
				maxc = rem_half[j];
			else if (subscore[t][maxc] == subscore[t][rem_half[j]])
			{
				if (set_age[t][maxc] < set_age[t][rem_half[j]])
					maxc = rem_half[j];
			}
		}
	}
	// With k=1 the sole selected vertex is also tabu. It is still a valid
	// removal; never fall through to the default vertex 0 unless it is that
	// selected vertex.
	if (maxx == INT_MIN)
		maxc = rem_half[0];
	sad = maxc;
}

#ifndef HAIM_LEGACY_SWAP
// Compute the size-dependent limits once per k, not once per search move.
// Explicit -D candidate limits and the fixed-parameter ablation retain the
// original constant limits. Always clamp to the available side of the set.
int joint_swap_remove_limit()
{
#ifdef HAIM_ADAPTIVE_JOINT_REMOVE
	static int cached_k = -1;
	static int cached_limit = 0;
	if (cached_k != max_k)
	{
		cached_k = max_k;
		cached_limit = min(max_k,
			min(64, max(16, (int)ceil(2.0 * sqrt((double)max_k)))));
	}
	return cached_limit;
#else
	return min(max_k, JOINT_SWAP_CANDIDATE_LIMIT);
#endif
}

int joint_swap_add_limit()
{
	return min(n - max_k, JOINT_SWAP_CANDIDATE_LIMIT);
}

// Rank promising removal vertices by low weighted removal loss. For vertices
// in C, score is the negative loss, so a larger score is better.
bool better_joint_remove(int t, int lhs, int rhs)
{
	if (cs[t][lhs].score != cs[t][rhs].score)
		return cs[t][lhs].score > cs[t][rhs].score;
	if (subscore[t][lhs] != subscore[t][rhs])
		return subscore[t][lhs] < subscore[t][rhs];
	if (set_age[t][lhs] != set_age[t][rhs])
		return set_age[t][lhs] > set_age[t][rhs];
	return lhs < rhs;
}

// Rank promising insertion vertices by weighted uncovered-edge gain.
bool better_joint_add(int t, int lhs, int rhs)
{
	if (cs[t][lhs].score != cs[t][rhs].score)
		return cs[t][lhs].score > cs[t][rhs].score;
	if (subscore[t][lhs] != subscore[t][rhs])
		return subscore[t][lhs] > subscore[t][rhs];
	if (set_age[t][lhs] != set_age[t][rhs])
		return set_age[t][lhs] > set_age[t][rhs];
	return lhs < rhs;
}

// Keep a small sorted list without sorting every vertex on every move.
void insert_joint_candidate(int t, int vertex, int *candidates,
	int &candidate_count, int candidate_limit, bool remove_side)
{
	int position;
	if (candidate_count < candidate_limit)
	{
		position = candidate_count;
		candidate_count++;
	}
	else
	{
		bool is_better = remove_side
			? better_joint_remove(t, vertex, candidates[candidate_count - 1])
			: better_joint_add(t, vertex, candidates[candidate_count - 1]);
		if (!is_better)
			return;
		position = candidate_count - 1;
	}

	while (position > 0)
	{
		bool is_better = remove_side
			? better_joint_remove(t, vertex, candidates[position - 1])
			: better_joint_add(t, vertex, candidates[position - 1]);
		if (!is_better)
			break;
		candidates[position] = candidates[position - 1];
		position--;
	}
	candidates[position] = vertex;
}

// Keep the first exact-delta tie. In the hybrid build this deliberately does
// not consume rand(), so a non-improving fallback preserves the legacy random
// trajectory instead of perturbing it during candidate evaluation.
void consider_joint_pair(int remove_vertex, int add_vertex, long long delta,
	long long &best_delta, int &best_remove, int &best_add)
{
	if (delta > best_delta)
	{
		best_delta = delta;
		best_remove = remove_vertex;
		best_add = add_vertex;
	}
}

// Jointly choose a 1-for-1 exchange. Before removal, an outside vertex's
// score is its insertion gain and an inside vertex's score is minus its
// removal loss. If the pair shares an edge, removing one endpoint and adding
// the other keeps that edge covered, hence the shared-edge bonus:
//
//   delta(remove c, add v) = score[v] + score[c] + weight(c,v)
//
// The best size-limited removal candidates are considered. For each we compare
// the best globally configuration-allowed insertions and every outside
// neighbor of the removal vertex. Neighbor candidates become configuration-
// allowed when that vertex is removed, matching remove()'s behavior.
bool find_best_joint_swap(int t, long long &selected_delta)
{
	if (max_k < 1 || max_k >= n)
		return false;
	const int remove_limit = joint_swap_remove_limit();
	const int add_limit = joint_swap_add_limit();
	int remove_candidates[HAIM_JOINT_REMOVE_CAPACITY];
	int add_candidates[JOINT_SWAP_CANDIDATE_LIMIT];
	int remove_count = 0;
	int add_count = 0;

	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1 && vertex != tabu[t])
			insert_joint_candidate(t, vertex, remove_candidates,
				remove_count, remove_limit, true);
		else if (cs[t][vertex].is_in_c == 0 && cs[t][vertex].config == 1)
			insert_joint_candidate(t, vertex, add_candidates,
				add_count, add_limit, false);
	}

	// A size-one solution can leave only the tabu vertex removable. Falling
	// back to it is safer than returning an invalid default vertex.
	if (remove_count == 0)
	{
		for (int vertex = 0; vertex < n; vertex++)
		{
			if (cs[t][vertex].is_in_c == 1)
				insert_joint_candidate(t, vertex, remove_candidates,
					remove_count, remove_limit, true);
		}
	}

	if (remove_count == 0)
		return false;

	static vector<int> shared_bonus;
	static vector<unsigned char> is_top_add;
	static vector<int> touched_vertices;
	if ((int)shared_bonus.size() != n)
	{
		shared_bonus.assign(n, 0);
		is_top_add.assign(n, 0);
		touched_vertices.reserve(n);
	}
	else
	{
		fill(is_top_add.begin(), is_top_add.end(), 0);
	}

	for (int i = 0; i < add_count; i++)
		is_top_add[add_candidates[i]] = 1;

	long long best_delta = -(1LL << 60);
	int best_remove = -1;
	int best_add = -1;

	for (int r = 0; r < remove_count; r++)
	{
		int remove_vertex = remove_candidates[r];
		touched_vertices.clear();

		// Accumulate the exact shared-edge bonus. The current parser gives
		// every element exactly two endpoints; accumulation also handles
		// parallel edges correctly.
		for (int h = 0; h < nrow[t][remove_vertex]; h++)
		{
			int element = col[t][remove_vertex][h];
			for (int l = 0; l < ncol[t][element]; l++)
			{
				int add_vertex = row[t][element][l];
				if (add_vertex == remove_vertex || cs[t][add_vertex].is_in_c == 1)
					continue;
				if (shared_bonus[add_vertex] == 0)
					touched_vertices.push_back(add_vertex);
				shared_bonus[add_vertex] += rs[t][element].weight;
			}
		}

		for (int a = 0; a < add_count; a++)
		{
			int add_vertex = add_candidates[a];
			long long delta = (long long)cs[t][remove_vertex].score
				+ cs[t][add_vertex].score + shared_bonus[add_vertex];
			consider_joint_pair(remove_vertex, add_vertex, delta,
				best_delta, best_remove, best_add);
		}

		// Configuration-disabled neighbors become enabled by remove().
		for (size_t a = 0; a < touched_vertices.size(); a++)
		{
			int add_vertex = touched_vertices[a];
			if (is_top_add[add_vertex])
				continue;
			long long delta = (long long)cs[t][remove_vertex].score
				+ cs[t][add_vertex].score + shared_bonus[add_vertex];
			consider_joint_pair(remove_vertex, add_vertex, delta,
				best_delta, best_remove, best_add);
		}

		for (size_t a = 0; a < touched_vertices.size(); a++)
			shared_bonus[touched_vertices[a]] = 0;
	}

	if (best_remove < 0 || best_add < 0)
		return false;

	sad = best_remove;
	happy = best_add;
	selected_delta = best_delta;
	return true;
}
#endif

void remove(int t, int c)
{
	cs[t][c].is_in_c = 0;
	cs[t][c].score = -cs[t][c].score;
	cs[t][c].config = 0;
	set_age[t][c] = 0.9*set_age[t][c];
	int i, j, h, l;
	for (h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] --;
		if (covered[t][i] == 0)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				cs[t][j].score += rs[t][i].weight;
				subscore[t][j] = subscore[t][j] - rs[t][i].weight;
			}
			index_uncover[t][i] = 1;
			uncover_num++;
		}
		else if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 0) {
					subscore[t][j] += rs[t][i].weight;
				}
				else {
					subscore[t][j] -= rs[t][i].weight;
					cs[t][j].score -= rs[t][i].weight;
				}
			}
		}
		else if (covered[t][i] == 2)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 1) {
					subscore[t][j] += rs[t][i].weight;
				}
			}
		}
		else if (covered[t][i] > 2) {
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
			}
		}
	}
}
void find_ran_vout(int t)
{
	int maxc = 0, varr = 0;
	int maxx = INT_MIN;
	int count1 = 0, ram = 0;
	ram = rand() % uncover_num;
	for (int j = 0; j < m; j++)
	{
		if (index_uncover[t][j] == -1)
			continue;
		if (count1 == ram)
		{
			varr = j;
			break;
		}
		count1++;
	}
	//8.19 randstep
	int therand = ncol[t][varr];
	randstep = rand() % therand;
	//
	int jj = 0;
	for (int j = randstep; j < ncol[t][varr]; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1)
			continue;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
		else if (maxx == cs[t][jj].score)
		{
			if (subscore[t][maxc] < subscore[t][jj])
				maxc = jj;
			else if (subscore[t][maxc] == subscore[t][jj]) {
				if (set_age[t][maxc] < set_age[t][jj])
					maxc = jj;
			}
		}
	}
	for (int j = 0; j < randstep; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1)
			continue;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
		else if (maxx == cs[t][jj].score)
		{
			if (subscore[t][maxc] < subscore[t][jj])
				maxc = jj;
			else if (subscore[t][maxc] == subscore[t][jj]) {
				if (set_age[t][maxc] < set_age[t][jj])
					maxc = jj;
			}
		}
	}
	happy = maxc;
}




void find_random_in_vout(int t)
{
	int maxc = 0, varr = 0;
	int maxx = INT_MIN;
	int count1 = 0, ram = 0;
	ram = rand() % uncover_num;
	for (int j = 0; j < m; j++)
	{
		if (index_uncover[t][j] == -1)
			continue;
		if (count1 == ram)
		{
			varr = j;
			break;
		}
		count1++;
	}
	int therand = ncol[t][varr];
	randstep = rand() % therand;
	//
	int jj = 0;
	int jpj = 0;
	for (int j = randstep; j < ncol[t][varr]; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1 || cs[t][jj].config == 0)
			continue;
		jpj = 1;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
		else if (maxx == cs[t][jj].score)
		{
			if (subscore[t][maxc] < subscore[t][jj])
				maxc = jj;
			else if (subscore[t][maxc] == subscore[t][jj]) {
				if (set_age[t][maxc] < set_age[t][jj])
					maxc = jj;
			}
		}
	}
	for (int j = 0; j < randstep; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1 || cs[t][jj].config == 0)
			continue;
		jpj = 1;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
		else if (maxx == cs[t][jj].score)
		{
			//subscore
			if (subscore[t][maxc] < subscore[t][jj])
				maxc = jj;
			else if (subscore[t][maxc] == subscore[t][jj]) {
				if (set_age[t][maxc] < set_age[t][jj])
					maxc = jj;
			}
		}
	}
	happy = maxc;
	if (jpj == 0)
		find_ran_vout(t);
}

void add(int t, int c)
{
	cs[t][c].is_in_c = 1;
	cs[t][c].score = -cs[t][c].score;
	set_age[t][c] = 0.9*set_age[t][c];
	tabu[t] = c;
	int i, j, h, l;
	for (h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] ++;

		if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;//config = 1;
				cs[t][j].score -= rs[t][i].weight;
				subscore[t][j] += rs[t][i].weight;
			}
			index_uncover[t][i] = -1;
			uncover_num--;
		}
		else if (covered[t][i] == 2)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 0) {
					subscore[t][j] -= rs[t][i].weight;
				}
				else {
					cs[t][j].score += rs[t][i].weight;
					subscore[t][j] += rs[t][i].weight;
				}
			}
		}
		else if (covered[t][i] == 3)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 1) {
					subscore[t][j] -= rs[t][i].weight;
				}
			}
		}
		else if (covered[t][i] > 3)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
			}
		}
	}
}

#ifdef HAIM_CHECK_INVARIANTS
void check_search_state(int t, int expected_selected, bool check_rem_list)
{
	int selected_count = 0;
	for (int vertex = 0; vertex < n; vertex++)
		selected_count += cs[t][vertex].is_in_c ? 1 : 0;
	assert(selected_count == expected_selected);

	int actual_uncovered = 0;
	for (int element = 0; element < m; element++)
	{
		int actual_covered = 0;
		for (int h = 0; h < ncol[t][element]; h++)
			actual_covered += cs[t][row[t][element][h]].is_in_c ? 1 : 0;
		assert(covered[t][element] == actual_covered);
		assert((index_uncover[t][element] == -1) == (actual_covered > 0));
		actual_uncovered += actual_covered == 0 ? 1 : 0;
	}
	assert(uncover_num == actual_uncovered);

	if (check_rem_list)
	{
		vector<unsigned char> seen(n, 0);
		for (int i = 0; i < max_k; i++)
		{
			assert(rem_temp[i] >= 0 && rem_temp[i] < n);
			assert(cs[t][rem_temp[i]].is_in_c == 1);
			assert(seen[rem_temp[i]] == 0);
			seen[rem_temp[i]] = 1;
		}
	}
}
#endif

#ifdef HAIM_TWO_SWAP_ENABLED
int two_swap_remove_limit()
{
#ifdef HAIM_ADAPTIVE_TWO_SWAP_REMOVE
	static int cached_k = -1;
	static int cached_limit = 0;
	if (cached_k != max_k)
	{
		cached_k = max_k;
		cached_limit = min(max_k,
			min(16, max(8, (int)ceil(sqrt((double)max_k)))));
	}
	return cached_limit;
#else
	return min(max_k, HAIM_TWO_SWAP_REMOVE_LIMIT);
#endif
}

int two_swap_add_limit()
{
#ifdef HAIM_ADAPTIVE_TWO_SWAP_ADD
	const int outside_count = n - max_k;
	static int cached_outside = -1;
	static int cached_limit = 0;
	if (cached_outside != outside_count)
	{
		cached_outside = outside_count;
		cached_limit = min(outside_count,
			min(32, max(16, (int)ceil(2.0 * sqrt((double)outside_count)))));
	}
	return cached_limit;
#else
	return min(n - max_k, HAIM_TWO_SWAP_ADD_LIMIT);
#endif
}

// Rank removals by raw loss first. Weighted loss and the existing CCA
// tie-breakers retain guidance from dynamically weighted difficult edges.
bool better_two_swap_remove(int t, int lhs, int rhs,
	const vector<int> &raw_base)
{
	if (raw_base[lhs] != raw_base[rhs])
		return raw_base[lhs] > raw_base[rhs];
	if (cs[t][lhs].score != cs[t][rhs].score)
		return cs[t][lhs].score > cs[t][rhs].score;
	if (subscore[t][lhs] != subscore[t][rhs])
		return subscore[t][lhs] < subscore[t][rhs];
	if (set_age[t][lhs] != set_age[t][rhs])
		return set_age[t][lhs] > set_age[t][rhs];
	return lhs < rhs;
}

bool better_two_swap_add(int t, int lhs, int rhs,
	const vector<long long> &raw_key,
	const vector<long long> &weighted_key)
{
	if (raw_key[lhs] != raw_key[rhs])
		return raw_key[lhs] > raw_key[rhs];
	if (weighted_key[lhs] != weighted_key[rhs])
		return weighted_key[lhs] > weighted_key[rhs];
	if (subscore[t][lhs] != subscore[t][rhs])
		return subscore[t][lhs] > subscore[t][rhs];
	if (set_age[t][lhs] != set_age[t][rhs])
		return set_age[t][lhs] > set_age[t][rhs];
	return lhs < rhs;
}

void insert_two_swap_remove(int t, int vertex, int *candidates,
	int &candidate_count, int candidate_limit, const vector<int> &raw_base)
{
	int position;
	if (candidate_count < candidate_limit)
		position = candidate_count++;
	else
	{
		if (!better_two_swap_remove(t, vertex,
			candidates[candidate_count - 1], raw_base))
			return;
		position = candidate_count - 1;
	}
	while (position > 0 && better_two_swap_remove(t, vertex,
		candidates[position - 1], raw_base))
	{
		candidates[position] = candidates[position - 1];
		position--;
	}
	candidates[position] = vertex;
}

void insert_two_swap_add(int t, int vertex, int *candidates,
	int &candidate_count, int candidate_limit, const vector<long long> &raw_key,
	const vector<long long> &weighted_key)
{
	int position;
	if (candidate_count < candidate_limit)
		position = candidate_count++;
	else
	{
		if (!better_two_swap_add(t, vertex,
			candidates[candidate_count - 1], raw_key, weighted_key))
			return;
		position = candidate_count - 1;
	}
	while (position > 0 && better_two_swap_add(t, vertex,
		candidates[position - 1], raw_key, weighted_key))
	{
		candidates[position] = candidates[position - 1];
		position--;
	}
	candidates[position] = vertex;
}

// Find a strictly improving simultaneous exchange
// S' = S - {r1,r2} + {a1,a2}. For graph edge weight W(u,v), the exact
// weighted delta is
//
// sum(score[r]) + sum(score[a]) - W(r1,r2) - W(a1,a2)
//     + sum W(ri,aj).
//
// The same formula with unit edge weights gives the raw covered-edge delta.
// Candidate restriction makes enumeration inexpensive, while every evaluated
// quartet still has an exact delta (including parallel edges). Configuration
// checking is deliberately ignored here: this strictly improving move is an
// aspiration step called periodically, or on stagnation in comparison builds.
bool try_restricted_two_swap(int t, bool weighted_search)
{
#ifdef HAIM_TWO_SWAP_STATS
	two_swap_calls++;
#endif
	if (!two_swap_graph_compatible || uncover_num == 0
		|| max_k < 2 || n - max_k < 2
		|| time_test.NeedToStop())
		return false;

	static vector<int> raw_base;
	raw_base.assign(n, 0);
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1)
		{
			int loss = 0;
			for (int h = 0; h < nrow[t][vertex]; h++)
				loss += covered[t][col[t][vertex][h]] == 1 ? 1 : 0;
			raw_base[vertex] = -loss;
		}
		else
		{
			int gain = 0;
			for (int h = 0; h < nrow[t][vertex]; h++)
				gain += covered[t][col[t][vertex][h]] == 0 ? 1 : 0;
			raw_base[vertex] = gain;
		}
		if ((vertex & 63) == 63 && time_test.NeedToStop())
			return false;
	}
	if (time_test.NeedToStop())
		return false;

	const int remove_limit = two_swap_remove_limit();
	const int add_limit = two_swap_add_limit();
	int remove_candidates[HAIM_TWO_SWAP_REMOVE_CAPACITY];
	int remove_count = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1 && vertex != tabu[t])
			insert_two_swap_remove(t, vertex, remove_candidates,
				remove_count, remove_limit, raw_base);
	}
	if (remove_count < 2)
	{
		remove_count = 0;
		for (int vertex = 0; vertex < n; vertex++)
		{
			if (cs[t][vertex].is_in_c == 1)
				insert_two_swap_remove(t, vertex, remove_candidates,
					remove_count, remove_limit, raw_base);
		}
	}
	if (remove_count < 2)
		return false;

	// Estimate each insertion's best possible interaction with two removal
	// candidates. This admits vertices whose current one-point gain is small
	// but which become valuable only when two selected neighbors are removed.
	static vector<int> cross_count;
	static vector<long long> cross_weight;
	cross_count.assign(remove_count * n, 0);
	cross_weight.assign(remove_count * n, 0);
	for (int ri = 0; ri < remove_count; ri++)
	{
		int remove_vertex = remove_candidates[ri];
		for (int h = 0; h < nrow[t][remove_vertex]; h++)
		{
			int element = col[t][remove_vertex][h];
			int u = row[t][element][0];
			int v = row[t][element][1];
			int other = u == remove_vertex ? v : u;
			if (cs[t][other].is_in_c == 0)
			{
				int offset = ri * n + other;
				cross_count[offset]++;
				cross_weight[offset] += rs[t][element].weight;
			}
		}
		if (time_test.NeedToStop())
			return false;
	}

	static vector<long long> add_raw_key;
	static vector<long long> add_weighted_key;
	add_raw_key.assign(n, 0);
	add_weighted_key.assign(n, 0);
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1)
			continue;
		int first_count = 0, second_count = 0;
		long long first_weight = 0, second_weight = 0;
		for (int ri = 0; ri < remove_count; ri++)
		{
			int count_value = cross_count[ri * n + vertex];
			if (count_value > first_count)
			{
				second_count = first_count;
				first_count = count_value;
			}
			else if (count_value > second_count)
				second_count = count_value;

			long long weight_value = cross_weight[ri * n + vertex];
			if (weight_value > first_weight)
			{
				second_weight = first_weight;
				first_weight = weight_value;
			}
			else if (weight_value > second_weight)
				second_weight = weight_value;
		}
		add_raw_key[vertex] = (long long)raw_base[vertex]
			+ first_count + second_count;
		add_weighted_key[vertex] = (long long)cs[t][vertex].score
			+ first_weight + second_weight;
		if ((vertex & 127) == 127 && time_test.NeedToStop())
			return false;
	}
	if (time_test.NeedToStop())
		return false;

	int add_candidates[HAIM_TWO_SWAP_ADD_CAPACITY];
	int add_count = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 0)
			insert_two_swap_add(t, vertex, add_candidates, add_count,
				add_limit, add_raw_key, add_weighted_key);
	}
	if (add_count < 2)
		return false;

	static vector<int> remove_position;
	static vector<int> add_position;
	remove_position.assign(n, -1);
	add_position.assign(n, -1);
	for (int ri = 0; ri < remove_count; ri++)
		remove_position[remove_candidates[ri]] = ri;
	for (int ai = 0; ai < add_count; ai++)
		add_position[add_candidates[ai]] = ai;

	static vector<int> rr_count, aa_count, ra_count;
	static vector<long long> rr_weight, aa_weight, ra_weight;
	rr_count.assign(remove_count * remove_count, 0);
	aa_count.assign(add_count * add_count, 0);
	ra_count.assign(remove_count * add_count, 0);
	rr_weight.assign(remove_count * remove_count, 0);
	aa_weight.assign(add_count * add_count, 0);
	ra_weight.assign(remove_count * add_count, 0);

	// The remove/add interactions are already available from insertion ranking.
	// For same-side interactions scan only candidate adjacency lists, counting
	// an edge from its lower candidate index once and mirroring the result.
	// Distinct parallel edges remain distinct entries in the adjacency lists.
	for (int ri = 0; ri < remove_count; ri++)
	{
		int vertex = remove_candidates[ri];
		for (int ai = 0; ai < add_count; ai++)
		{
			int offset = ri * n + add_candidates[ai];
			ra_count[ri * add_count + ai] = cross_count[offset];
			ra_weight[ri * add_count + ai] = cross_weight[offset];
		}
		for (int h = 0; h < nrow[t][vertex]; h++)
		{
			if ((h & 65535) == 65535 && time_test.NeedToStop())
				return false;
			int element = col[t][vertex][h];
			int other = row[t][element][0] == vertex
				? row[t][element][1] : row[t][element][0];
			int rj = remove_position[other];
			if (rj <= ri)
				continue;
			long long weight = rs[t][element].weight;
			rr_count[ri * remove_count + rj]++;
			rr_count[rj * remove_count + ri]++;
			rr_weight[ri * remove_count + rj] += weight;
			rr_weight[rj * remove_count + ri] += weight;
		}
		if (time_test.NeedToStop())
			return false;
	}
	for (int ai = 0; ai < add_count; ai++)
	{
		int vertex = add_candidates[ai];
		for (int h = 0; h < nrow[t][vertex]; h++)
		{
			if ((h & 65535) == 65535 && time_test.NeedToStop())
				return false;
			int element = col[t][vertex][h];
			int other = row[t][element][0] == vertex
				? row[t][element][1] : row[t][element][0];
			int aj = add_position[other];
			if (aj <= ai)
				continue;
			long long weight = rs[t][element].weight;
			aa_count[ai * add_count + aj]++;
			aa_count[aj * add_count + ai]++;
			aa_weight[ai * add_count + aj] += weight;
			aa_weight[aj * add_count + ai] += weight;
		}
		if (time_test.NeedToStop())
			return false;
	}
	if (time_test.NeedToStop())
		return false;

	long long best_raw_delta = 0;
	long long best_weighted_delta = LLONG_MIN;
	int best_r1 = -1, best_r2 = -1, best_a1 = -1, best_a2 = -1;
	long long evaluated = 0;
	for (int ri = 0; ri < remove_count; ri++)
	{
		for (int rj = ri + 1; rj < remove_count; rj++)
		{
			int r1 = remove_candidates[ri];
			int r2 = remove_candidates[rj];
			for (int ai = 0; ai < add_count; ai++)
			{
				for (int aj = ai + 1; aj < add_count; aj++)
				{
					int a1 = add_candidates[ai];
					int a2 = add_candidates[aj];
					long long raw_delta = (long long)raw_base[r1]
						+ raw_base[r2] + raw_base[a1] + raw_base[a2]
						- rr_count[ri * remove_count + rj]
						- aa_count[ai * add_count + aj]
						+ ra_count[ri * add_count + ai]
						+ ra_count[ri * add_count + aj]
						+ ra_count[rj * add_count + ai]
						+ ra_count[rj * add_count + aj];
					long long weighted_delta =
						(long long)cs[t][r1].score + cs[t][r2].score
						+ cs[t][a1].score + cs[t][a2].score
						- rr_weight[ri * remove_count + rj]
						- aa_weight[ai * add_count + aj]
						+ ra_weight[ri * add_count + ai]
						+ ra_weight[ri * add_count + aj]
						+ ra_weight[rj * add_count + ai]
						+ ra_weight[rj * add_count + aj];
					evaluated++;
					if (raw_delta > best_raw_delta
						|| (raw_delta == best_raw_delta && raw_delta > 0
							&& weighted_delta > best_weighted_delta))
					{
						best_raw_delta = raw_delta;
						best_weighted_delta = weighted_delta;
						best_r1 = r1;
						best_r2 = r2;
						best_a1 = a1;
						best_a2 = a2;
					}
				}
			}
			if (time_test.NeedToStop())
			{
#ifdef HAIM_TWO_SWAP_STATS
				two_swap_combinations += evaluated;
#endif
				return false;
			}
		}
	}
#ifdef HAIM_TWO_SWAP_STATS
	two_swap_combinations += evaluated;
#endif
	if (best_raw_delta <= 0 || best_r1 < 0)
		return false;

	int raw_value_before = m - uncover_num;
	assert(cs[t][best_r1].is_in_c == 1);
	assert(cs[t][best_r2].is_in_c == 1);
	assert(cs[t][best_a1].is_in_c == 0);
	assert(cs[t][best_a2].is_in_c == 0);
	remove(t, best_r1);
	remove(t, best_r2);
	add(t, best_a1);
	add(t, best_a2);
	sad = best_r2;
	happy = best_a2;

	for (int vertex = 0; vertex < n; vertex++)
	{
		if (vertex != best_r1 && vertex != best_r2
			&& vertex != best_a1 && vertex != best_a2)
			set_age[t][vertex]++;
	}
	tpp = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1)
			rem_temp[tpp++] = vertex;
	}
	assert(tpp == max_k);
	if (weighted_search)
		reload_weight_score(t);
	int actual_raw_gain = m - uncover_num - raw_value_before;
	assert(actual_raw_gain == best_raw_delta);
	(void)actual_raw_gain;

#ifdef HAIM_CHECK_INVARIANTS
	check_search_state(t, max_k, true);
#endif
#ifdef HAIM_TWO_SWAP_STATS
	two_swap_moves++;
	two_swap_raw_gain += actual_raw_gain;
#endif
	return true;
}
#endif

#ifndef HAIM_DISABLE_LNS
// LNS deliberately ignores configuration checking while rebuilding: it is a
// diversification move, not another local 1-swap. All state transitions still
// go through remove()/add(), so weighted and unweighted searches share it.
bool better_lns_candidate(int t, int lhs, int rhs, bool remove_side)
{
	if (cs[t][lhs].score != cs[t][rhs].score)
		return cs[t][lhs].score > cs[t][rhs].score;
	if (subscore[t][lhs] != subscore[t][rhs])
		return remove_side
			? subscore[t][lhs] < subscore[t][rhs]
			: subscore[t][lhs] > subscore[t][rhs];
	if (set_age[t][lhs] != set_age[t][rhs])
		return set_age[t][lhs] > set_age[t][rhs];
	return lhs < rhs;
}

void insert_lns_candidate(int t, int vertex, int *candidates,
	int &candidate_count, bool remove_side)
{
	int position;
	if (candidate_count < HAIM_LNS_RCL_SIZE)
	{
		position = candidate_count++;
	}
	else
	{
		if (!better_lns_candidate(t, vertex,
			candidates[candidate_count - 1], remove_side))
			return;
		position = candidate_count - 1;
	}

	while (position > 0 && better_lns_candidate(t, vertex,
		candidates[position - 1], remove_side))
	{
		candidates[position] = candidates[position - 1];
		position--;
	}
	candidates[position] = vertex;
}

int select_lns_remove_vertex(int t)
{
	int candidates[HAIM_LNS_RCL_SIZE];
	int candidate_count = 0;
	int selected_count = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 0)
			continue;
		selected_count++;
		insert_lns_candidate(t, vertex, candidates, candidate_count, true);
	}
	if (selected_count == 0)
		return -1;

	// One quarter of destroy moves are fully random. Otherwise use a
	// loss-biased restricted candidate list (75% best, 25% random in top RCL).
	if (rand() % 4 == 0)
	{
		int target = rand() % selected_count;
		for (int vertex = 0; vertex < n; vertex++)
		{
			if (cs[t][vertex].is_in_c == 0)
				continue;
			if (target-- == 0)
				return vertex;
		}
	}
	if (rand() % 4 != 0)
		return candidates[0];
	return candidates[rand() % candidate_count];
}

int select_lns_add_vertex(int t, const vector<unsigned char> &destroyed,
	bool require_novel)
{
	int candidates[HAIM_LNS_RCL_SIZE];
	int candidate_count = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1)
			continue;
		if (require_novel && destroyed[vertex])
			continue;
		insert_lns_candidate(t, vertex, candidates, candidate_count, false);
	}
	if (candidate_count == 0)
		return -1;
	if (rand() % 4 != 0)
		return candidates[0];
	return candidates[rand() % candidate_count];
}

void lns_destroy_repair(int t)
{
	if (max_k <= 0 || max_k >= n)
		return;
#ifdef HAIM_LNS_STATS
	int raw_value_before = m - uncover_num;
#endif

	static vector<unsigned char> destroyed;
	static vector<unsigned char> moved;
	destroyed.assign(n, 0);
	moved.assign(n, 0);

	int percent_range = HAIM_LNS_MAX_DESTROY_PERCENT
		- HAIM_LNS_MIN_DESTROY_PERCENT + 1;
	int destroy_percent = HAIM_LNS_MIN_DESTROY_PERCENT
		+ rand() % percent_range;
	int destroy_count = (int)((1LL * max_k * destroy_percent + 99) / 100);
	if (max_k > 1)
		destroy_count = std::max(2, destroy_count);
	destroy_count = std::min(destroy_count, max_k);
	destroy_count = std::min(destroy_count, HAIM_LNS_MAX_DESTROY_VERTICES);

#ifdef HAIM_CHECK_INVARIANTS
	check_search_state(t, max_k, true);
#endif

	int removed_count = 0;
	for (; removed_count < destroy_count; removed_count++)
	{
		int vertex = select_lns_remove_vertex(t);
		if (vertex < 0)
			break;
		destroyed[vertex] = 1;
		moved[vertex] = 1;
		sad = vertex;
		remove(t, vertex);
	}

#ifdef HAIM_CHECK_INVARIANTS
	check_search_state(t, max_k - removed_count, false);
#endif

	// Force as many genuinely new vertices as the original outside set can
	// supply; only then may repair reinsert a destroyed vertex.
	int novel_steps = std::min(removed_count, n - max_k);
	for (int step_index = 0; step_index < removed_count; step_index++)
	{
		int vertex = select_lns_add_vertex(t, destroyed,
			step_index < novel_steps);
		if (vertex < 0)
			vertex = select_lns_add_vertex(t, destroyed, false);
		assert(vertex >= 0);
		moved[vertex] = 1;
		happy = vertex;
		add(t, vertex);
	}

	for (int vertex = 0; vertex < n; vertex++)
	{
		if (!moved[vertex])
			set_age[t][vertex] += removed_count;
	}

	tpp = 0;
	for (int vertex = 0; vertex < n; vertex++)
	{
		if (cs[t][vertex].is_in_c == 1)
			rem_temp[tpp++] = vertex;
	}
	assert(tpp == max_k);

#ifdef HAIM_CHECK_INVARIANTS
	check_search_state(t, max_k, true);
#endif

#ifdef HAIM_LNS_STATS
	lns_calls++;
	lns_destroyed_vertices += removed_count;
	lns_raw_delta_sum += (m - uncover_num) - raw_value_before;
#endif
}
#endif

void reload_weight_score(int t)
{
	
	int fpx, fpx1, fpx2;
	int r = 1000;
	double p = 0.3;
	int init_weight = 0;
	for (fpx = 0; fpx < m; fpx++)
	{
		if (index_uncover[t][fpx] == -1)
			continue;
		if (rs[t][fpx].weight > r)
		{
			init_weight = rs[t][fpx].weight * p;
			for (fpx1 = 0; fpx1 < ncol[t][fpx]; fpx1++)
			{
				fpx2 = row[t][fpx][fpx1];
				cs[t][fpx2].score = cs[t][fpx2].score - rs[t][fpx].weight + init_weight;
			}
			rs[t][fpx].weight = init_weight;
		}
		else
		{
			rs[t][fpx].weight = rs[t][fpx].weight + 1;
			for (fpx1 = 0; fpx1 < ncol[t][fpx]; fpx1++)
			{
				fpx2 = row[t][fpx][fpx1];
				cs[t][fpx2].score = cs[t][fpx2].score + 1;

			}
		}
	}
}


void reload_age(int t)
{
	for (int traversal = 0; traversal < n; traversal++)
	{
		if (traversal != happy && traversal != sad)
		{
			set_age[t][traversal] ++;
		}
	}
}


int del_remtemp()
{
	int i;
	for (i = 0; i < max_k; i++)
	{
		if (rem_temp[i] == sad)
			break;
	}
	return i;
}

void CCA_local_search(int t)
{

	int remtemp = 0;
	if (uncover_num == 0)
		return;
#ifdef HAIM_LEGACY_SWAP
	bool use_joint_swap = false;
	find_worst_in_c(t);
#else
	long long joint_delta = 0;
	bool use_joint_swap = find_best_joint_swap(t, joint_delta);
	if (!use_joint_swap)
		find_worst_in_c(t);
#ifndef HAIM_PURE_JOINT_SWAP
	// Best-improvement intensifies the search. On a local-optimum plateau,
	// retain the original randomized move to preserve diversification.
	else if (joint_delta <= 0)
	{
		use_joint_swap = false;
		find_worst_in_c(t);
	}
#endif
#endif
	remove(t, sad);
	remtemp = del_remtemp();
	if (!use_joint_swap)
		find_random_in_vout(t);
	add(t, happy);

	rem_temp[remtemp] = happy;
	reload_age(t);
}

void CCA_local_searchw(int t)
{
	int remtemp = 0;
	if (uncover_num == 0)
		return;
#ifdef HAIM_LEGACY_SWAP
	bool use_joint_swap = false;
	find_worst_in_c(t);
#else
	long long joint_delta = 0;
	bool use_joint_swap = find_best_joint_swap(t, joint_delta);
	if (!use_joint_swap)
		find_worst_in_c(t);
#ifndef HAIM_PURE_JOINT_SWAP
	else if (joint_delta <= 0)
	{
		use_joint_swap = false;
		find_worst_in_c(t);
	}
#endif
#endif
	remove(t, sad);
	remtemp = del_remtemp();
	if (!use_joint_swap)
		find_random_in_vout(t);
	add(t, happy);
	rem_temp[remtemp] = happy;
	reload_weight_score(t);
	reload_age(t);
}


void init_aco()
{
	int t, i, j;
	for (t = 0; t < ants; t++) {
		for (j = 0; j < n; j++)
			uncover[t][j] = nrow[t][j];
	}
	for (j = 0; j < n; j++)
	{
		tau[j] = 0.5;
		eta[j] = 1.0*uncover[0][j];
	}

	for (j = 0; j < n; j++)
	{
		te_beta[j] = tau[j];
		for (i = 0; i < beta_parma; i++)
			te_beta[j] = te_beta[j] * eta[j];
	}
}


void init_eta_tebeta()
{
	int i, j;
	for (i = 0; i < MAXA; i++)
	{
		for (j = 0; j < n; j++)
			uncover[i][j] = nrow[i][j];
	}

	for (j = 0; j < n; j++)
	{
		eta[j] = 1.0*nrow[0][j];
		te_beta[j] = tau[j];
		for (i = 0; i < beta_parma; i++)
			te_beta[j] = te_beta[j] * eta[j];
	}

}

void add_aoc(int t, int c)
{
;
	int i, j, h, l;
	te_beta[c] = 0.0;
	eta[c] = 0.0;
	cs[t][c].is_in_c = 1;
	cs[t][c].score = -cs[t][c].score;
	for (h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] ++;
		if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				cs[t][j].score -= rs[t][i].weight;
				subscore[t][j] += rs[t][i].weight;
				if (eta[j] != 0.) {
					uncover[t][j] --;
					eta[j] = uncover[t][j] / sqrt(m);
				}
				te_beta[j] = tau[j];
				for (int t = 0; t < beta_parma; t++)
					te_beta[j] = te_beta[j] * eta[j];
			}
			index_uncover[t][i] = -1;
			uncover_num--;
		}
		if (covered[t][i] == 2)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 0) {
					subscore[t][j] -= rs[t][i].weight;
				}
				else {
					cs[t][j].score += rs[t][i].weight;
					subscore[t][j] += rs[t][i].weight;
				}
			}
		}
		if (covered[t][i] == 3)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
				if (cs[t][j].is_in_c == 1) {
					subscore[t][j] -= rs[t][i].weight;
				}
			}
		}
		if (covered[t][i] > 3)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].config = 1;
			}
		}
	}
}

void reload_alltau()
{
	for (int i = 0; i < n; i++)
	{
		tau[i] = tau[i] * omiga;
	}
}

void reload_alltaupro(int c)
{
	if (tau[c] < taumin)
		tau[c] = taumin;
	if (tau[c] > taumax)
		tau[c] = taumax;
}

void reload_bt(int c)
{
	tau[c] = tau[c] + delta_tau / 3;
}


void reload_bs(int c)
{
	tau[c] = tau[c] + (delta_tau * 2) / 3;//更新该集合的信息素tau
}





void maxmin()
{
	taumax = best_value / (m * omiga);
	taumin = 1 / (sqrt(m) * sqrt(sqrt(n)));
}


void add_rand(int t, int randi)
{
	int i, mid, randii;
	int pre_k = max_k;
	for (i = 0; i < randi; i++)
	{
		randii = rand() % pre_k;
		add_aoc(t, Remember[randii]);
		pre_k--;
		mid = Remember[randii];
		Remember[randii] = Remember[pre_k];
		Remember[pre_k] = mid;
	}
}
void ANT_1()
{
	int t = 0;
	init_eta_tebeta();
	ant_empty(t);
	for (int i = 0; i < max_k; i++)
	{
		add_aoc(t, Remember[i]);
	}
	ant_search(t);
}

void ANT_2()
{
	int t = 1;
	init_eta_tebeta();
	ant_empty(t);
	ant_build(t, max_k);
	ant_search(t);
}

void ANT_3()
{
	int t = 2;
	init_eta_tebeta();
	ant_empty(t);
	//0%--50%
	randant = rand() % valuek;
	add_rand(t, randant);
	int itor = max_k - randant;
	ant_build(t, itor);
	ant_search(t);
}
void ANT_4()
{
	int t = 3;
	init_eta_tebeta();
	ant_empty(t);
	//50%--
	randant = rand() % valuek;
	int A = max_k / 2 + randant;
	add_rand(t, A);
	int itor = max_k - A;
	ant_build(t, itor);
	ant_search(t);
}
void ANT_5()
{
	int t = 4;
	init_eta_tebeta();
	ant_empty(t);
	for (int i = 0; i < max_k; i++)
	{
		add_aoc(t, Remember[i]);
	}
	ant_searchw(t);
}
void ANT_6()
{
	int t = 5;
	init_eta_tebeta();
	ant_empty(t);
	ant_build(t, max_k);
	ant_searchw(t);
}

void ANT_7()
{
	int t = 6;
	init_eta_tebeta();
	ant_empty(t);
	//0%-50%
	randant = rand() % valuek;
	add_rand(t, randant);
	int itor = max_k - randant;
	ant_build(t, itor);
	ant_searchw(t);
}
void ANT_8()
{
	int t = 7;
	init_eta_tebeta();
	ant_empty(t);
	//50%--
	randant = rand() % valuek;
	int A = max_k / 2 + randant;
	add_rand(t, A);
	int itor = max_k - A;
	ant_build(t, itor);
	ant_searchw(t);
}

void ANT_greedy()
{
	int t = 10;
	init_eta_tebeta();
	ant_empty_greedy(t);
	for (int i = 0; i < max_k; i++)
	{
		add_aoc_greedy(t, Remember[i]);
	}
	ant_search_greedy(t);
}


void ant_empty_greedy(int t)
{
	for (int i = 0; i < n; i++)
	{
		if (cs[t][i].is_in_c == 1)
		{
			remove_greedy(t, i);
		}
	}
	uncover_num = m;
}

void remove_greedy(int t, int c)
{
	cs[t][c].is_in_c = 0;
	cs[t][c].score = -cs[t][c].score;
	int i, j, h, l;
	for(h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] --;
		if (covered[t][i] == 0)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].score += rs[t][i].weight;
			}
			index_uncover[t][i] = 1;
			uncover_num++;
		}
		else if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				if (cs[t][j].is_in_c != 0) {
					cs[t][j].score -= rs[t][i].weight;
				}
			}
		}
	}
}

void add_aoc_greedy(int t, int c)
{
	int i, j, h, l;
	te_beta[c] = 0.0;
	eta[c] = 0.0;
	cs[t][c].is_in_c = 1;
	cs[t][c].score = -cs[t][c].score;
	for (h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] ++;
		if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].score -= rs[t][i].weight;
				if (eta[j] != 0.) {
					uncover[t][j] --;
					eta[j] = uncover[t][j] / sqrt(m);
				}
				te_beta[j] = tau[j];
				for (int t = 0; t < beta_parma; t++)
					te_beta[j] = te_beta[j] * eta[j];
			}
			index_uncover[t][i] = -1;
			uncover_num--;
		}
		if (covered[t][i] == 2)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				if (cs[t][j].is_in_c != 0) {
					cs[t][j].score += rs[t][i].weight;
				}
			}
		}
	}
}

void add_greedy(int t,int c)
{
	cs[t][c].is_in_c = 1;
	cs[t][c].score = -cs[t][c].score;
	//tabu start
	tabu[t] = c;
	//tabu end
	int i, j, h, l;
	for (h = 0; h < nrow[t][c]; h++)
	{
		i = col[t][c][h];
		covered[t][i] ++;
		if (covered[t][i] == 1)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				cs[t][j].score -= rs[t][i].weight;
			}
			index_uncover[t][i] = -1;
			uncover_num--;
		}
		else if (covered[t][i] == 2)
		{
			for (l = 0; l < ncol[t][i]; l++)
			{
				j = row[t][i][l];
				if (j == c)
					continue;
				if (cs[t][j].is_in_c != 0) {
					cs[t][j].score += rs[t][i].weight;
				}
			}
		}
	}	
}

void ant_search_greedy(int t)
{
	cntants = 0;
	tpp = 0;
	int r = 0;
	tpp = 0;
	for (r = 0; r < n; r++)
	{
		if (cs[t][r].is_in_c == 1)
		{
			rem_temp[tpp] = r;
			tpp++;
		}
	}
	while (1)
	{
		greedy_ls(t);

		if (m - uncover_num > best_value)
		{
			tpp = 0;
			cntants = 0;
			for (r = 0; r < n; r++)
			{
				if (cs[t][r].is_in_c == 1)
				{
					Remember[tpp] = r;
					tpp++;
				}
			}
			update_best_sol();
		}
		else
		{
			cntants++;
			if (cntants > greedy_limit)
			{
				break;
			}
		}
		if (time_test.NeedToStop())	return;
	}
}

void greedy_ls(int t)
{
	int remtemp = 0;
	greedy_worst(t);
	remove_greedy(t, sad);
	remtemp = del_remtemp();
	greedy_in_vout(t);
	add(t, happy);
	rem_temp[remtemp] = happy;
}


void greedy_worst(int t)
{
	int maxc = 0, maxx = INT_MIN;
	int k_temp = max_k;
	int tempk = 0;
	for (int j = 0; j < max_k; j++)
	{
		randnum = rand() % k_temp;
		rem_half[j] = rem_temp[randnum];
		k_temp--;
		tempk = rem_temp[randnum];
		rem_temp[randnum] = rem_temp[k_temp];
		rem_temp[k_temp] = tempk;
	}
	randstep = rand() % k_half;
	for (int j = randstep; j < k_half; j++)
	{
		if (rem_half[j] == tabu[t])
			continue;
		if (maxx < cs[t][rem_half[j]].score)
		{
			maxc = rem_half[j];
			maxx = cs[t][rem_half[j]].score;
		}
	}
	for (int j = 0; j < randstep; j++)
	{

		if (rem_half[j] == tabu[t])
			continue;

		if (maxx < cs[t][rem_half[j]].score)
		{
			maxc = rem_half[j];
			maxx = cs[t][rem_half[j]].score;
		}
	}
	if (maxx == INT_MIN)
		maxc = rem_half[0];
	sad = maxc;
}

void greedy_in_vout(int t)
{
	int maxc = 0, varr = 0;
	int maxx = INT_MIN;
	int count1 = 0, ram = 0;
	ram = rand() % uncover_num;
	for (int j = 0; j < m; j++)
	{
		if (index_uncover[t][j] == -1)
			continue;
		if (count1 == ram)
		{
			varr = j;
			break;
		}
		count1++;
	}
	int therand = ncol[t][varr];
	randstep = rand() % therand;
	int jj = 0;
	for (int j = randstep; j < ncol[t][varr]; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1)
			continue;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
	}
	for (int j = 0; j < randstep; j++)
	{
		jj = row[t][varr][j];
		if (cs[t][jj].is_in_c == 1)
			continue;
		if (maxx < cs[t][jj].score)
		{
			maxc = jj;
			maxx = cs[t][jj].score;
		}
	}
	happy = maxc;
}

void ant_empty(int t)
{
	for (int i = 0; i < n; i++)
	{
		if (cs[t][i].is_in_c == 1)
		{
			remove(t, i);
		}
	}
	uncover_num = m;
}


void ant_build(int t, int itor)
{
	int antk;
	for (antk = 0; antk < itor; antk++)
	{
		int j = 0;
		double pper = 0.;
		int Irand = rand() % INT_MAX;
		double Frand = Irand / (INT_MAX * 1.0);
		double temp_sum = 0.;
		for (j = 0; j < n; j++)
		{
			if (cs[t][j].is_in_c == 1)
				continue;
			temp_sum = temp_sum + te_beta[j];
		}
		for (j = 0; j < n; j++)
		{
			if (cs[t][j].is_in_c == 0)
			{
				pper += te_beta[j] / temp_sum;
			}
			if (pper > Frand)
				break;
		}
		add_aoc(t, j);
	}
}

#if HAIM_PERIODIC_SEARCH_ENABLED
// Each ant starts at zero. Every executed base-search call counts, including
// an identity exchange. Best-value improvements do not reset these counters.
struct PeriodicSearchSchedule
{
	int lns_base_steps, two_swap_base_steps;
	bool after_lns;

	PeriodicSearchSchedule()
		: lns_base_steps(0), two_swap_base_steps(0), after_lns(false) {}

	void on_base_search()
	{
#ifndef HAIM_DISABLE_LNS
		if (max_k > 0 && max_k < n && lns_base_steps < HAIM_LNS_BASE_INTERVAL)
			++lns_base_steps;
#endif
#ifdef HAIM_TWO_SWAP_ENABLED
		if (two_swap_graph_compatible && max_k >= 2 && n - max_k >= 2
			&& two_swap_base_steps < HAIM_TWO_SWAP_BASE_INTERVAL)
			++two_swap_base_steps;
#endif
	}

	bool lns_due() const
	{
#ifndef HAIM_DISABLE_LNS
		return max_k > 0 && max_k < n && lns_base_steps >= HAIM_LNS_BASE_INTERVAL;
#else
		return false;
#endif
	}

	bool two_swap_due() const
	{
#ifdef HAIM_TWO_SWAP_ENABLED
		return two_swap_graph_compatible && max_k >= 2 && n - max_k >= 2
			&& two_swap_base_steps >= HAIM_TWO_SWAP_BASE_INTERVAL;
#else
		return false;
#endif
	}

	void after_two_swap() { two_swap_base_steps = 0; }
	void after_perturbation() { lns_base_steps = 0; after_lns = true; }
	void on_new_best() { after_lns = false; }
};
#else
// The old budget belongs to one uninterrupted best-value stagnation period.
int lns_search_budget()
{
#ifndef HAIM_DISABLE_LNS
#if defined(HAIM_FIXED_SEARCH_PARAMETERS) || defined(HAIM_EXPLICIT_LNS_BUDGET)
	return HAIM_LNS_PERTURBATIONS_PER_SEARCH;
#else
	return std::min(HAIM_LNS_MAX_PERTURBATIONS,
		std::max(3, (int)ceil(sqrt(max_k / 10.0))));
#endif
#else
	return 0;
#endif
}

struct LocalSearchSchedule
{
	int lns_uses, lns_budget, lns_interval;
	long long next_lns_at;
	int two_swap_uses, two_swap_allowance, two_swap_failures;
	long long next_two_swap_at;
	bool after_lns;

	LocalSearchSchedule()
	{
		lns_budget = lns_search_budget();
		lns_interval = 1;
#ifndef HAIM_DISABLE_LNS
		lns_interval = std::max(HAIM_LNS_MIN_TRIGGER,
			cca_limit / (lns_budget + 1));
#ifdef HAIM_FIXED_SEARCH_PARAMETERS
		lns_interval = std::min(HAIM_LNS_MAX_TRIGGER, lns_interval);
#endif
#endif
		reset();
	}

	void reset()
	{
		lns_uses = 0;
		next_lns_at = lns_interval;
#ifndef HAIM_DISABLE_LNS
#ifndef HAIM_FIXED_SEARCH_PARAMETERS
		// Start early, but reserve later perturbations across the search.
		next_lns_at = std::min(HAIM_LNS_MAX_TRIGGER, lns_interval);
#endif
#endif
		two_swap_uses = 0;
		two_swap_failures = 0;
		two_swap_allowance = 0;
		next_two_swap_at = 0;
#ifdef HAIM_TWO_SWAP_ENABLED
		two_swap_allowance = HAIM_TWO_SWAP_MAX_USES;
		next_two_swap_at = HAIM_TWO_SWAP_TRIGGER;
#endif
		after_lns = false;
	}

	void after_two_swap(bool improved_current)
	{
		++two_swap_uses; // Failed attempts also spend the budget.
#ifdef HAIM_TWO_SWAP_ENABLED
#ifdef HAIM_FIXED_SEARCH_PARAMETERS
		(void)improved_current;
		next_two_swap_at += HAIM_TWO_SWAP_TRIGGER;
#else
		two_swap_failures = improved_current ? 0
			: std::min(2, two_swap_failures + 1);
		next_two_swap_at = cntants
			+ (1LL * HAIM_TWO_SWAP_TRIGGER << two_swap_failures);
#endif
#else
		(void)improved_current;
#endif
	}

	void after_perturbation()
	{
		++lns_uses;
		next_lns_at += lns_interval;
		after_lns = true;
#if defined(HAIM_TWO_SWAP_ENABLED) && !defined(HAIM_FIXED_SEARCH_PARAMETERS)
		// Add one opportunity; do not erase the total number already spent.
		++two_swap_allowance;
		two_swap_failures = 0;
		// Leave room for 1-swap repair and a check before the next short LNS.
		int repair_steps = std::min(HAIM_TWO_SWAP_TRIGGER,
			std::max(1, lns_interval / 2));
		next_two_swap_at = cntants + repair_steps;
#endif
	}
};
#endif

bool save_search_best(int t)
{
	if (m - uncover_num <= best_value)
		return false;
	tpp = 0;
	for (int vertex = 0; vertex < n; vertex++)
		if (cs[t][vertex].is_in_c)
			Remember[tpp++] = vertex;
	assert(tpp == max_k);
	update_best_sol();
	return true;
}

// Both weighted and unweighted ants count each executed base-search call.
void ant_search_impl(int t, bool weighted_search)
{
	cntants = 0;
	tpp = 0;
#if HAIM_PERIODIC_SEARCH_ENABLED
	PeriodicSearchSchedule schedule;
#else
	LocalSearchSchedule schedule;
#endif
	for (int vertex = 0; vertex < n; vertex++)
		if (cs[t][vertex].is_in_c)
			rem_temp[tpp++] = vertex;
	assert(tpp == max_k);
	// A freshly constructed solution can already improve the incumbent.
	save_search_best(t);
#ifdef HAIM_CHECK_INVARIANTS
	check_search_state(t, max_k, true);
#endif

	while (!time_test.NeedToStop() && uncover_num > 0)
	{
#if HAIM_PERIODIC_SEARCH_ENABLED
		if (weighted_search)
			CCA_local_searchw(t);
		else
			CCA_local_search(t);
		schedule.on_base_search();
#if defined(HAIM_LNS_STATS) || defined(HAIM_TWO_SWAP_STATS)
		++base_search_calls;
#endif
#ifdef HAIM_CHECK_INVARIANTS
		check_search_state(t, max_k, true);
#endif
		if (save_search_best(t))
		{
#ifdef HAIM_LNS_STATS
			if (schedule.after_lns) ++lns_followup_best_improvements;
#endif
			cntants = 0;
			schedule.on_new_best();
		}
		else
			++cntants;

		// Save every improvement before the next operator can change the
		// current solution. A fully covered instance needs no perturbation.
		if (uncover_num == 0 || time_test.NeedToStop())
			return;
#ifdef HAIM_TWO_SWAP_ENABLED
		if (schedule.two_swap_due())
		{
#ifdef HAIM_TWO_SWAP_STATS
			double two_swap_start = time_test.GetTime();
			if (schedule.after_lns) ++two_swap_after_lns_calls;
#endif
			bool moved = try_restricted_two_swap(t, weighted_search);
			schedule.after_two_swap();
#ifdef HAIM_TWO_SWAP_STATS
			two_swap_cpu_seconds += time_test.GetTime() - two_swap_start;
#endif
			if (moved && save_search_best(t))
			{
#ifdef HAIM_TWO_SWAP_STATS
				++two_swap_best_improvements;
#endif
#ifdef HAIM_LNS_STATS
				if (schedule.after_lns) ++lns_followup_best_improvements;
#endif
				cntants = 0;
				schedule.on_new_best();
			}
		}
#endif
		if (uncover_num == 0 || time_test.NeedToStop())
			return;
#ifndef HAIM_DISABLE_LNS
		if (schedule.lns_due())
		{
#ifdef HAIM_LNS_STATS
			double lns_start = time_test.GetTime();
#endif
			lns_destroy_repair(t);
			schedule.after_perturbation();
#ifdef HAIM_LNS_STATS
			lns_cpu_seconds += time_test.GetTime() - lns_start;
#endif
			if (save_search_best(t))
			{
#ifdef HAIM_LNS_STATS
				++lns_best_improvements;
#endif
				cntants = 0;
				schedule.on_new_best();
			}
		}
#endif
#else
#ifdef HAIM_TWO_SWAP_ENABLED
		if (schedule.two_swap_uses < schedule.two_swap_allowance
			&& cntants >= schedule.next_two_swap_at)
		{
			long long one_swap_delta = 0;
			bool positive_one_swap = find_best_joint_swap(t, one_swap_delta)
				&& one_swap_delta > 0;
			if (time_test.NeedToStop())
				return;
			if (positive_one_swap)
				schedule.next_two_swap_at += HAIM_TWO_SWAP_TRIGGER;
			else
			{
#ifdef HAIM_TWO_SWAP_STATS
				double two_swap_start = time_test.GetTime();
				if (schedule.after_lns)
					++two_swap_after_lns_calls;
#endif
				bool moved = try_restricted_two_swap(t, weighted_search);
#ifdef HAIM_TWO_SWAP_STATS
				two_swap_cpu_seconds += time_test.GetTime() - two_swap_start;
#endif
				schedule.after_two_swap(moved);
				if (moved)
				{
					if (save_search_best(t))
					{
#ifdef HAIM_TWO_SWAP_STATS
						++two_swap_best_improvements;
#endif
#ifdef HAIM_LNS_STATS
						if (schedule.after_lns)
							++lns_followup_best_improvements;
#endif
						cntants = 0;
						schedule.reset();
					}
					continue;
				}
				if (time_test.NeedToStop())
					return;
			}
		}
#endif
		if (weighted_search)
			CCA_local_searchw(t);
		else
			CCA_local_search(t);
#if defined(HAIM_LNS_STATS) || defined(HAIM_TWO_SWAP_STATS)
		++base_search_calls;
#endif
#ifdef HAIM_CHECK_INVARIANTS
		check_search_state(t, max_k, true);
#endif
		if (save_search_best(t))
		{
#ifdef HAIM_LNS_STATS
			if (schedule.after_lns)
				++lns_followup_best_improvements;
#endif
			cntants = 0;
			schedule.reset();
			continue;
		}

		++cntants;
#ifndef HAIM_DISABLE_LNS
		if (schedule.lns_uses < schedule.lns_budget
			&& cntants >= schedule.next_lns_at && !time_test.NeedToStop())
		{
#ifdef HAIM_LNS_STATS
			double lns_start = time_test.GetTime();
#endif
			lns_destroy_repair(t);
#ifdef HAIM_LNS_STATS
			lns_cpu_seconds += time_test.GetTime() - lns_start;
#endif
			schedule.after_perturbation();
			if (save_search_best(t))
			{
#ifdef HAIM_LNS_STATS
				++lns_best_improvements;
#endif
				cntants = 0;
				schedule.reset();
			}
		}
#endif
#endif
		if (cntants > cca_limit)
			break;
	}
}

void ant_search(int t)
{
	ant_search_impl(t, false);
}

void ant_searchw(int t)
{
	ant_search_impl(t, true);
}

void init_search(int t)
{
	cntants = 0;
	tpp = 0;
	int r = 0;
	tpp = 0;
	for (r = 0; r < n; r++)
	{
		if (cs[t][r].is_in_c == 1)
		{
			rem_temp[tpp] = r;
			tpp++;
		}
	}
	while (1)
	{
		CCA_local_search(t);

		if (m - uncover_num > best_value)
		{
			tpp = 0;
			cntants = 0;
			for (r = 0; r < n; r++)
			{
				if (cs[t][r].is_in_c == 1)
				{
					Remember[tpp] = r;
					tpp++;
				}
			}
			update_best_sol();
		}
		else
		{
			cntants++;
			if (cntants > 100)
			{
				break;
			}
		}
		if (time_test.NeedToStop())	return;
	}
}

void init_searchw(int t)
{
	cntants = 0;
	tpp = 0;
	int r = 0;
	tpp = 0;
	for (r = 0; r < n; r++)
	{
		if (cs[t][r].is_in_c == 1)
		{
			rem_temp[tpp] = r;
			tpp++;
		}
	}
	while (1)
	{
		CCA_local_searchw(t);
		if (m - uncover_num > best_value)
		{
			tpp = 0;
			cntants = 0;
			for (r = 0; r < n; r++)
			{
				if (cs[t][r].is_in_c == 1)
				{
					Remember[tpp] = r;
					tpp++;
				}
			}
			update_best_sol();
		}
		else
		{
			cntants++;
			if (cntants > 100)
			{
				break;
			}
		}
		if (time_test.NeedToStop())	return;
	}
}
int adc = 0;
void MMAS_pro_density()
{
	valuek = std::max(1, max_k / 5);
	init_eta_tebeta();
	if (tag_init == 1)
	{
		int j = 0;
		tag_init = 0;
		uncover_num = m;
		int tempi = 8;
		FourlevelGreedy(tempi);
		for (j = 0; j < max_k; j++)
		{
			remember[j] = Remember[j];
		}
		best_temp = best_value;
		best_value = max_k;
		ANT_1();
		if (time_test.NeedToStop())
			return;
		tempi = 9;
		uncover_num = m;
		FourlevelGreedyw(tempi);
		ANT_5();
		if (time_test.NeedToStop())
			return;

		if (best_value < best_temp)
		{
			best_value = best_temp;
			best_temp = best_value;
		}

		for (j = 0; j < max_k; j++)
		{
			rtemp[j] = Remember[j];
			Remember[j] = remember[j];
		}
		ANT_2();
		if (time_test.NeedToStop())
			return;
		ANT_4();
		if (time_test.NeedToStop())
			return;
		for (j = 0; j < max_k; j++)
		{
			remember[j] = Remember[j];
			Remember[j] = rtemp[j];
		}
		int lintemp = best_value;
		best_value = best_temp;
		best_temp = lintemp;
		ANT_6();
		if (time_test.NeedToStop())
			return;
		ANT_8();
		if (time_test.NeedToStop())
			return;

		if (best_value < best_temp)
		{
			best_value = best_temp;
			best_temp = best_value;
			for (j = 0; j < max_k; j++)
			{
				Remember[j] = rtemp[j];
			}
		}
	}
	else if (tag_init == 0)
	{
		ANT_1();
		if (time_test.NeedToStop())
			return;
		ANT_5();
		if (time_test.NeedToStop())
			return;
		ANT_2();
		if (time_test.NeedToStop())
			return;
		ANT_4();
		if (time_test.NeedToStop())
			return;
		ANT_6();
		if (time_test.NeedToStop())
			return;
		ANT_8();
		if (time_test.NeedToStop())
			return;
		ANT_greedy();
		if (time_test.NeedToStop())
			return;
		
		if(greedy_limit > 10000 && smooth%2 == 1){
			smooth ++;
			greedy_limit = greedy_limit - 30000;
		}
		else
			greedy_limit = 1000;
		if(cca_limit < 10000)
			cca_limit = cca_limit*10;
		else if(cca_limit < 50000)
			cca_limit = cca_limit + 20000;
		//>100,/5;<10000,*10;<50000,+10000
		adc++;
	}
	restore_global_incumbent();
	maxmin();
	reload_alltau();
	for (int j = 0; j < max_k; j++)
	{
		int vertex = Remember[j];
		reload_bs(vertex);
		reload_alltaupro(vertex);
	}
}

void MMAS_pro()
{
	valuek = std::max(1, max_k / 5);
	init_eta_tebeta();
	if (tag_init == 1)
	{
		int j = 0;
		tag_init = 0;
		uncover_num = m;
		int tempi = 8;
		FourlevelGreedy(tempi);
		for (j = 0; j < max_k; j++)
		{
			remember[j] = Remember[j];
		}
		best_temp = best_value;
		best_value = max_k;
		ANT_1();
		if (time_test.NeedToStop())
			return;
		tempi = 9;
		uncover_num = m;
		FourlevelGreedyw(tempi);
		ANT_5();
		if (time_test.NeedToStop())
			return;

		if (best_value < best_temp)
		{
			best_value = best_temp;
			best_temp = best_value;
		}

		for (j = 0; j < max_k; j++)
		{
			rtemp[j] = Remember[j];
			Remember[j] = remember[j];
		}
		ANT_2();
		if (time_test.NeedToStop())
			return;
		ANT_3();
		if (time_test.NeedToStop())
			return;
		ANT_4();
		if (time_test.NeedToStop())
			return;

		for (j = 0; j < max_k; j++)
		{
			remember[j] = Remember[j];
			Remember[j] = rtemp[j];
		}

		int lintemp = best_value;
		best_value = best_temp;
		best_temp = lintemp;

		ANT_6();
		if (time_test.NeedToStop())
			return;

		ANT_7();
		if (time_test.NeedToStop())
			return;

		ANT_8();
		if (time_test.NeedToStop())
			return;

		if (best_value < best_temp)
		{
			best_value = best_temp;
			best_temp = best_value;
			for (j = 0; j < max_k; j++)
			{
				Remember[j] = rtemp[j];
			}
		}
	}
	else if (tag_init == 0)
	{
		ANT_1();
		if (time_test.NeedToStop())
			return;
		ANT_5();
		if (time_test.NeedToStop())
			return;
		ANT_2();
		if (time_test.NeedToStop())
			return;
		ANT_3();
		if (time_test.NeedToStop())
			return;
		ANT_4();
		if (time_test.NeedToStop())
			return;
		ANT_6();
		if (time_test.NeedToStop())
			return;
		ANT_7();
		if (time_test.NeedToStop())
			return;
		ANT_8();
		if (time_test.NeedToStop())
			return;
	}
	restore_global_incumbent();
	maxmin();
	reload_alltau();

	for (int j = 0; j < max_k; j++)
	{
		int vertex = Remember[j];
		reload_bs(vertex);
		reload_alltaupro(vertex);
	}
}


void NuAnts_density()
{
	initial_solution_saved = false;
	incumbent_valid = false;
	incumbent_value = -1;
	incumbent_time = 0.0;
	best_value = max_k;
	init();
	init_aco();
	best_temp = best_value;
	maxmin();
	for (int mnb = 0; mnb < n; mnb++)
		tau[mnb] = (taumax + taumin) / 2;
	while (1)
	{
		MMAS_pro_density();
		if (time_test.NeedToStop())	break;
	}
	restore_global_incumbent();
	return;
}


void NuAnts()
{
	initial_solution_saved = false;
	incumbent_valid = false;
	incumbent_value = -1;
	incumbent_time = 0.0;
	best_value = max_k;
	init();
	init_aco();
	best_temp = best_value;
	maxmin();
	for (int mnb = 0; mnb < n; mnb++)
		tau[mnb] = (taumax + taumin) / 2;
	while (1)
	{
		MMAS_pro();
		if (time_test.NeedToStop())	break;
	}
	restore_global_incumbent();
	return;
}


void consider_global_incumbent(int candidate_value, double found_time)
{
	if (incumbent_valid && candidate_value <= incumbent_value)
		return;

#ifdef HAIM_CHECK_INVARIANTS
	vector<unsigned char> selected(n, 0);
	for (int i = 0; i < max_k; i++)
	{
		assert(Remember[i] >= 0 && Remember[i] < n);
		assert(selected[Remember[i]] == 0);
		selected[Remember[i]] = 1;
	}
	int covered_elements = 0;
	for (int element = 0; element < m; element++)
	{
		bool is_covered = false;
		for (int h = 0; h < ncol[0][element]; h++)
			is_covered = is_covered || selected[row[0][element][h]];
		covered_elements += is_covered ? 1 : 0;
	}
	assert(covered_elements == candidate_value);
#endif

	for (int i = 0; i < max_k; i++)
		incumbent_solution[i] = Remember[i];
	incumbent_value = candidate_value;
	incumbent_time = found_time;
	incumbent_valid = true;
}

void restore_global_incumbent()
{
	if (!incumbent_valid)
		return;
	for (int i = 0; i < max_k; i++)
		Remember[i] = incumbent_solution[i];
	best_value = incumbent_value;
	best_temp = incumbent_value;
	real_time = incumbent_time;
}

void update_best_sol()
{
	if (m - uncover_num > best_value)
	{
		real_time = time_test.GetTime();
		best_value = m - uncover_num;
		consider_global_incumbent(best_value, real_time);
	}
	
}

int main(int argc, char *argv[])
{
	if (argc != 5)
	{
		fprintf(stderr, "Usage: %s instance k seed time_limit_seconds\n", argv[0]);
		return EXIT_FAILURE;
	}

	build_instance(argv[1]);
	max_k = atoi(argv[2]);
	seed = atoi(argv[3]);
	time_limit = atof(argv[4]);
	if (max_k <= 0 || max_k > n)
	{
		fprintf(stderr, "Invalid k=%d; expected 1 <= k <= %d\n", max_k, n);
		free_all();
		return EXIT_FAILURE;
	}
	if (!(time_limit > 0.0) || !std::isfinite(time_limit))
	{
		fprintf(stderr, "Invalid time limit: %s\n", argv[4]);
		free_all();
		return EXIT_FAILURE;
	}

	printf("HAIM %s  K=%d   seed=%d   t=%fs\n", argv[1], max_k, seed, time_limit);
#ifndef HAIM_LEGACY_SWAP
#ifdef HAIM_PURE_JOINT_SWAP
	printf("Swap strategy: pure joint candidate 1-swap (limit=%d)\n", JOINT_SWAP_CANDIDATE_LIMIT);
#else
	printf("Swap strategy: positive joint 1-swap + legacy diversification (limit=%d)\n",
		JOINT_SWAP_CANDIDATE_LIMIT);
#endif
#else
	printf("Swap strategy: legacy two-stage random exchange\n");
#endif
#ifdef HAIM_TWO_SWAP_ENABLED
#if HAIM_PERIODIC_SEARCH_ENABLED
	printf("2-swap intensification: exact %d-remove x %d-add candidates, every %d base-search calls\n",
		HAIM_TWO_SWAP_REMOVE_LIMIT, HAIM_TWO_SWAP_ADD_LIMIT,
		HAIM_TWO_SWAP_BASE_INTERVAL);
#else
	printf("2-swap intensification: exact %d-remove x %d-add candidates, trigger=%d\n",
		HAIM_TWO_SWAP_REMOVE_LIMIT, HAIM_TWO_SWAP_ADD_LIMIT,
		HAIM_TWO_SWAP_TRIGGER);
#endif
#else
	printf("2-swap intensification: disabled\n");
#endif
#ifndef HAIM_DISABLE_LNS
	printf("LNS diversification: %d-%d%% destroy-repair, max %d vertices, RCL=%d\n",
		HAIM_LNS_MIN_DESTROY_PERCENT, HAIM_LNS_MAX_DESTROY_PERCENT,
		HAIM_LNS_MAX_DESTROY_VERTICES, HAIM_LNS_RCL_SIZE);
#else
	printf("LNS diversification: disabled\n");
#endif
#if HAIM_PERIODIC_SEARCH_ENABLED
	printf("Search schedule: periodic per ant; identity moves count; best improvements keep counters\n");
#ifndef HAIM_DISABLE_LNS
	printf("LNS period: every %d base-search calls\n", HAIM_LNS_BASE_INTERVAL);
#endif
#else
	printf("Search schedule: finite budgets with stagnation triggers\n");
#endif
	if (m == 0 || max_k == n || max_k >= m)
	{
		best_value = m;
		real_time = 0.0;
		printf("Instance: %s\t\t K:%d\t\t seed:%d\t\t Result:%d\t\t Time:%.2f\n",
			argv[1], max_k, seed, best_value, real_time);
		free_all();
		return EXIT_SUCCESS;
	}

	delta_tau = 1.0 / sqrt((double)m);
	srand(seed);
	k_half = std::max(1, (int)(2LL * max_k / 3));
	if( density == 1)
	{
		time_test.Init();
		NuAnts_density();
	}
	else{
		time_test.Init();
		NuAnts();
	}
	restore_global_incumbent();
	
#if defined(HAIM_LNS_STATS) || defined(HAIM_TWO_SWAP_STATS)
	printf("Base search stats: calls=%lld\n", base_search_calls);
#endif
#ifdef HAIM_LNS_STATS
	printf("LNS stats: calls=%lld destroyed=%lld raw_delta_sum=%lld best_improvements=%lld cpu_seconds=%.6f\n",
		lns_calls, lns_destroyed_vertices, lns_raw_delta_sum,
		lns_best_improvements, lns_cpu_seconds);
#endif
#if defined(HAIM_TWO_SWAP_ENABLED) && defined(HAIM_TWO_SWAP_STATS)
	printf("2-swap stats: calls=%lld combinations=%lld moves=%lld raw_gain=%lld best_improvements=%lld cpu_seconds=%.6f\n",
		two_swap_calls, two_swap_combinations, two_swap_moves,
		two_swap_raw_gain, two_swap_best_improvements, two_swap_cpu_seconds);
#endif
	printf("Instance: %s\t\t K:%d\t\t seed:%d\t\t Result:%d\t\t Time:%.2f\n", argv[1], max_k, seed, best_value, real_time);
	free_all();
	return 0;
}
