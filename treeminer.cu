//============================================================================
// treeminer.cu
// 
// Main file that calls a cuda kernel to compute the SPM tree mining algorithm
//
// Invocation may be read in README.md
//
//============================================================================

// LIBRARIES
#include <string>
#include <unistd.h>
#include <stdio.h>
#include <stack>
#include <list>
#include <iostream>
#include <map>
#include <vector>
#include <set>

// HEADERS
#include "treeminer.h"
#include "timetrack.h"
#include "calcdb.h"
#include "eqclass.h"
#include "hashtree.h"
#include "stats.h"
#include "cuda.h"
#include "tools.h"

// CUDA KERNEL
#include "cuda_kernel.cu"

// GPU VARIABLES
int warp_size=0;
int shared_memory_size=0; // in bytes
int node_size=-1;
int block_dim=512;
int maxNodeSz=200;
int blk_max_size=0;

// TIMING
TimeTracker kernel_tt;
double kernel_time;
TimeTracker preproc_tt;
double preproc_time;

// GLOBAL VARIABLES
string *infile;
string *outfile = new string("summary.out");
HashTree *CandK = NULL;
FreqHT FK;
Dbase_Ctrl_Blk *DCB;
Stats stats;

typedef vector<bool> bit_vector;
int tot_trans_cnt=0; //total number of valid transactions
map<int, int> treeSz_loc_mp; //each set of tree size ends where in the DB_array

double MINSUP_PER;
int MINSUPPORT = -1;
int DBASE_MAXITEM;
int DBASE_NUM_TRANS;

// DEFAULT FLAGS
bool output_console = false; //don't print freq subtrees to console
bool count_unique = true; //count support only once per tree
sort_vals sort_type = nosort; //default is to sort in increasing order
prune_vals prune_type = prune; //prune candidates by default
set<vector<int> > freq_cand;

vector<int> *ITCNT = NULL; //used for sorting F1
bool F1cmp(int x, int y) {
	bool res = false;
	if ((*ITCNT)[x] < (*ITCNT)[y])
		res = true;

	if (sort_type == incr)
		return res;
	else
		return !res;
}


/**
*
* Parses arguments based on user set flags
*
* Flags:
*		-i,      input file of tree dataset
*		-s,      support threshold between (0,1)
*		-c,      <True> if printing the frequent subtrees to console. Default is <False>
*		-p,      <True> if pruning the database, <False> otherwise. Default is <True>
*		-u,      <True> if counting the subtree matches once per tree, <False> if weighted counting. Default is <True>
*		-o,      output file for results summary. Output is appended, not overwritten. Default is "summary.out"
*
*/
void parse_args(int argc, char **argv) {
	extern char * optarg;
	int c;

	if (argc < 5){
		cout << "usage: gpuTreeMiner -i<input_file> -s<support> -c<print output> -p<prune> -u<unique counting> -o<output_file>\n";
		cout << " -i,      dataset of trees\n";
		cout << " -s,      support threshold between (0,1)\n";
		cout << " -c,      <True> if printing the frequent subtrees to console. Default is <False> \n";
		cout << " -p,      <True> if pruning the database, <False> otherwise. Default is <True> \n";
		cout << " -u,      <True> if counting the subtree matches once per tree, <False> if weighted counting. Default is <True> \n";
		cout << " -o,      output file for results summary. Output is appended, not overwritten. Default is \"summary.out\"";
		exit(0);
	}
	else {
		while ((c = getopt(argc, argv, "bh:i:cp:s:S:uz:o:")) != -1) {
			switch (c) {
			case 'b':
				Dbase_Ctrl_Blk::binary_input = true;
				break;
			case 'h': //hash threshold
				HashTree::threshold() = atoi(optarg);
				break;
			case 'i': //input files
				infile = new string(optarg);
				break;
			case 'c': //print freq subtrees
				output_console = true;
				break;
			case 'p':
				prune_type = (prune_vals) atoi(optarg);
				break;
			case 's': //support value for L2
				MINSUP_PER = atof(optarg);
				break;
			case 'S': //absolute support
				MINSUPPORT = atoi(optarg);
				break;
			case 'u': //count support multiple times per tree
				count_unique = false;
				break;
			case 'z':
				sort_type = (sort_vals) atoi(optarg);
				break;
			case 'o':
				outfile = new string(optarg);
				break;
			}
		}
	}
}

void erase_set(set<vector<int> > &freq_set){ 
	freq_set.erase(freq_set.begin(),freq_set.end());
}

/**
*
* F1 Frequency Generation
*
* Initial frequency generation and pruning from database using minsup value
*
*/
void get_F1() {
	// SETUP TIME TRACKER
	TimeTracker tt;
	double te;

	// F1 SETUP
	int i, j, it;
	vector<int> itcnt;
	vector<int> flgs;

	tt.Start();

	DBASE_MAXITEM = 0;
	DBASE_NUM_TRANS = 0;

	while (DCB->get_next_trans()) {
		for (i = 0; i < DCB->TransSz; i++) {
			it = DCB->TransAry[i];
			if (it != BranchIt) {
				if (it >= DBASE_MAXITEM) {
					for (j = DBASE_MAXITEM; j <= it; j++) {
						itcnt.push_back(0);
						flgs.push_back(-1);
					}
					DBASE_MAXITEM = it + 1;
				}

				if (count_unique) {
					if (flgs[it] == DCB->Cid)
						continue;
					else
						flgs[it] = DCB->Cid;
				}
				itcnt[it]++;
			}
		}

		if (DCB->MaxTransSz < DCB->TransSz)
			DCB->MaxTransSz = DCB->TransSz;
		DBASE_NUM_TRANS++;
	}

	// SET VALUE OF MINSUPPORT
	if (MINSUPPORT == -1)
		MINSUPPORT = (int) (MINSUP_PER * DBASE_NUM_TRANS + 0.5);

	if (MINSUPPORT < 1)
		MINSUPPORT = 1;
	cout << "DBASE_NUM_TRANS : " << DBASE_NUM_TRANS << endl;
	cout << "DBASE_MAXITEM : " << DBASE_MAXITEM << endl;
	cout << "MINSUPPORT : " << MINSUPPORT << " (" << MINSUP_PER << ")" << endl;

	// COUNT NUMBER OF FREQUENT ITEMS
	DCB->NumF1 = 0;
	for (i = 0; i < DBASE_MAXITEM; i++)
		if (itcnt[i] >= MINSUPPORT)
			DCB->NumF1++;

	int *it_order = new int[DBASE_MAXITEM];
	for (i = 0; i < DBASE_MAXITEM; i++)
		it_order[i] = i;

	if (sort_type != nosort) {
		ITCNT = &itcnt;
		sort(&it_order[0], &it_order[DBASE_MAXITEM], F1cmp);
	}

	// CONSTRUCT FORWARD & REVERSE MAPPNG FROM ITEMS TO FREQ ITEMS
	DCB->FreqIdx = new int[DCB->NumF1];
	DCB->FreqMap = new int[DBASE_MAXITEM];
	for (i = 0, j = 0; i < DBASE_MAXITEM; i++) {
		if (itcnt[it_order[i]] >= MINSUPPORT) {
			if (output_console)
				cout << i << " - " << itcnt[it_order[i]] << endl;
			DCB->FreqIdx[j] = it_order[i];
			DCB->FreqMap[it_order[i]] = j;
			j++;
		} else
			DCB->FreqMap[it_order[i]] = -1;
	}
	
	//console output F1 summary
	cout << "F1 - " << DCB->NumF1 << " " << DBASE_MAXITEM << endl;

	if (sort_type != nosort) {
		ITCNT = NULL;
		delete[] it_order;
	}

	te = tt.Stop();
	stats.add(DBASE_MAXITEM, DCB->NumF1, te);
}

/**
*
* Output pointer array to console
*
* @param array Pointer array of type int
* @param size Size of the array
*
*/
void print_array(int* array, int size) {
	for (int i=0; i<size; i++) {
		cout << array[i] << " ";
	}
	cout << endl;
}

void get_F2() {
	int i, j;
	int it1, it2;
	int scnt;
	int tree_id=0;

	// SETUP TIMETRACKER
	TimeTracker tt;
	double te;

	// START F2 TIME
	tt.Start();
	//itcnt2 is a matrix of pairs p, p.first is count, p.second is flag
	int **itcnt2 = new int*[DCB->NumF1];
	int **flgs = new int*[DCB->NumF1];
	
	for (i = 0; i < DCB->NumF1; i++) {
		itcnt2[i] = new int[DCB->NumF1];
		flgs[i] = new int[DCB->NumF1];
		
		for (j = 0; j < DCB->NumF1; j++) {
			itcnt2[i][j] = 0;
			flgs[i][j] = -1;
		}
	}

	// CREATE DB ARRAY
	DCB->DB_array = new int*[DBASE_NUM_TRANS];
	multimap<int, int> tree_sz_mp; //key: size of the tree, value: list of tree of that size. For sorting the dataset
	vector<int>* freqCand;
	int nod_num = 0;

	while (DCB->get_next_trans()) {
		nod_num = 0;

		DCB->get_valid_trans();

		//Creating DB array with the valid transaction (removing infrequent items), the size of the transaction is decreased here
		if(DCB->TransSz > 1) {
			tot_trans_cnt++;

			DCB->DB_array[tree_id] = new int[DCB->TransSz + 2];
            DCB->DB_array_size+= DCB->TransSz+1;
			DCB->DB_array[tree_id][0] = DCB->TransSz;
			for (int trans_iter = 0; trans_iter < DCB->TransSz; trans_iter++) {
				DCB->DB_array[tree_id][trans_iter + 2] = DCB->TransAry[trans_iter];
				if(DCB->TransAry[trans_iter] != BranchIt){ //Number of nodes in one transaction
				nod_num++;
				}
			}
			DCB->DB_array[tree_id][1]=nod_num;
			tree_sz_mp.insert(pair<int,int>(nod_num,tree_id));
			tree_id++;

		//count a pair only once per cid
		for (i = 0; i < DCB->TransSz; i++) {
			it1 = DCB->TransAry[i];
			if (it1 != BranchIt && it1 != DCB->NumF1) {
				scnt = 0;

				for (j = i + 1; scnt >= 0 && j < DCB->TransSz; j++) {
					it2 = DCB->TransAry[j];
					if (it2 != BranchIt && it2 != DCB->NumF1) {
						scnt++;
						if (count_unique) {

							if (flgs[it1][it2] == DCB->Cid)
								continue;

							else
								flgs[it1][it2] = DCB->Cid;
								}
						itcnt2[it1][it2]++;
					} else
						scnt--;
				}
			}
		}
	} else
		continue;
}

		// SORT DATASET & MAKE treeSz_loc_map TO SEE TREE SIZE LOCATION IN DATABASE
		int** DB_array_tmp = new int*[tree_id];
		int loc_in_sorted_db=0;
		
		// SORT DB_array
		for(multimap<int, int>::iterator it=tree_sz_mp.begin(); it!=tree_sz_mp.end();it=tree_sz_mp.upper_bound(it->first)){
			pair<multimap<int, int>::iterator, multimap<int, int>::iterator>  eql_rng = tree_sz_mp.equal_range(it->first);
			for(multimap<int, int>::iterator it2=eql_rng.first;it2!=eql_rng.second;it2++){
				DB_array_tmp[loc_in_sorted_db] = DCB->DB_array[it2->second];
				loc_in_sorted_db++;
			}
			treeSz_loc_mp[it->first]=loc_in_sorted_db-1;
		}
		DCB->DB_array = DB_array_tmp;

	DBASE_NUM_TRANS = tree_id;

	int F2cnt = 0;

	// GENERATE NEW HASHTREE
	CandK = new HashTree(0);
	CandK->maxdepth() = 1;
	if (prune_type == prune)
		FK.clearall();
	
	// COUNT FREQUENT PATTERNS & GENERATE eqclass
	Eqclass *eq;
	for (i = 0; i < DCB->NumF1; i++) {
		eq = NULL;
		for (j = 0; j < DCB->NumF1; j++) {
			if (itcnt2[i][j] >= MINSUPPORT) {
				F2cnt++;
				if (eq == NULL) {
					eq = new Eqclass();
					eq->prefix().push_back(i);
				}
				eq->add_node(j, 0, itcnt2[i][j]);

				if (prune_type == prune){
					freqCand = new vector<int>;
					freqCand->push_back(i);
					freqCand->push_back(j);
					freq_cand.insert(*freqCand);
				}
			}
			if (eq != NULL) {
			if (output_console)
				cout << DCB->FreqIdx[i] << " " << DCB->FreqIdx[j] << " - "
						<< itcnt2[i][j] << endl;
			}
		}
		if (eq != NULL) {
			CandK->add_element(eq);
			CandK->eqlist()->push_front(eq);
			CandK->count()++;
		}
	}

	// FREE MEMORY
	for (i = 0; i < DCB->NumF1; i++) {
		delete[] itcnt2[i];
		delete[] flgs[i];
	}

	delete[] itcnt2;
	delete[] flgs;

	cout << "F2 - " << F2cnt << " " << DCB->NumF1 * DCB->NumF1 << endl;
	// LOG F2 END TIME
	te = tt.Stop();
	stats.add(DCB->NumF1 * DCB->NumF1, F2cnt, te);
}

/**
*
* Add node to Eqclass, checks for pruning
*
*/
void add_node(int iter, Eqclass *neq, int val, int pos) {
	if (prune_type == noprune) {
		//don't do any pruning
		neq->add_node(val, pos);
		return;
	}

	// PRUNE BASED ON FREQUENT SUBTREE
	static vector<int> cand;
	static vector<int> subtree;

	int hval;
	int scope, scnt;

	// FORM THE CANDIDATE PREFIX
	cand = neq->prefix();
	scnt = neq->get_scope(pos, scope); //checks the scope of node.pos

	while (scnt > scope) {
		cand.push_back(BranchIt);
		scnt--;
	}
	cand.push_back(val);


	int cnt=0;
	vector<int> candTmp;
	vector<int>::iterator it1, it2;
	
	// PRUNING
	candTmp = cand;
	int num_root_visiting=0; //used for checking if deleting root node or not

    // CHECKING THE ROOT
    if(find(candTmp.begin(), candTmp.end(), BranchIt) == candTmp.end()){
    	for(it1=candTmp.begin(); it1 != candTmp.end()-1; it1++){
    		candTmp.erase(it1);
    		if(freq_cand.find(candTmp) == freq_cand.end()){
				return;
			}
    		candTmp.clear();
    		candTmp = cand;
    	}
    }

    // CHECKING THE ROOT CANDIDATE
	candTmp.clear();
	candTmp = cand;
    cnt=0;
    it1=candTmp.begin();
    for(it2=candTmp.begin()+1; it2 != candTmp.end(); it2++){
		if(*it2 != BranchIt){
			cnt++;
		}
		else{
			cnt--;
			if(cnt == 0)
				num_root_visiting++;
		}
	}
    if(num_root_visiting==0){
		candTmp.erase(it1);
		if(freq_cand.find(candTmp) == freq_cand.end()){
			return;
		}
		candTmp.clear();
		candTmp = cand;
	}

    // CHECKING THE REST OF THE NODES
	for(it1=candTmp.begin()+1; it1 != candTmp.end()-1; it1++){
		if(*it1 != BranchIt){
			cnt=0;
			for(it2=it1+1; it2 != candTmp.end(); it2++){
				if(*it2 != BranchIt){
					cnt++;
				}
				else{
					cnt--;
					if(cnt==-1){
						candTmp.erase(it1);
						candTmp.erase(it2-1);
						break;
					}
				}
			}
		if(candTmp.size() == cand.size()){
			candTmp.erase(it1);
		}
		if(freq_cand.find(candTmp) == freq_cand.end()){
			return;
		}
		candTmp.clear();
		candTmp = cand;
        }
	}
	//otherwise add the node
	neq->add_node(val, pos);
}

void cand_gen(int iter, Eqclass &eq, list<Eqclass *> &neweql) {
	Eqclass *neq;
	list<Eqnode>::iterator ni, nj;

	for (ni = eq.nlist().begin(); ni != eq.nlist().end(); ++ni) {
		neq = NULL;
		for (nj = eq.nlist().begin(); nj != eq.nlist().end(); ++nj) {
			if (ni->pos < nj->pos)
				continue;
			if (neq == NULL) {
				neq = new Eqclass;

				neq->set_prefix(eq.prefix(), *ni);
			}
			if (ni->pos > nj->pos)
				add_node(iter, neq, nj->val, nj->pos);
			else { 
				add_node(iter, neq, nj->val, nj->pos);
				add_node(iter, neq, nj->val, neq->prefix().size() - 1);
			}
		}
		if (!neq->nlist().empty()) {
			neweql.push_back(neq);
		} else
			delete neq;
		}
}

void candidate_generation(int iter, HashTree *ht, int &candcnt) {
	list<Eqclass *> *oldeql = ht->eqlist();
	list<Eqclass *> *neweql = new list<Eqclass *>;
	Eqclass *eq;

	ht->flag() = -1; //reset the flag
	while (!oldeql->empty()) {
		eq = oldeql->front();

		cand_gen(iter, *eq, *neweql);
		delete eq;
		ht->count()--;
		oldeql->pop_front();
	}

	list<Eqclass *>::iterator ni;
	for (ni = neweql->begin(); ni != neweql->end(); ni++) {
		ht->eqlist()->push_back(*ni);
		ht->count()++;
		candcnt += (*ni)->nlist().size();
	}

	delete neweql;
}

ostream & operator<<(ostream& fout, vector<int> &vec) {
	fout << vec[0];
	for (int i = 1; i < vec.size(); i++)
		fout << " " << vec[i];
	return fout;
}

bool incr_nodes(Eqclass *eq, int tpos, int tscope, stack<int> &stk,
		bit_vector &cflgs) {
	int i, f, st, en, l;
	bool retval = false;
	int fcnt = 0;
	int scope, ttscope, ttpos;
	stack<int> tstk;

	list<Eqnode>::iterator ni = eq->nlist().begin();
	for (f = 0; ni != eq->nlist().end(); ni++, f++) {

		//if unique counts and node has been counted, skip to next node
		if (count_unique && cflgs[f]) {
			fcnt++;
			continue;
		}

		ttscope = tscope;
		scope = ttscope;
		ttpos = tpos;
		bool skip = false;
		int st, en;
		en = eq->get_scope(ni->pos, st);
		if (en > st) {
			skip = true;
			while (en > st) {
				st++;
				tstk.push(stk.top());
				stk.pop();
			}
			ttscope = tstk.top();
		}

		while (skip && scope >= ttscope && ttpos < DCB->DB_array[DCB->db_iter][0]) {
			if (DCB->DB_array[DCB->db_iter][ttpos + 1] == BranchIt)
				scope--;
			else
				scope++;
			ttpos++;
		}

		if (skip)
			ttscope = stk.top();
		//search for the last item within cur_scope
		for (i = ttpos; i < DCB->DB_array[DCB->db_iter][0]; i++) {
			if (DCB->DB_array[DCB->db_iter][i + 1] == BranchIt)
				scope--;
			else
				scope++;

			if (scope < ttscope)
				break;

			if (ni->val == DCB->DB_array[DCB->db_iter][i + 1]) {
				if (count_unique) {
					if (!cflgs[f]) {
						cflgs[f] = true;
						fcnt++;
						ni->sup++;
					}
				} else
					ni->sup++;
			}
		}

		while (!tstk.empty()) {
			stk.push(tstk.top());
			tstk.pop();
		}
	}

	//all nodes have been seen
	if (count_unique && fcnt == cflgs.size())
		retval = true;

	return retval;
}

bool incr_support(Eqclass *eq, int tpos, int ppos, int tscope, stack<int> &stk,
		bit_vector &cflgs) {
	int i;
	int scope, ttscope;
	stack<int> tstk;

	scope = tscope;
	bool skip = false;
	if (eq->prefix()[ppos] == BranchIt) {
		skip = true;
		while (eq->prefix()[ppos] == BranchIt) {
			tstk.push(stk.top());
			stk.pop();
			ppos++;
		}
		tscope = tstk.top();
	}

	while (skip && scope >= tscope && tpos < DCB->DB_array[DCB->db_iter][0]) {
		if (DCB->DB_array[DCB->db_iter][tpos + 1] == BranchIt)
			scope--;
		else
			scope++;
		tpos++;
	}

	if (skip)
		tscope = stk.top();

	bool allfound = false;

	for (i = tpos; i < DCB->DB_array[DCB->db_iter][0] && !allfound; i++) {
		if (DCB->DB_array[DCB->db_iter][i + 1] == BranchIt)
			scope--;
		else
			scope++;
		if (scope < tscope)
			break;
		if (DCB->DB_array[DCB->db_iter][i + 1] == eq->prefix()[ppos]) {
			stk.push(scope);

			if (ppos == eq->prefix().size() - 1) {
				allfound = incr_nodes(eq, i + 1, scope, stk, cflgs);
			} else {
				allfound = incr_support(eq, i + 1, ppos + 1, scope, stk, cflgs);
			}
			stk.pop();
		}

	}

	while (!tstk.empty()) {
		stk.push(tstk.top());
		tstk.pop();
	}
	return allfound;
}


/**
*
* Returns false if Eqnode support is greater than or equal to the minimum support threshold, true if otherwise
*
*/
static bool notfrequent(Eqnode &n) {
	if (n.sup >= MINSUPPORT)
		return false;
	else
		return true;
}

bool get_frequent(int iter, HashTree *ht, int &freqcnt) {
	int i;

	bool empty_leaf = false;

	if (ht->isleaf()) {
		list<Eqclass *> *eql = ht->eqlist();
		Eqclass *eq;
		list<Eqclass *>::iterator ni;

		for (ni = eql->begin(); ni != eql->end() && !eql->empty();) {
			eq = *ni;

			list<Eqnode>::iterator nj;
			nj = remove_if(eq->nlist().begin(), eq->nlist().end(), notfrequent);
			eq->nlist().erase(nj, eq->nlist().end());

			freqcnt += eq->nlist().size();

			if (output_console && !eq->nlist().empty())
				eq->print(DCB);

			if (eq->nlist().empty()) {
				ni = eql->erase(ni);
				CandK->count()--;
			}
			else {
				if (prune_type == prune)
					FK.add(eq);
				ni++;
			}
		}
		if (eql->empty())
			empty_leaf = true;
	} else {
		HTable::iterator ti, hi = ht->htable().begin();
		int ecnt = 0;
		for (; hi != ht->htable().end();) {
			bool ret = get_frequent(iter, (*hi).second, freqcnt);
			if (ret) {
				ecnt++;
				ti = hi;
				hi++;
				ht->htable().erase(ti);
			} else
				hi++;
		}
	}

	return empty_leaf;
}

int* create_cand_array(int candcnt, int iter){
	int* cand_array = new int[candcnt*(2*iter-1)];
	int cand_array_it = 0;
	int lastBranchDepth = 0;
	list<Eqclass *> *eql = CandK->eqlist();
	list<Eqclass *>::iterator ei;
	vector<int> righ_path_pos; //position in the prefix vector

	for(ei = eql->begin(); ei != eql->end(); ei++){
		Eqclass* eq = *ei;
		list<Eqnode>::iterator ni;

		// CREATE RIGHTMOST PATH
		righ_path_pos.clear();
		for(int i=0; i<eq->prefix().size();i++){
			if(eq->prefix()[i] != BranchIt){
				righ_path_pos.push_back(i);
			}
			else{
				righ_path_pos.pop_back();
			}
		}

		for(ni=eq->nlist().begin(); ni!=eq->nlist().end(); ni++){
			//copy the prefix
			for(int i=0; i<eq->prefix().size();i++){
				cand_array[cand_array_it] = eq->prefix()[i];
				cand_array_it++;
			}

			//add the extension node and the remaining branches
			for(int i=righ_path_pos.size()-1; i>-1;i--){
				if(ni->pos == righ_path_pos[i]){
					cand_array[cand_array_it] = ni->val;
					cand_array_it++;
					cand_array[cand_array_it] = BranchIt;
					cand_array_it++;
				}
				else{
					cand_array[cand_array_it] = BranchIt;
					cand_array_it++;
				}
			}
		}
	}
	return cand_array;
}


/**
*
* Receive GPU results & update their support
*
* @param candcnt Candidate count
* @param freqcnt Frequency count
* @param gpu_result Results from GPU kernel invocation
*
*/
void update_sup(int& candcnt, int& freqcnt, int* gpu_result){

	list<Eqclass *> *eql = CandK->eqlist();
	list<Eqclass *>::iterator ei;
	int candIt=0;

	for(ei = eql->begin(); ei != eql->end() && !eql->empty();){
		Eqclass* eq = *ei;
		list<Eqnode>::iterator ni;

		for(ni=eq->nlist().begin(); ni!=eq->nlist().end(); ni++){
			ni->sup = gpu_result[candIt];
			candIt++;
		}

		// CHECK FREQUENCY
		list<Eqnode>::iterator nj;
		list<Eqnode>::iterator njj;
		nj = remove_if(eq->nlist().begin(), eq->nlist().end(), notfrequent);
		eq->nlist().erase(nj, eq->nlist().end());
		vector<int>* cand;
		int cnt;
		int node_tmp;
		int depth_pos=0;//the depth position of the parent node of the extension node


		freqcnt += eq->nlist().size();
		if (eq->nlist().empty()) {
			ei = eql->erase(ei);
			CandK->count()--;
		}
		else {
			//push to Fk
			if (prune_type == prune){
				for(njj = eq->nlist().begin(); njj !=  eq->nlist().end(); njj++){
					cnt = -1;
					cand = new vector<int>;
					for(int j=0; j< eq->prefix().size();j++){
					   node_tmp = eq->prefix()[j];
						if(node_tmp == BranchIt)
							cnt--;
						else
							cnt++;
						if(j == njj->pos)
                    		depth_pos = cnt;
						(*cand).push_back(eq->prefix()[j]);
					}
					while(cnt != depth_pos){
						(*cand).push_back(BranchIt);
						cnt--;
					}
					(*cand).push_back(njj->val);
					freq_cand.insert((*cand));
				}
			}
			ei++;
		}

	}

	if(candIt != candcnt){
		cerr << "Error in updating the support" << endl;
		exit(0);
	}
}



void get_Fk() {
	// Fk SETUP
	int candcnt=0, freqcnt=0;

	// SETUP TIME TRACKER
	TimeTracker tt;
	double te;

	////////////////////////////
	/////////GPU////////////////
	int* trees_d;
	int* tr_start_ind_d;
	int* freq_result_d;
	int* cand_d;
	int* cand_h;

	// DEVICE MEMORY ALLOCATION AND COPYING FROM DATABASE
	ERROR_CHECK(cudaMalloc(&trees_d, DCB->DB_array_size*sizeof(int)));
	ERROR_CHECK(cudaMemcpy(trees_d, DCB->trees_h, DCB->DB_array_size*sizeof(int), cudaMemcpyHostToDevice));

	ERROR_CHECK(cudaMalloc(&tr_start_ind_d, DBASE_NUM_TRANS*sizeof(int)));
	ERROR_CHECK(cudaMemcpy(tr_start_ind_d, DCB->tr_start_ind_h, DBASE_NUM_TRANS*sizeof(int), cudaMemcpyHostToDevice));

	for (int iter = 3; !CandK->isempty(); iter++) {
		tt.Start();
		CandK->maxdepth() = iter - 1;
		candcnt = 0;
		freqcnt = 0;

		// GENERATE LIST OF CANDIDATES
		candidate_generation(iter, CandK, candcnt);

		cand_h = create_cand_array(candcnt,iter);

		if (candcnt > 0) {

			// START KERNEL RUN TIME
			kernel_tt.Start();
			ERROR_CHECK(cudaMalloc(&cand_d, (2*iter-1)*candcnt*sizeof(int)));
			ERROR_CHECK(cudaMemcpy(cand_d, cand_h, (2*iter-1)*candcnt*sizeof(int), cudaMemcpyHostToDevice));

			ERROR_CHECK(cudaMallocManaged(&freq_result_d, candcnt*sizeof(int)));
			ERROR_CHECK(cudaMemset(freq_result_d, 0, candcnt*sizeof(int)));

			// SET GPU MEMORY DIMENSIONS (BLOCK SIZE, GRID SIZE, CONSTANT MEMORY SIZE)
			int threadNum = block_dim;

			int blockNum = (DBASE_NUM_TRANS/threadNum > 65535) ? 65535 : (DBASE_NUM_TRANS-1)/threadNum+1;

			frequency_counter<<<blockNum,threadNum>>>(trees_d, tr_start_ind_d,
					DBASE_NUM_TRANS, iter, candcnt, cand_d, freq_result_d);

			if ((cudaDeviceSynchronize()) != cudaSuccess) printf("error in cuda device synchronization\n");

			// STOP & APPEND KERNEL RUN TIME TO TOTAL KERNEL RUN TIME
			kernel_time += kernel_tt.Stop();

			if (prune_type == prune)
				FK.clearall();

			if (prune_type == prune){
				erase_set(freq_cand);
			}
			update_sup(candcnt,freqcnt,freq_result_d);

			// FREE DEVICE MEMORY
			cudaFree(freq_result_d);
			cudaFree(cand_d);
		}
		cout << "F" << iter << " - " << freqcnt << " " << candcnt << endl;

		te = tt.Stop();
		stats.add(candcnt, freqcnt, te);
	}
	if (prune_type == prune){
		erase_set(freq_cand);
	}
}

void create_gpu_stats(){
    
    //making DB for GPU -- map to "trees" variable for GPU
    //AND making start point of trees (tr_start_ind_h), the index consider the size of the tree as well
    
    int trees_h_it=0;
    int tr_start_ind_h_it=0;
    
    DCB->trees_h = new int[DCB->DB_array_size];
    DCB->tr_start_ind_h = new int[DBASE_NUM_TRANS];
    
    for(int i=0; i<DBASE_NUM_TRANS; i++){
        DCB->trees_h[trees_h_it] = DCB->DB_array[i][0];
        DCB->tr_start_ind_h[tr_start_ind_h_it] = trees_h_it;
        tr_start_ind_h_it++;
        trees_h_it++;

        for(int j=0; j<DCB->DB_array[i][0]; j++){
            DCB->trees_h[trees_h_it] = DCB->DB_array[i][j+2];
            trees_h_it++;
            
            if(trees_h_it>DCB->DB_array_size){
                cerr << "Error in creating DB for GPU" << endl;
                exit(0);
            }
        }
    }
}

int main(int argc, char **argv) {

	// START RUN TIME
	TimeTracker tt;
	tt.Start();

	// PARSE ARGUMENTS TO VARIABLES
	parse_args(argc, argv);

	// GENERATE DATABASE AND GET F1 & F2 SEQUENCES
	DCB = new Dbase_Ctrl_Blk(infile->c_str());
	get_F1();
	get_F2();

	// GET DEVICE PROPERTIES
	cudaDeviceProp  prop;
	cudaGetDeviceProperties( &prop, 0 );
	shared_memory_size=prop.sharedMemPerBlock;
	warp_size=prop.warpSize;

	// GENERATE GPU STATS
	create_gpu_stats();

	// RUN TO GET Fk SEQUENCES
	get_Fk();

	// END RUN TIME
 	double tottime = tt.Stop();
	stats.tottime = tottime;

	// WRITE RESULTS TO SUMMARY FILE
	ofstream summary(outfile->c_str(), ios::app);
	summary << "HTREEMINER ";
	switch (sort_type) {
	case incr:
		summary << "INCR ";
		break;
	case decr:
		summary << "DECR ";
		break;
	default:
		break;
	}
	switch (prune_type) {
	case prune:
		summary << "PRUNE ";
		break;
		deafult: break;
	}
	if (!count_unique)
		summary << "MULTIPLE ";

	summary << *infile << " " << MINSUP_PER << " " << DBASE_NUM_TRANS << " "
			<< MINSUPPORT << " ";
	summary << stats << endl;
	summary.close();


	// PRINT RESULTS TO CONSOLE
	cout << stats << endl;
	cout << "TIME = " << tottime << endl;

	cout << endl << "Total time = " << tottime << endl;
	cout << "Kernel time = " << kernel_time << endl;
	cout << "Pre-proc time = " << tottime - kernel_time << endl;

	// EXIT SUCCESSFULLY
	exit(0);
}

