// LIBRARIEs
#include <stack>

// HEADERS
#include "eqclass.h"
#include "treeminer.h"

// Eqnode
Eqnode::Eqnode(int v, int p, int s):
   val(v), pos(p), sup(s){}
 
Eqclass::Eqclass() {
}

void Eqclass::add_node(int val, int pos, int sup){
	Eqnode *eqn = new Eqnode(val,pos,sup);
	_nodelist.push_back(*eqn);
}


int Eqclass::item(int n) {
	int i,k;
	for (i=0,k=0; i < _prefix.size(); i++){
		if (_prefix[i] != BranchIt){
			if (k == n) return _prefix[i];
			k++;
		}
	}
	return -1;
}

void Eqclass::set_prefix(vector<int> &pref, Eqnode &node){
   int i;
   _prefix = pref;
   
   int scope, scnt;
   scnt = get_scope(node.pos, scope); // checks scope of node.pos

   while(scnt > scope){ 
     _prefix.push_back(BranchIt);
     scnt--;
   }
   _prefix.push_back(node.val);
}

/**
*
* Returns the scope of the prefix
*
*/
int Eqclass::get_scope(int pos, int &scope){
	int scnt=0;
	for (int i=0; i < _prefix.size(); i++){
	if (_prefix[i] == BranchIt) scnt--;
	else scnt++;
	if (i == pos) scope = scnt;
	}
	return scnt;
}

/**
*
* Returns the parent position of the node at pos; pos must be within range
*
*/
int Eqclass::child_of (int pos){
	int i, k, retval;
	static stack<int> stk;
  
	for (i=0, k=0; i < pos; i++){
	if (_prefix[i] == BranchIt) stk.pop();
	else{
		stk.push(k);
		k++;
	}
	}
	if (stk.empty()) retval = 0;
	else retval = stk.top();
	while (!stk.empty()) stk.pop();
	return retval;
}

/**
*
* Print with items remapped to their original value
*
*/
void Eqclass::print(Dbase_Ctrl_Blk *DCB){
	list<Eqnode>::iterator ni = _nodelist.begin();
  
	int st, en;
	for (; ni != _nodelist.end(); ni++){
	for (int i=0; i < _prefix.size(); i++){
		if (_prefix[i] == BranchIt) cout << BranchIt << " ";
		else cout << DCB->FreqIdx[_prefix[i]] << " ";
	}
	en = get_scope(ni->pos, st);
	while (en > st){
		st++;
		cout << BranchIt << " ";
	}
	cout << DCB->FreqIdx[ni->val] << " - " << ni->sup << endl;    
	}
}

/**
*
* Output stream for the eqclass
*
*/
ostream& operator << (ostream& fout, Eqclass& eq){
  int i;

  list<Eqnode>::iterator ni = eq._nodelist.begin();

  int st, en;
  for (; ni != eq._nodelist.end(); ni++){
     fout << eq.prefix();

    en = eq.get_scope(ni->pos, st);
    while (en > st){
      st++;
      fout << " " << BranchIt;
    }

    fout << " " << ni->val << " - " << ni->sup << endl;    
  }
  return fout;
}
 
