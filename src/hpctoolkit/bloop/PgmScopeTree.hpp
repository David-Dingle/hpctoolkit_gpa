// -*-Mode: C++;-*-
// $Id$
// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002, Rice University 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage. 
// 
// ******************************************************* EndRiceCopyright *

//***************************************************************************
//
// File:
//    $Source$
//
// Purpose:
//    [The purpose of this file]
//
// Description:
//    [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

#ifndef PgmScopeTree_H 
#define PgmScopeTree_H

//************************* System Include Files ****************************

#include <iostream>
#include <list> // STL
#include <set>  // STL

//*************************** User Include Files ****************************

#include <include/general.h>

#include <lib/support/NonUniformDegreeTree.h>
#include <lib/support/Unique.h>
#include <lib/support/String.h>
#include <lib/support/Files.h>
#include <lib/support/Nan.h>

//*************************** Forward Declarations ***************************

class ScopeInfo;

// Some possibly useful containers
typedef std::list<ScopeInfo*> ScopeInfoList;
typedef std::set<ScopeInfo*> ScopeInfoSet;

//*************************** Forward Declarations ***************************

int AddXMLEscapeChars(int dmpFlag);

const suint UNDEF_LINE = 0;

class LoadModScopeMap;
class FileScopeMap;
class ProcScopeMap;

//***************************************************************************
// PgmScopeTree
//***************************************************************************

class PgmScope;

class PgmScopeTree: public Unique {
public:
  enum {
    // User-level bit flags
    XML_FALSE =	(0 << 0),	/* No XML format */
    XML_TRUE  =	(1 << 0),	/* XML format */

    COMPRESSED_OUTPUT = (1 << 1),  /* Use compressed output format */

    // Not-generally-user-level bit flags
    XML_NO_ESC_CHARS = (1 << 10), /* don't substitute XML escape characters */

    // Private bit flags
    XML_EMPTY_TAG    = (1 << 15)  /* this is an empty XML tag */
    
  };

public:
  // Constructor/Destructor
  PgmScopeTree(PgmScope* _root = NULL);
  virtual ~PgmScopeTree();

  // Tree data
  PgmScope* GetRoot() const { return root; }
  bool      IsEmpty() const { return (root == NULL); }

  void SetRoot(PgmScope* x) { root = x; }
  
  // Dump contents for inspection (use flags from ScopeInfo)
  virtual void Dump(std::ostream& os = std::cerr, 
		    int dmpFlag = XML_TRUE) const;
  virtual void DDump() const;
 
private:
  PgmScope* root;
};

//***************************************************************************
// ScopeInfo, CodeInfo.
//***************************************************************************

// FIXME: It would make more sense for LoadModScope to simply be a
// ScopeInfo and not a CodeInfo, but the assumption that *only* a
// PgmScope is not a CodeInfo is deeply embedded and would take a
// while to untangle.

class ScopeInfo;   // Base class for all scopes
class CodeInfo;    // Base class for everyone but PGM

class PgmScope;    // Tree root
class GroupScope;
class LoadModScope;
class FileScope;
class ProcScope;
class LoopScope;
class StmtRangeScope;

// ---------------------------------------------------------
// ScopeInfo: The base node for a program scope tree
// ---------------------------------------------------------
class ScopeInfo: public NonUniformDegreeTreeNode, public Unique {
public:
  enum ScopeType {
    PGM,
    GROUP,
    LM,
    FILE,
    PROC,
    LOOP,
    STMT_RANGE,
    ANY,
    NUMBER_OF_SCOPES
  };

  static const char* ScopeTypeToName(ScopeType tp); 
  static ScopeType   IntToScopeType(long i);

private:
  static const char* ScopeNames[NUMBER_OF_SCOPES];
  
public:
  ScopeInfo(ScopeType type, ScopeInfo* parent);
  virtual ~ScopeInfo(); 
  
  // --------------------------------------------------------
  // General Interface to fields 
  // --------------------------------------------------------
  ScopeType     Type() const         { return type; };
  unsigned int  UniqueId() const     { return uid; };

  // Name() is overwritten by some Scopes
  virtual String Name() const        { return ScopeTypeToName(Type()); };
  
  // --------------------------------------------------------
  // Parent
  // --------------------------------------------------------
  ScopeInfo *Parent() const 
                { return (ScopeInfo*) NonUniformDegreeTreeNode::Parent(); };
  
  CodeInfo  *CodeInfoParent() const;  // return dyn_cast<CodeInfo*>(Parent())
  
  // --------------------------------------------------------
  // Ancestor: find first ScopeInfo in path from this to root with given type
  // --------------------------------------------------------
  ScopeInfo* Ancestor(ScopeType type) const; 
  
  PgmScope*       Pgm() const;           // return Ancestor(PGM)
  GroupScope*     Group() const;         // return Ancestor(GROUP)
  LoadModScope*   LoadMod() const;       // return Ancestor(LM)
  FileScope*      File() const;          // return Ancestor(FILE)
  ProcScope*      Proc() const;          // return Ancestor(PROC)
  LoopScope*      Loop() const;          // return Ancestor(LOOP)
  StmtRangeScope* StmtRange() const;     // return Ancestor(STMT_RANGE)

  // Note: We assume that a node cannot be an ancestor of itself.

  // LeastCommonAncestor: Given two ScopeInfo nodes, return the least
  // common ancestor (deepest nested common ancestor) or NULL.
  static ScopeInfo* LeastCommonAncestor(ScopeInfo* n1, ScopeInfo* n2);

  // --------------------------------------------------------
  // Tree navigation 
  //   1) all ScopeInfos contain CodeInfos as children 
  //   2) PgmRoot is the only ScopeInfo type that is not also a CodeInfo; 
  //      since PgmScopes have no siblings, it is safe to make Next/PrevScope 
  //      return CodeInfo pointers 
  // --------------------------------------------------------
  CodeInfo* FirstEnclScope() const;      // return  FirstChild()
  CodeInfo* LastEnclScope()  const;      // return  LastChild()
  CodeInfo* NextScope()      const;      // return  NULL or NextSibling()
  CodeInfo* PrevScope()      const;      // return  NULL or PrevSibling()
  bool      IsLeaf()         const       { return  FirstEnclScope() == NULL; }

  // --------------------------------------------------------
  // Paths and Merging
  // --------------------------------------------------------

  // Distance: Given two ScopeInfo nodes, a node and some ancestor,
  // return the distance of the path between the two.  The distance
  // between a node and its direct ancestor is 1.  If there is no path
  // between the two nodes, returns a negative number; if the two
  // nodes are equal, returns 0.
  static int Distance(ScopeInfo* ancestor, ScopeInfo* descendent);

  // ArePathsOverlapping: Given two nodes and their least common
  // ancestor, lca, returns whether the paths from the nodes to lca
  // overlap.
  //
  // Let d1 and d2 be two nodes descended from their least common
  // ancestor, lca.  Furthermore, let the path p1 from d1 to lca be as
  // long or longer than the path p2 from d2 to lca.  (Thus, d1 is
  // nested as deep or more deeply than d2.)  If the paths p1 and p2 are
  // overlapping then d2 will be somewhere on the path between d1 and
  // lca.
  //
  // Examples: 
  // 1. Overlapping: lca --- d2 --- ... --- d1
  //
  // 2. Divergent:   lca --- d1
  //                    \--- d2
  //
  // 3. Divergent:   lca ---...--- d1
  //                    \---...--- d2
  static bool ArePathsOverlapping(ScopeInfo* lca, ScopeInfo* desc1, 
				  ScopeInfo* desc2);
  
  // MergePaths: Given divergent paths (as defined above), merges the path
  // from 'toDesc' into 'fromDesc'. If a merge takes place returns true.
  static bool MergePaths(ScopeInfo* lca, 
			 ScopeInfo* toDesc, ScopeInfo* fromDesc);
  
  // Merge: Given two nodes, 'fromNode' and 'toNode', merges the
  // former into the latter, if possible.  If the merge takes place,
  // deletes 'fromNode' and returns true; otherwise returns false.
  static bool Merge(ScopeInfo* toNode, ScopeInfo* fromNode);

  // IsMergable: Returns whether 'fromNode' is capable of being merged
  // into 'toNode'
  static bool IsMergable(ScopeInfo* toNode, ScopeInfo* fromNode);
  
  // --------------------------------------------------------
  // debugging and printing 
  // --------------------------------------------------------
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const; 
  virtual String Types() ; // lists this instance's base and derived types 
  
  void DumpSelfBefore(std::ostream &os = std::cerr, 
		      int dmpFlag = PgmScopeTree::XML_TRUE,
		      const char* prefix = "") const;
  void DumpSelfAfter (std::ostream &os = std::cerr, 
		      int dmpFlag = PgmScopeTree::XML_TRUE,
		      const char* prefix = "") const;
  void Dump          (std::ostream &os = std::cerr, 
		      int dmpFlag = PgmScopeTree::XML_TRUE,
		      const char* pre = "") const;
  void DumpLineSorted(std::ostream &os = std::cerr, 
		      int dmpFlag = PgmScopeTree::XML_TRUE,
		      const char* pre = "") const;

  void DDump();     // stupid SGI dbx...
  void DDumpSort(); // stupid SGI dbx...
  
private: 
  ScopeType type;
  unsigned int uid; 
};

// ---------------------------------------------------------
// CodeInfo is a base class for all scopes other than PGM and LM.
// Describes some kind of code, i.e. Files, Procedures, Loops...
// ---------------------------------------------------------
class CodeInfo : public ScopeInfo {
protected: 
  CodeInfo(ScopeType t, ScopeInfo* mom, 
	   suint begLn = UNDEF_LINE, suint endLn = UNDEF_LINE); 
  
public: 
  virtual ~CodeInfo();

  suint  BegLine() const { return begLine; } // in source code
  suint  EndLine() const { return endLine; } // in source code
  
  bool      ContainsLine(suint ln) const; 
  CodeInfo* CodeInfoWithLine(suint ln) const; 

  // returns a string of the form: 
  //   File()->Name() + ":" + <Line-Range> 
  //
  // where Line-Range is either: 
  //                     StartLine() + "-" + StopLine()      or simply 
  //                     StartLine() 
  virtual String CodeName() const; 

  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const;
  virtual String DumpLineRange(int dmpFlag = PgmScopeTree::XML_TRUE) const;

  void SetLineRange(suint begLn, suint endLn); // be careful when using!
  
protected: 
  void Relocate(); 
  suint begLine; 
  suint endLine; 
}; 

// - if x < y; 0 if x == y; + otherwise
// N.B.: in the case that x == y and x and y are ProcScopes, sort by name.
int CodeInfoLineComp(CodeInfo* x, CodeInfo* y);

//***************************************************************************
// PgmScope, GroupScope, LoadModScope, FileScope, ProcScope, LoopScope,
// StmtRangeScope
//***************************************************************************

// ---------------------------------------------------------
// PgmScope is root of the scope tree
// ---------------------------------------------------------
class PgmScope: public ScopeInfo {
public: 
  PgmScope(const char* pgmName); 
  virtual ~PgmScope(); 

  String Name() const { return name; }

  // find by 'realpath'
  LoadModScope* FindLoadMod(const char* nm) const;
  FileScope*    FindFile(const char* nm) const;

  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const; 

  void Freeze() { frozen = true;} // disallow additions to/deletions from tree
  bool IsFrozen() const { return frozen; } 

  void DumpLineSorted(std::ostream &os = std::cerr, 
		      int dmpFlag = PgmScopeTree::XML_TRUE,
		      const char *pre = "") const;
  
protected: 
private: 
  void AddToLoadModMap(LoadModScope& lm);
  void AddToFileMap(FileScope& file);
  friend class LoadModScope;   
  friend class FileScope; 

  bool frozen;
  String name;            // the program name
  LoadModScopeMap* lmMap; // mapped by 'realpath'
  FileScopeMap* fileMap;  // mapped by 'realpath'
}; 

// ---------------------------------------------------------
// GroupScopes are children of PgmScope's, GroupScope's, LoadModScopes's, 
//   FileScope's, ProcScope's, LoopScope's
// children: GroupScope's, LoadModScope's, FileScope's, ProcScope's,
//   LoopScope's, StmtRangeScopes,
// They may be used to describe several different types of scopes
//   (including user-defined ones)
// ---------------------------------------------------------
class GroupScope: public CodeInfo {
public: 
  GroupScope(const char *grpName, CodeInfo *mom, int begLn, int endLn); 
  virtual ~GroupScope();
  
  String Name() const                  { return name; } // same as grpName
  
  virtual String CodeName() const;
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const;

private: 
  String name; 
};

// ---------------------------------------------------------
// LoadModScopes are children of PgmScope's or GroupScope's
// children: GroupScope's, FileScope's
// ---------------------------------------------------------
// FIXME: See note about LoadModScope above.
class LoadModScope: public CodeInfo {
public: 
  LoadModScope(const char* lmName, ScopeInfo* mom);
  virtual ~LoadModScope(); 

  String Name() const { return name; }

  virtual String CodeName() const;
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const;
  
protected: 
private: 
  String name; // the load module name
}; 

// ---------------------------------------------------------
// FileScopes are children of PgmScope's, GroupScope's and LoadModScope's.
// children: GroupScope's, ProcScope's, LoopScope's, or StmtRangeScope's.
// FileScopes may refer to an unreadable file
// ---------------------------------------------------------
class FileScope: public CodeInfo {
public: 
  FileScope(const char *fileNameWithPath, bool srcIsReadable_, ScopeInfo *mom, 
	    suint begLn = UNDEF_LINE, suint endLn = UNDEF_LINE);
            // fileNameWithPath/mom must not be NULL
            // srcIsReadable == fopen(fileNameWithPath, "r") works 
  virtual ~FileScope(); 

  String Name() const { return name; } // fileNameWithPath from constructor 

  ProcScope* FindProc(const char* nm) const;
                                        
  virtual String BaseName() const      { return BaseFileName(name); }
  virtual String CodeName() const;
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const; 
  
  bool HasSourceFile() const { return srcIsReadable; } // srcIsReadable 
  
private: 
  void AddToProcMap(ProcScope& proc); 
  friend class ProcScope; 

  bool srcIsReadable; 
  String name; // the file name including the path 
  ProcScopeMap* procMap; 
};

// ---------------------------------------------------------
// ProcScopes are children of GroupScope's or FileScope's
// children: GroupScope's, LoopScope's, StmtRangeScope's
// ---------------------------------------------------------
class ProcScope: public CodeInfo {
public: 
  ProcScope(const char* name, CodeInfo *mom, const char* linkname,
	    suint begLn = UNDEF_LINE, suint endLn = UNDEF_LINE); 
  virtual ~ProcScope();
  
  virtual String Name() const       { return name; }
  virtual String LinkName() const   { return linkname; }
  virtual String CodeName() const; 
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const;

  void FixIntervalBoundaries();

private: 
  String name; 
  String linkname;
};

// ---------------------------------------------------------
// LoopScopes are children of GroupScope's, FileScope's, ProcScope's,
//   or LoopScope's.
// children: GroupScope's, LoopScope's, or StmtRangeScope's
// ---------------------------------------------------------
class LoopScope: public CodeInfo {
public: 
  LoopScope(CodeInfo *mom, int begLn, int endLn, int _id = -1);
  virtual ~LoopScope();
  
  virtual String CodeName() const;
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const; 
  
  bool FixIntervalBoundaries(suint pStartLn, suint pEndLn, 
           suint& newStart, suint& newEnd);
private:
  int id;
};

// ---------------------------------------------------------
// StmtRangeScopes are children of GroupScope's, FileScope's,
//   ProcScope's, or LoopScope's.
// children: none
// ---------------------------------------------------------
class StmtRangeScope: public CodeInfo {
public: 
  StmtRangeScope(CodeInfo *mom, int begLn, int endLn, int _id = -1); 
  virtual ~StmtRangeScope();
  
  virtual String CodeName() const;
  virtual String ToDumpString(int dmpFlag = PgmScopeTree::XML_TRUE) const;

private:
  int id;
};


#include "PgmScopeTreeIterator.h" 

/************************************************************************/
// testing 
/************************************************************************/
extern void ScopeInfoTester(int argc, const char** argv); 

#endif 
