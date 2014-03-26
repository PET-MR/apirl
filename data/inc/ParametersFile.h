#ifndef _PARAMETERS_FILE_H
#define	_PARAMETERS_FILE_H


// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32
	#else
		#ifdef HAVE_GCCVISIBILITYPATCH   // GCC 4.x and patched GCC 3.4 under Linux
			#define DLLEXPORT __attribute__ ((visibility("default")))
			#define DLLLOCAL __attribute__ ((visibility("hidden")))
		#else
			#define DLLEXPORT
			#define DLLLOCAL
		#endif
	#endif
#else                         // static linking
	#define DLLEXPORT
	#define DLLLOCAL
#endif
/*
#ifdef __cplusplus
	extern "C" 
#endif
*/

// Códigos de Error:
#define PMF_NO_ERROR	0
#define PMF_KEY_REPEATED	1
#define PMF_KEY_NOT_FOUND	2
#define PMF_FILE_CANT_OPEN	3
#define PMF_WRONG_FILE_FORMAT	4
#define PMF_METHOD_NOT_FOUND	5

// Tamaño de keys
#define MAX_KEY_LENGTH 256
#define MAX_VALUE_LENGTH 256

DLLEXPORT int parametersFile_read(char* fileName, char* parameterType, char* searchWord, char* returnValue, char* errorMessage);
DLLEXPORT int parametersFile_readMultipleKeys(char* fileName, char* parameterType, char** searchWords, int numWords, char** returnValue, char* errorMessage);

#endif
