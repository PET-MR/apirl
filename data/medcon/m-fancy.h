/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-fancy.h                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-fancy.c header file                                    *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-fancy.h,v 1.34 2010/08/28 23:44:23 enlf Exp $
 */

/*
   Copyright (C) 1997-2010 by Erik Nolf

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Place - Suite 330, Boston, MA 02111-1307, USA.  */

#ifndef __M_FANCY_H__
#define __M_FANCY_H__

/****************************************************************************
                              H E A D E R S
****************************************************************************/

#include "m-defs.h"
#include "m-structs.h"
#include "m-algori.h"
#include "m-global.h"
#include "m-error.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MDC_FULL_LENGTH  79
#define MDC_HALF_LENGTH  39


#define MDC_BOX_SIZE     16

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

void MdcPrintLine(char c, int length);
void MdcPrintChar(int c);
void MdcPrintStr(char *str);
void MdcPrintBoxLine(char c, int t);
void MdcPrintYesNo(int value );
void MdcPrintImageLayout(FILEINFO *fi, Uint32 gen, Uint32 img
                                     , Uint32 *abs, int repeat);
int MdcPrintValue(FILE *fp, Uint8 *pvalue, Uint16 type);
void MdcLowStr(char *str);
void MdcUpStr(char *str);
void MdcKillSpaces(char string[]);
void MdcRemoveAllSpaces(char string[]);
void MdcRemoveEnter(char string[]);
void MdcGetStrLine(char string[], int maxchars, FILE *fp);
void MdcGetStrInput(char string[], int maxchars);
int MdcGetSubStr(char *dest, char *src, int dmax, char sep, int n);
void MdcGetSafeString(char *dest, char *src, Uint32 length, Uint32 maximum);
int MdcUseDefault(const char string[]);
int MdcPutDefault(char string[]);
int MdcGetRange(const char *item, Uint32 *from, Uint32 *to, Uint32 *step);
char *MdcHandleEcatList(char *list, Uint32 **dims, Uint32 max);
char *MdcHandleNormList(char *list, Uint32 **inrs, Uint32 *it
                               , Uint32 *bt,Uint32 max);
char *MdcHandlePixelList(char *list, Uint32 **cols, Uint32 **rows
                               , Uint32 *it, Uint32 *bt);
char *MdcGetStrAcquisition(int acq_type);
char *MdcGetStrRawConv(int rawconv);
char *MdcGetStrEndian(int endian);
char *MdcGetStrCompression(int compression);
char *MdcGetStrPixelType(int type);
char *MdcGetStrColorMap(int map);
char *MdcGetStrYesNo(int boolean);
char *MdcGetStrSlProjection(int slice_projection);
char *MdcGetStrPatSlOrient(int patient_slice_orient);
char *MdcGetStrPatPos(int patient_slice_orient);
char *MdcGetStrPatOrient(int patient_slice_orient);
char *MdcGetStrSliceOrient(int patient_slice_orient);
char *MdcGetStrRotation(int rotation);
char *MdcGetStrMotion(int motion);
char *MdcGetStrModality(int modint);
char *MdcGetStrGSpectNesting(int nesting);
char *MdcGetStrHHMMSS(float msecs);
int MdcGetIntModality(char *modstr);
int MdcGetIntSliceOrient(int patient_slice_orient);
const char *MdcGetLibLongVersion(void);
const char *MdcGetLibShortVersion(void);
Uint32 MdcCheckStrSize(char *str_to_add, Uint32 current_size, Uint32 max);
int MdcMakeScanInfoStr(FILEINFO *fi);
int MdcIsDigit(char c);
void MdcWaitForEnter(int page);
Int32 MdcGetSelectionType(void);
void MdcFlushInput(void);
int MdcWhichDecompress(void);
int MdcWhichCompression(const char *fname);
void MdcAddCompressionExt(int ctype, char *fname);
#endif

