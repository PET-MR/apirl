/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-split.h                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-split.c header file                                    *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-split.h,v 1.12 2010/08/28 23:44:23 enlf Exp $
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

#pragma once

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MDC_SPLIT_NONE      MDC_NO  /* keep images in same volum   */
#define MDC_SPLIT_PER_SLICE 1       /* split over each image slice */
#define MDC_SPLIT_PER_FRAME 2       /* split over each time  frame */

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
Int16 MdcGetSplitAcqType(FILEINFO *fi);
Uint32 MdcGetNrSplit(void);
char *MdcGetSplitBaseName(char *path);
void MdcUpdateSplitPrefix(char *dpath, char *spath, char *bname, int nr);
char *MdcCopySlice(FILEINFO *ofi, FILEINFO *ifi, Uint32 slice0);
char *MdcCopyFrame(FILEINFO *ofi, FILEINFO *ifi, Uint32 frame0);
char *MdcSplitSlices(FILEINFO *fi, int format, int prefixnr);
char *MdcSplitFrames(FILEINFO *fi, int format, int prefixnr);
