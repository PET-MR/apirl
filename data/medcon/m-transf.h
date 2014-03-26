/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-transf.h                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-transf.c header file                                   *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-transf.h,v 1.17 2010/08/28 23:44:23 enlf Exp $
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

#ifndef __M_TRANSF_H__
#define __M_TRANSF_H__

/****************************************************************************
                                D E F I N E S 
 ****************************************************************************/

/* image transformations */
#define MDC_TRANSF_HORIZONTAL 1       /* flip horizontal    */
#define MDC_TRANSF_VERTICAL   2       /* flip vertical      */          
#define MDC_TRANSF_REVERSE    3       /* reverse sorting    */
#define MDC_TRANSF_CINE_APPLY 4       /* cine apply sorting */
#define MDC_TRANSF_CINE_UNDO  5       /* cine undo  sorting */
#define MDC_TRANSF_SQR1       6       /* make square        */
#define MDC_TRANSF_SQR2       7       /* make square pwr2   */
#define MDC_TRANSF_CROP       8       /* crop image dims    */

/* crop structure */
typedef struct Mdc_Crop_Info_t {

  Uint32 xoffset;
  Uint32 yoffset;
  Uint32 width;
  Uint32 height;

} MDC_CROP_INFO;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
int MdcFlipImgHorizontal(IMG_DATA *id);
int MdcFlipImgVertical(IMG_DATA *id);
char *MdcFlipHorizontal(FILEINFO *fi);
char *MdcFlipVertical(FILEINFO *fi);
char *MdcSortReverse(FILEINFO *fi);
char *MdcSortCineApply(FILEINFO *fi);
char *MdcSortCineUndo(FILEINFO *fi);
char *MdcMakeSquare(FILEINFO *fi, int SQR_TYPE);
char *MdcCropImages(FILEINFO *fi, MDC_CROP_INFO *ecrop);
char *MdcMakeGray(FILEINFO *fi);
char *MdcHandleColor(FILEINFO *fi);
char *MdcContrastRemap(FILEINFO *fi);
#endif

