/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-algori.h                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-algori.c header file                                   *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-algori.h,v 1.23 2010/08/28 23:44:23 enlf Exp $
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

#define  MdcPixels2Bytes(n) ((n+7)/8)                      /* bits -> bytes */

#define  MdcSWAP(x)  MdcSwapBytes( (Uint8 *)&x, sizeof(x)) /* HOST <> FILE  */

#define  MdcGRAY(r,g,b) ( ((int)(r)*11 + (int)(g)*16 + (int)(b)*5) >> 5)

#define  MdcFree(p)     { if (p!=NULL) free(p); p=NULL; }

#define  MdcMakeIEEEfl(a)  MdcVAXfl_to_IEEEfl(&a)
#define  MdcMakeVAXfl(a)   MdcIEEEfl_to_VAXfl(&a)

#define  MdcmCi2MBq(dose)  (dose * 37.)                    /* mCi -> MBq    */
#define  MdcMBq2mCi(dose)  (dose / 37.)                    /* MBq -> mCi    */

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
void *MdcRealloc(void *p, Uint32 bytes);
Uint32 MdcCeilPwr2(Uint32 x);
float MdcRotateAngle(float angle, float rotate);
int MdcDoSwap(void);
int MdcHostBig(void);
void MdcSwapBytes(Uint8 *ptr, int bytes);
void MdcForceSwap(Uint8 *ptr, int bytes);
void MdcIEEEfl_to_VAXfl(float *f);
void MdcVAXfl_to_IEEEfl(float *f);
int MdcFixFloat(float *ref);
int MdcFixDouble(double *ref);
int MdcType2Bytes(int type);
int MdcType2Bits(int type);
double MdcTypeIntMax(int type);
float MdcSingleImageDuration(FILEINFO *fi, Uint32 frame);
char *MdcImagesPixelFiddle(FILEINFO *fi);
double MdcGetDoublePixel(Uint8 *buf, int type);
void MdcPutDoublePixel(Uint8 *buf, double pix, int type);
int MdcDoSimpleCast(double minv, double maxv, double negmin, double posmax);
Uint8 *MdcGetResizedImage(FILEINFO *fi,Uint8 *buffer,int type,Uint32 img);
Uint8 *MdcGetDisplayImage(FILEINFO *fi, Uint32 img);
Uint8 *MdcMakeBIT8_U(Uint8 *cbuf, FILEINFO *fi, Uint32 img);
Uint8 *MdcGetImgBIT8_U(FILEINFO *fi, Uint32 img);
Uint8 *MdcMakeBIT16_S(Uint8 *cbuf, FILEINFO *fi, Uint32 img);
Uint8 *MdcGetImgBIT16_S(FILEINFO *fi, Uint32 img);
Uint8 *MdcMakeBIT32_S(Uint8 *cbuf, FILEINFO *fi, Uint32 img);
Uint8 *MdcGetImgBIT32_S(FILEINFO *fi, Uint32 img);
Uint8 *MdcMakeFLT32(Uint8 *cbuf, FILEINFO *fi, Uint32 img);
Uint8 *MdcGetImgFLT32(FILEINFO *fi, Uint32 img);
Uint8 *MdcMakeImgSwapped(Uint8 *cbuf, FILEINFO *fi, Uint32 img,
                         Uint32 width, Uint32 height, int type);
Uint8 *MdcGetImgSwapped(FILEINFO *fi, Uint32 img);
int MdcUnpackBIT12(FILEINFO *fi, Uint32 img);
Uint32 MdcHashDJB2(unsigned char *str);
Uint32 MdcHashSDBM(unsigned char *str);
