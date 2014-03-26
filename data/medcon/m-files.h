/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-files.h                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-files.c header file                                    *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-files.h,v 1.33 2010/08/28 23:44:23 enlf Exp $
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

#ifndef __M_FILES_H__
#define __M_FILES_H__

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MdcSplitPath(x,y,z)    MdcMySplitPath(x,&y,&z)
#define MdcMergePath(x,y,z)    MdcMyMergePath(x,y,&z)
#define MdcCloseFile(fp)       {                                          \
                                 if (fp!=NULL  && fp!=stderr &&           \
                                     fp!=stdin && fp!=stdout )            \
                                   fclose(fp);                            \
                                 fp=NULL;                                 \
                               }

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

int MdcOpenFile(FILEINFO *fi, const char *path);
int MdcReadFile(FILEINFO *fi, int filenr, char *(*ReadFunc)());
int MdcWriteFile(FILEINFO *fi, int format, int prefixnr, char *(*WriteFunc)());
int MdcLoadFile(FILEINFO *fi);
int MdcSaveFile(FILEINFO *fi, int format, int prefixnr);
int MdcLoadPlane(FILEINFO *fi, Uint32 img);
int MdcDecompressFile(const char *path);
void MdcStringCopy(char *s1, char *s2, Uint32 length);
int MdcFileSize(FILE *fp);
int MdcFileExists(const char *fname);
int MdcKeepFile(const char *fname);
int MdcGetFrmt(FILEINFO *fi);
Uint8 *MdcGetImgBuffer(Uint32 bytes);
char *MdcHandleTruncated(FILEINFO *fi, Uint32 images, int remap);
int MdcWriteLine(IMG_DATA *id, Uint8 *buf, int type, FILE *fp);
int MdcWriteDoublePixel(double pix, int type, FILE *fp);
char *MdcGetFname(char path[]);
char *MdcGetLastPathDelim(char *path);
void MdcMySplitPath(char path[], char **dir, char **fname);
void MdcMyMergePath(char path[], char *dir, char **fname);
void MdcSetExt(char path[], char *ext);
void MdcNewExt(char dest[], char *src, char *ext);
void MdcPrefix(int n);
int MdcGetPrefixNr(FILEINFO *fi, int nummer);
void MdcNewName(char dest[], char *src, char *ext);
char *MdcAliasName(FILEINFO *fi, char alias[]);
void MdcEchoAliasName(FILEINFO *fi);
void MdcDefaultName(FILEINFO *fi, int format, char dest[], char *src);
void MdcRenameFile(char *name);
void MdcFillImgPos(FILEINFO *fi, Uint32 nr, Uint32 plane, float translation);
void MdcFillImgOrient(FILEINFO *fi, Uint32 nr);
int MdcGetOrthogonalInt(float f);
Int8 MdcGetPatSliceOrient(FILEINFO *fi, Uint32 i);
Int8 MdcTryPatSliceOrient(char *pat_orient);
Int8 MdcCheckQuantitation(FILEINFO *fi);
float MdcGetHeartRate(GATED_DATA *gd, Int16 type);
#endif

