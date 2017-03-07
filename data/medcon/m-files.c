/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-files.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : Files edit, file I/O, image buffers                      *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcOpenFile()         - Open file                        *
 *                MdcReadFile()         - Read file                        *
 *                MdcWriteFile()        - Write file                       * 
 *                MdcLoadFile()         - Load file                        *
 *                MdcSaveFile()         - Save file                        *
 *                MdcLoadPlane()        - Load one plane                   *
 *                MdcSavePlane()        - Save one plane                   *
 *                MdcDecompressFile()   - Decompress file                  *
 *                MdcStringCopy()       - Copy a string                    *
 *                MdcFileSize()         - Get the size of a file           *
 *                MdcFileExists()       - Check if file exists             *
 *                MdcKeepFile()         - Prevent overwrite existing file  *
 *                MdcGetFrmt()          - Get format of imagefile          *
 *                MdcGetImgBuffer()     - Malloc a buffer                  *
 *                MdcHandleTruncated()  - Reset FILEINFO for truncated file*
 *                MdcWriteLine()        - Write image line                 *
 *                MdcWriteDoublePixel() - Write single pixel double input  *
 *                MdcGetFname()         - Get filename                     *
 *                MdcSetExt()           - Set filename extension           *
 *                MdcNewExt()           - Create filename extension        *
 *                MdcPrefix()           - Create filename prefix           * 
 *                MdcGetPrefixNr()      - Get proper filename prefix       *
 *                MdcGetLastPathDelim() - Get pointer last path delimiter  * 
 *                MdcMySplitPath()      - Split path from filename         *
 *                MdcMyMergePath()      - Merge path to filename           *
 *                MdcNewName()          - Create new filename              *
 *                MdcAliasName()        - Create alias name based on ID's  *
 *                MdcEchoAliasName()    - Echo   alias name based on ID's  *
 *                MdcDefaultName()      - Create new filename for format   *
 *                MdcRenameFile()       - Let user give a new name         *
 *                MdcFillImgPos()       - Fill the image_pos(_dev/_pat)    *
 *                MdcFillImgOrient()    - Fill the image_orient(_dev/_pat) *
 *                MdcGetOrthogonalInt() - Get orthogonal direction cosine  *
 *                MdcGetPatSliceOrient()- Get patient_slice_orient         *
 *                MdcTryPatSliceOrient()- Try to get it from pat_orient    *
 *                MdcCheckQuantitation()- Check quantitation preservation  *
 *                MdcGetHeartRate()     - Get heart rate from gated data   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-files.c,v 1.113 2010/08/28 23:44:23 enlf Exp $ 
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
 
/****************************************************************************
                              H E A D E R S
****************************************************************************/

#include "m-depend.h"

#include <stdio.h>
#include <ctype.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#ifndef _WIN32
#include <strings.h>
#endif
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "medcon.h"

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
int MdcOpenFile(FILEINFO *fi, const char *path)
{
  int ctype;

  if (MDC_FILE_STDIN == MDC_NO) {
    if ( (ctype = MdcWhichCompression(path)) != MDC_NO ) {
      if ((MdcDecompressFile(path)) != MDC_OK) {

        MdcPrntWarn("Decompression failed");

        /* no longer quit, because NIFTI */
        /* can read .gz files directly   */
        /* return(MDC_BAD_OPEN); */
       
        /* prevent unwanted unlink later */ 
        ctype = MDC_NO;
      }
    }
  }else{
    ctype = MDC_NO;
  }

  MdcInitFI(fi, path);  fi->compression = (Int8)ctype;

  if (MDC_FILE_STDIN == MDC_NO) {
    if ( (fi->ifp=fopen(fi->ipath,"rb")) == NULL) {
      MdcPrntWarn("Couldn't open <%s> for reading",fi->ipath);
      return(MDC_BAD_OPEN);
    }
  }else{
    fi->ifp = stdin; strcpy(fi->ipath,"stdin");
  }

  if (ctype != MDC_NO) unlink(path);

  MdcSplitPath(fi->ipath,fi->idir,fi->ifname);
 
  return(MDC_OK);
  
}

int MdcReadFile(FILEINFO *fi, int filenr, char *(*ReadFunc)(FILEINFO *fi))
{
  int FORMAT=MDC_FRMT_NONE;
  const char *msg=NULL;

  if (ReadFunc == NULL) {
    /* get the format or fallback */
    if ( (FORMAT=MdcGetFrmt(fi) ) == MDC_FRMT_NONE ) {
      MdcCloseFile(fi->ifp);
      MdcPrntWarn("Unsupported format in <%s>",fi->ifname);
      return(MDC_BAD_CODE);
    }else if (FORMAT < 0) {
      MdcCloseFile(fi->ifp);
      MdcPrntWarn("Unsuccessful read from <%s>",fi->ifname);
      return(MDC_BAD_READ);
    }
  }

  /* print a header message */
  if (MDC_INFO && !MDC_INTERACTIVE) {
    MdcPrntScrn("\n");
    MdcPrintLine('*',MDC_FULL_LENGTH);
    MdcPrntScrn("FILE %03d : %s\t\t\t",filenr,fi->ifname);
    MdcPrntScrn("FORMAT: %s\n",FrmtString[fi->iformat]);
    MdcPrintLine('*',MDC_FULL_LENGTH);
    MdcPrntScrn("\n");
  }

  /* read the appropriate format */
  switch (FORMAT) {

    case MDC_FRMT_RAW:  msg=MdcReadRAW(fi);  break;
#if MDC_INCLUDE_ACR
    case MDC_FRMT_ACR:  msg=MdcReadACR(fi);  break;
#endif
#if MDC_INCLUDE_GIF
    case MDC_FRMT_GIF:  msg=MdcReadGIF(fi);  break;
#endif
#if MDC_INCLUDE_INW
    case MDC_FRMT_INW:  msg=MdcReadINW(fi);  break;
#endif
#if MDC_INCLUDE_ECAT
    case MDC_FRMT_ECAT6: msg=MdcReadECAT6(fi); break;

    case MDC_FRMT_ECAT7: msg=MdcReadECAT7(fi); break;
#endif
#if MDC_INCLUDE_INTF 
    case MDC_FRMT_INTF: msg=MdcReadINTF(fi); break;
#endif
#if MDC_INCLUDE_ANLZ
    case MDC_FRMT_ANLZ: msg=MdcReadANLZ(fi); break;
#endif
#if MDC_INCLUDE_DICM
    case MDC_FRMT_DICM: msg=MdcReadDICM(fi); break;
#endif
#if MDC_INCLUDE_PNG
    case MDC_FRMT_PNG:  msg=MdcReadPNG(fi);  break;
#endif
#if MDC_INCLUDE_CONC
    case MDC_FRMT_CONC: msg=MdcReadCONC(fi); break;
#endif
#if MDC_INCLUDE_NIFTI
    case MDC_FRMT_NIFTI: msg=MdcReadNIFTI(fi); break;
#endif
    default:
      if (ReadFunc != NULL) {
        msg = ReadFunc(fi);
      }else{
        MdcPrntWarn("Reading: Unsupported format");
        return(MDC_BAD_FILE);
      }
  }

  /* read error handling */
  if (msg != NULL) {
    MdcPrntWarn("Reading: %s",msg);
    if (strstr(msg,"Truncated image") == NULL) {
      MdcCleanUpFI(fi);
      return(MDC_BAD_READ);
    }else{
      MdcCloseFile(fi->ifp);
    }
  }

  /* database info | dangerous: not all image info read */
  if (MDC_INFO_DB == MDC_YES) return(MDC_OK);

  /* echo alias, quickly leave */
  if (MDC_ECHO_ALIAS == MDC_YES) return(MDC_OK);

  /* set the proper color map */
  if (fi->map == MDC_MAP_GRAY) {
    /* gray scale images, set selected colormap  */
    if (MDC_COLOR_MAP < MDC_MAP_GRAY) MDC_COLOR_MAP = MDC_MAP_GRAY; 
    fi->map = MDC_COLOR_MAP;
  }else{
    /* colored images, preserve map */
    fi->map = (Uint8)MdcSetPresentMap(fi->palette);
  }

  /* get the proper color map */
  MdcGetColorMap((int)fi->map,fi->palette);

  /* the obligated pixel handling */
  msg = MdcImagesPixelFiddle(fi);
  if (msg != NULL) {
    MdcCleanUpFI(fi);
    MdcPrntWarn("Reading: %s",msg);
    return(MDC_BAD_CODE);
  }

  /* do some requested transformations */
  msg = NULL;
  if (MDC_INFO == MDC_NO) {
    if ((msg == NULL) && (MDC_CONTRAST_REMAP  == MDC_YES))
      msg=MdcContrastRemap(fi);
    if ((msg == NULL) && (MDC_MAKE_SQUARE     != MDC_NO))
      msg=MdcMakeSquare(fi,MDC_MAKE_SQUARE);
    if ((msg == NULL) && (MDC_FLIP_HORIZONTAL == MDC_YES))
      msg=MdcFlipHorizontal(fi);
    if ((msg == NULL) && (MDC_FLIP_VERTICAL   == MDC_YES))
      msg=MdcFlipVertical(fi);
    if ((msg == NULL) && (MDC_SORT_REVERSE    == MDC_YES))
      msg=MdcSortReverse(fi);
    if ((msg == NULL) && (MDC_SORT_CINE_APPLY == MDC_YES))
      msg=MdcSortCineApply(fi);
    if ((msg == NULL) && (MDC_SORT_CINE_UNDO  == MDC_YES))
      msg=MdcSortCineUndo(fi);
    if ((msg == NULL) && (MDC_CROP_IMAGES     == MDC_YES))
      msg=MdcCropImages(fi,NULL);

    if (msg != NULL) {
      MdcCleanUpFI(fi);
      MdcPrntWarn("Transform: %s",msg);
      return(MDC_BAD_CODE);
    }
  }

  return(MDC_OK);
 
}

int MdcWriteFile(FILEINFO *fi, int format, int prefixnr, char *(*WriteFunc)())
{
  const char *msg=NULL;
  Int8 INTERNAL_ENDIAN;

  if (WriteFunc != NULL) format = MDC_FRMT_NONE;

  /* reset ID's rescaled stuff from any previous write */
  MdcResetIDs(fi);

  /* negative value = self made prefix */
  if (prefixnr >= 0 ) MdcPrefix(prefixnr);

  /* preserve internal file endian - global var issue */
  INTERNAL_ENDIAN = MDC_FILE_ENDIAN;

  switch (format) {
    case MDC_FRMT_RAW  : fi->rawconv = MDC_FRMT_RAW; 
                         msg=MdcWriteRAW(fi);
                         break;
    case MDC_FRMT_ASCII: fi->rawconv = MDC_FRMT_ASCII;
                         msg=MdcWriteRAW(fi);      
                         break;
#if MDC_INCLUDE_ACR
    case MDC_FRMT_ACR  : msg=MdcWriteACR(fi);  break; 
#endif
#if MDC_INCLUDE_GIF
    case MDC_FRMT_GIF  : msg=MdcWriteGIF(fi);  break;
#endif
#if MDC_INCLUDE_INW
    case MDC_FRMT_INW  : msg=MdcWriteINW(fi);  break;
#endif
#if MDC_INCLUDE_ECAT
    case MDC_FRMT_ECAT6: msg=MdcWriteECAT6(fi); break;
  #if MDC_INCLUDE_TPC
    case MDC_FRMT_ECAT7: msg=MdcWriteECAT7(fi); break;
  #endif
#endif
#if MDC_INCLUDE_INTF
    case MDC_FRMT_INTF : msg=MdcWriteINTF(fi); break;
#endif
#if MDC_INCLUDE_ANLZ
    case MDC_FRMT_ANLZ : msg=MdcWriteANLZ(fi); break;
#endif
#if MDC_INCLUDE_DICM
    case MDC_FRMT_DICM : msg=MdcWriteDICM(fi); break;
#endif
#if MDC_INCLUDE_PNG
    case MDC_FRMT_PNG  : msg=MdcWritePNG(fi);  break;
#endif
#if MDC_INCLUDE_CONC
    case MDC_FRMT_CONC : msg=MdcWriteCONC(fi); break;
#endif
#if MDC_INCLUDE_NIFTI
    case MDC_FRMT_NIFTI: msg=MdcWriteNIFTI(fi); break;
#endif
    default:
      if (WriteFunc != NULL) {
        msg = WriteFunc(fi);
      }else{
        MdcPrntWarn("Writing: Unsupported format");
        return(MDC_BAD_FILE);
      }
  } 

 /* restore internal file endian - global var issue */
  MDC_FILE_ENDIAN = INTERNAL_ENDIAN;

  MdcCloseFile(fi->ofp);

  if (msg != NULL) {
    MdcPrntWarn("Writing: %s",msg);
    return(MDC_BAD_WRITE);
  }

  return(MDC_OK);
 
}

int MdcLoadFile(FILEINFO *fi)
{
  int FORMAT=MDC_FRMT_NONE;
  const char *msg=NULL;

  /* get the format or fallback */
  if ( (FORMAT=MdcGetFrmt(fi) ) == MDC_FRMT_NONE ) {
    MdcCloseFile(fi->ifp);
    return(MDC_BAD_READ);
  }

  /* read the appropriate format */
  switch (FORMAT) {

    case MDC_FRMT_RAW:  msg=MdcReadRAW(fi);  break;
#if MDC_INCLUDE_ACR
    case MDC_FRMT_ACR:  msg=MdcReadACR(fi);  break;
#endif
#if MDC_INCLUDE_GIF
    case MDC_FRMT_GIF:  msg=MdcReadGIF(fi);  break;
#endif
#if MDC_INCLUDE_INW
    case MDC_FRMT_INW:  msg=MdcReadINW(fi);  break;
#endif
#if MDC_INCLUDE_ECAT
    case MDC_FRMT_ECAT6: msg=MdcReadECAT6(fi); break;

    case MDC_FRMT_ECAT7: msg=MdcReadECAT7(fi); break;
#endif
#if MDC_INCLUDE_INTF 
    case MDC_FRMT_INTF: msg=MdcReadINTF(fi); break;
#endif
#if MDC_INCLUDE_ANLZ
    case MDC_FRMT_ANLZ: msg=MdcReadANLZ(fi); break;
#endif
#if MDC_INCLUDE_DICM
    case MDC_FRMT_DICM: msg=MdcReadDICM(fi); break;
#endif
#if MDC_INCLUDE_PNG
    case MDC_FRMT_PNG:  msg=MdcReadPNG(fi);  break;
#endif
#if MDC_INCLUDE_CONC
    case MDC_FRMT_CONC: msg=MdcLoadCONC(fi); break;
#endif
#if MDC_INCLUDE_NIFTI
    case MDC_FRMT_NIFTI: msg=MdcReadNIFTI(fi); break;
#endif
    default:
      MdcPrntWarn("Loading: unsupported format");
      return(MDC_BAD_FILE);
  }

  /* read error handling */
  if (msg != NULL) {
    MdcPrntWarn("Loading: %s",msg);
    return(MDC_BAD_READ);
  }

  return(MDC_OK);
}

int MdcSaveFile(FILEINFO *fi, int format, int prefixnr)
{
  const char *msg=NULL;
  Int8 INTERNAL_ENDIAN;

  /* reset ID's rescaled stuff from any previous write */
  MdcResetIDs(fi);

  /* negative value = self made prefix */
  if (prefixnr >= 0 ) MdcPrefix(prefixnr);

  /* preserve internal file endian - global var issue */
  INTERNAL_ENDIAN = MDC_FILE_ENDIAN;

  switch (format) {
    case MDC_FRMT_RAW  : fi->rawconv = MDC_FRMT_RAW; 
                         msg=MdcWriteRAW(fi);
                         break;
    case MDC_FRMT_ASCII: fi->rawconv = MDC_FRMT_ASCII;
                         msg=MdcWriteRAW(fi);      
                         break;
#if MDC_INCLUDE_ACR
    case MDC_FRMT_ACR  : msg=MdcWriteACR(fi);  break; 
#endif
#if MDC_INCLUDE_GIF
    case MDC_FRMT_GIF  : msg=MdcWriteGIF(fi);  break;
#endif
#if MDC_INCLUDE_INW
    case MDC_FRMT_INW  : msg=MdcWriteINW(fi);  break;
#endif
#if MDC_INCLUDE_ECAT
    case MDC_FRMT_ECAT6: msg=MdcWriteECAT6(fi); break;
  #if MDC_INCLUDE_TPC
    case MDC_FRMT_ECAT7: msg=MdcWriteECAT7(fi); break;
  #endif
#endif
#if MDC_INCLUDE_INTF
    case MDC_FRMT_INTF : msg=MdcWriteINTF(fi); break;
#endif
#if MDC_INCLUDE_ANLZ
    case MDC_FRMT_ANLZ : msg=MdcWriteANLZ(fi); break;
#endif
#if MDC_INCLUDE_DICM
    case MDC_FRMT_DICM : msg=MdcWriteDICM(fi); break;
#endif
#if MDC_INCLUDE_PNG
    case MDC_FRMT_PNG  : msg=MdcWritePNG(fi);  break;
#endif
#if MDC_INCLUDE_CONC
    case MDC_FRMT_CONC : msg=MdcSaveCONC(fi); break;
#endif
#if MDC_INCLUDE_NIFTI
    case MDC_FRMT_NIFTI: msg=MdcWriteNIFTI(fi); break;
#endif
    default:
      MdcPrntWarn("Writing: Unsupported format");
      return(MDC_BAD_FILE);

  } 

 /* restore internal file endian - global var issue */
  MDC_FILE_ENDIAN = INTERNAL_ENDIAN;

  MdcCloseFile(fi->ofp);

  if (msg != NULL) {
    MdcPrntWarn("Saving: %s",msg);
    return(MDC_BAD_WRITE);
  }

  return(MDC_OK);
}

int MdcLoadPlane(FILEINFO *fi, Uint32 img)
{
  const char *msg=NULL;

  /* sanity check */
  if (img >= fi->number) {
    MdcPrntWarn("Loading plane %d: non-existent",img);
    return(MDC_BAD_CODE);
  }

  /* check the format */
  if (fi->iformat == MDC_FRMT_NONE) {
    MdcPrntWarn("Loading plane %d: unsupported format",img);
    return(MDC_BAD_CODE); 
  }

  /* check for loaded planes */
  if (fi->image[img].buf != NULL) {
    MdcPrntWarn("Loading plane %d: already loaded",img);
    return(MDC_OK);
  }

  /* read appropriate format */
  switch (fi->iformat) {
    case MDC_FRMT_RAW:  /* msg=MdcLoadPlaneRAW(fi, img); */  break;
#if MDC_INCLUDE_ACR
    case MDC_FRMT_ACR:  /* msg=MdcLoadPlaneACR(fi, img); */  break;
#endif
#if MDC_INCLUDE_GIF
    case MDC_FRMT_GIF:  /* msg=MdcLoadPlaneGIF(fi, img); */  break;
#endif
#if MDC_INCLUDE_INW
    case MDC_FRMT_INW:  /* msg=MdcLoadPlaneINW(fi, img); */  break;
#endif
#if MDC_INCLUDE_ECAT
    case MDC_FRMT_ECAT6: /* msg=MdcLoadPlaneECAT6(fi, img); */ break;

    case MDC_FRMT_ECAT7: /* msg=MdcLoadPlaneECAT7(fi, img); */ break;
#endif
#if MDC_INCLUDE_INTF 
    case MDC_FRMT_INTF: /* msg=MdcLoadPlaneINTF(fi, img); */ break;
#endif
#if MDC_INCLUDE_ANLZ
    case MDC_FRMT_ANLZ: /* msg=MdcLoadPlaneANLZ(fi, img); */ break;
#endif
#if MDC_INCLUDE_DICM
    case MDC_FRMT_DICM: /* msg=MdcLoadPlaneDICM(fi, img); */ break;
#endif
#if MDC_INCLUDE_PNG
    case MDC_FRMT_PNG:  /* msg=MdcLoadPlanePNG(fi, img); */ break;
#endif
#if MDC_INCLUDE_CONC
    case MDC_FRMT_CONC: msg=MdcLoadPlaneCONC(fi, (signed)img); break;
#endif
#if MDC_INCLUDE_NIFTI
    case MDC_FRMT_NIFTI: /* msg=MdcLoadPlaneNIFTI(fi, img); */ break;
#endif
    default:
      MdcPrntWarn("Loading plane %d: unsupported format",img);
      return(MDC_BAD_FILE);
  }

  /* error handling */
  if (msg != NULL) {
    MdcPrntWarn("Loading plane %d: %s",img,msg);
    return(MDC_BAD_READ);
  }

  return(MDC_OK);
}

int MdcDecompressFile(const char *path)
{
  char *ext;

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Decompress (Waiting)");

  if (MDC_VERBOSE) MdcPrntMesg("Decompression ...");

  /* get last extension (.gz or .Z) */
  ext = strrchr(path,'.');
  /* build system call, put paths between quotes     */
  /* in order to catch at least some weird filenames */
  sprintf(mdcbufr,"%s -c \"%s\" > \"",MDC_DECOMPRESS,path);
  /* remove extension from filename */
  *ext = '\0';
  /* add to pipe of system call */
  strcat(mdcbufr,path); strcat(mdcbufr,"\"");

  /* check if decompressed file already exists */
  if (MdcKeepFile(path)) {
    MdcPrntWarn("Decompressed filename exists!!");

    if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_END,0.,NULL);

    /* no overwrite, restore orig path */
    *ext = '.';

    return(MDC_BAD_CODE);
  }

  if (system(mdcbufr)) {

    if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_END,0.,NULL);

    unlink(path);

    /* no gunzip? restore orig path */
    *ext = '.';
    
    return(MDC_BAD_CODE);
  }

  return(MDC_OK);

}

void MdcStringCopy(char *s1, char *s2, Uint32 length)
{
  if ( length < MDC_MAXSTR) {
    memcpy(s1,s2,length);
    s1[length] = '\0';
  }else{
    memcpy(s1,s2,MDC_MAXSTR);
    s1[MDC_MAXSTR-1] = '\0';
  }
}
 

int MdcFileSize(FILE *fp)
{
  int size;

  fseek(fp,0,SEEK_END);

  size=ftell(fp);

  fseek(fp,0,SEEK_SET);

  return(size);

}

int MdcFileExists(const char *fname)
{

  FILE *fp;

  if ((fp=fopen(fname,"rb")) == NULL) return MDC_NO;

  MdcCloseFile(fp);

  return MDC_YES;

}

int MdcKeepFile(const char *fname)
{
  if (MDC_FILE_OVERWRITE == MDC_YES) return(MDC_NO);

  return(MdcFileExists(fname));
}

int MdcGetFrmt(FILEINFO *fi) 
{
  int i, format=MDC_FRMT_NONE;

  if (MDC_FILE_STDIN == MDC_YES && MDC_FRMT_INPUT != MDC_FRMT_NONE) {
    fi->iformat = MDC_FRMT_INPUT; return(MDC_FRMT_INPUT); 
  }

  if (MDC_INTERACTIVE) { fi->iformat = MDC_FRMT_RAW; return(MDC_FRMT_RAW); }

  for (i=MDC_MAX_FRMTS-1;i>=3; i--) {

    /* MARK: checks reversed; NIFTI must come before ANLZ */
    /* otherwise MdcReadANLZ() would handle NIFTI files   */
   
    switch (i) {

#if MDC_INCLUDE_ACR
     case MDC_FRMT_ACR:   format = MdcCheckACR(fi);  break;
#endif
#if MDC_INCLUDE_GIF
     case MDC_FRMT_GIF:   format = MdcCheckGIF(fi);  break;
#endif
#if MDC_INCLUDE_INW
     case MDC_FRMT_INW:   format = MdcCheckINW(fi);  break; 
#endif
#if MDC_INCLUDE_INTF
     case MDC_FRMT_INTF:  format = MdcCheckINTF(fi); break;
#endif
#if MDC_INCLUDE_ANLZ
     case MDC_FRMT_ANLZ:  format = MdcCheckANLZ(fi); break;
#endif
#if MDC_INCLUDE_ECAT
     case MDC_FRMT_ECAT6: format = MdcCheckECAT6(fi); break;

     case MDC_FRMT_ECAT7: format = MdcCheckECAT7(fi); break;
#endif
#if MDC_INCLUDE_DICM
     case MDC_FRMT_DICM:  format = MdcCheckDICM(fi); break;
#endif
#if MDC_INCLUDE_PNG
     case MDC_FRMT_PNG:   format = MdcCheckPNG(fi);  break;
#endif
#if MDC_INCLUDE_CONC
     case MDC_FRMT_CONC:  format = MdcCheckCONC(fi); break;
#endif
#if MDC_INCLUDE_NIFTI
     case MDC_FRMT_NIFTI: format = MdcCheckNIFTI(fi); break;
#endif
    }

    fseek(fi->ifp,0,SEEK_SET); 

    if ( format != MDC_FRMT_NONE ) break;

  }

  if (format == MDC_FRMT_NONE) {
    if (MDC_FALLBACK_FRMT != MDC_FRMT_NONE) {
      MdcPrntWarn("Image format unknown - trying fallback format");
      format = MDC_FALLBACK_FRMT;
    }
  }

  fi->iformat = format;

  return(format);

}


Uint8 *MdcGetImgBuffer(Uint32 bytes)
{
  return((Uint8 *)calloc(1,bytes));
}

char *MdcHandleTruncated(FILEINFO *fi, Uint32 images, int remap)
{
   Uint32 i;
  
   if (images == 0) images = 1;
   if ((remap == MDC_YES) && (images < fi->number)) {
     if (!MdcGetStructID(fi,images)) {
       return("Couldn't realloc truncated IMG_DATA structs");
     }
   }

   fi->truncated = MDC_YES;
   fi->dim[0] = 3;
   fi->dim[3] = fi->number;
   for (i=4; i<MDC_MAX_DIMS; i++) fi->dim[i] = 0;

   return NULL;
}

/* put whole line but MdcSWAP if necessary */
/* on success 1 on error 0 */
int MdcWriteLine(IMG_DATA *id, Uint8 *buf, int type, FILE *fp)
{
  Uint32 i, bytes = MdcType2Bytes(type);
  Uint8 *pbuf;

  if (bytes == 1) {

    fwrite(buf,id->width,bytes,fp); /* no MdcSWAP necessary */

  }else for (i=0; i<id->width; i++) {

   pbuf = buf + (i * bytes);

   switch (type) {
    case BIT16_S:
     {
        Int16 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case BIT16_U:
     { 
        Uint16 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case BIT32_S:
     {
        Int32 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case BIT32_U:
     {
        Uint32 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
#ifdef HAVE_8BYTE_INT
    case BIT64_S:
     {
        Int64 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case BIT64_U:
     {
        Uint64 pix;

        memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
        fwrite((char *)&pix,1,bytes,fp);
     }
     break;
#endif
    case FLT32:
     {
       float pix;

       memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
       fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case FLT64:
     {
       double pix;

       memcpy(&pix,pbuf,bytes); MdcSWAP(pix);
       fwrite((char *)&pix,1,bytes,fp);
     }
     break;
    case VAXFL32:
     {
       float  flt;

       memcpy(&flt,pbuf,bytes); 
       MdcMakeVAXfl(flt); 
       fwrite((char *)&flt,1,bytes,fp);

     }
     break;

   }
 }
 
 if (ferror(fp)) return MDC_NO;

 return MDC_YES;

} 

/* Put pixel but MdcSWAP if necessary */
/* on success 1  on error 0 */
int MdcWriteDoublePixel(double pix, int type, FILE *fp)
{
 unsigned int bytes = (unsigned)MdcType2Bytes(type);

 switch (type) {
  case BIT8_S:
   {
     Int8 c = (Int8)pix;
     fwrite((char *)&c,1,bytes,fp);
   }
   break; 
  case BIT8_U:
   {
     Uint8 c = (Uint8)pix;
     fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case BIT16_S:
   {
     Int16 c = (Int16)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case BIT16_U:
   {
     Uint16 c = (Uint16)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case BIT32_S:
   {
     Int32 c = (Int32)pix; 
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case BIT32_U:
   {
     Uint32 c = (Uint32)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
#ifdef HAVE_8BYTE_INT
  case BIT64_S:
   {
     Int64 c = (Int64)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case BIT64_U:
   {
     Uint64 c = (Uint64)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
#endif 
  case FLT32:
   {
     float c = (float)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
  case VAXFL32:
   {
     float flt = (float)pix;

     MdcMakeVAXfl(flt);
     fwrite((char *)&flt,1,bytes,fp);

   }
   break; 
  case FLT64:
   {
     double c = (double)pix;
     MdcSWAP(c); fwrite((char *)&c,1,bytes,fp);
   }
   break;
 }

 if (ferror(fp)) return MDC_NO;
 
 return MDC_YES;

}

char *MdcGetFname(char path[])
{
  char *p;

  p = MdcGetLastPathDelim(path);

  if ( p == NULL ) return(path);

  return(p+1); 
}

void MdcSetExt(char path[], char *ext)
{
  //char *p;

  if (path == NULL) return;

  if (ext  == NULL) return;  

  // Modificado Martín Belzunce 03/07/13
  // Esto lo comento, porque el nombre del archivo supongo que no traía ninguna extensión:
  // Lo incluyo en new ext, así se puede usar setext cuando no se quiere reemplazar la extensión
  // por una nueva.
  //p=(char *)strrchr(path,'.');

  //if ( p != NULL )  *p = '\0';

  strcat(path,".");
  strcat(path,ext);
}

void MdcNewExt(char dest[], char *src, char *ext)
{
  char *p, *s;

  if (mdcbasename != NULL) {
    /* forced output name/path */
    s = MdcGetLastPathDelim(mdcbasename);
    p = strrchr(mdcbasename,'.');
    if (s != NULL) {
      /* full pathname */
      strncpy(dest,mdcbasename,MDC_MAX_PATH); dest[MDC_MAX_PATH-5]='\0';
      if ((p != NULL) && (p < s)) {
        /* prevent "(.)./filename" without . for extension */
        strcat(dest,".ext");
      }
    }else{
      /* single basename */
      strncpy(dest,mdcbasename,MDC_MAX_PATH);
    }
  }else{
    /* default output name */
    if ((src != NULL) && (src[0] != '\0'))  strcat(dest,src);
  }
  // Agregao Martín Belzunce 03/07/13. Así NewExt reemplaza la anterior y setext solo
  // la agrega:
  p=(char *)strrchr(dest,'.');

  if ( p != NULL )  *p = '\0';
  MdcSetExt(dest,ext);
} 

/* create a prefix of the form: 000 ...  ... 999,A00................ZZZ
                                  <- normal ->  |  <----- extra ----->
                                     (1000)              (33696)

normal prefixes =
     integers from 000 to 999        (1000)

extra  prefixes =                        +
     1st char: A...Z                   (26)
     2nd char: 0...9,A...Z             (36)
     3rd char: 0...9,A...Z             (36)

This construct gives at first numeric prefixes, extended with
a lot more alphanumeric prefixes before overlap. A directory
listing will thus show the filenames in sequence of creation. */
        
void MdcPrefix(int n)
{
  int t, c1, c2, c3, v1, v2, v3;
  int A='A', Zero='0';  /* ascii values for A and Zero */
  char cprefix[6];
  if (MDC_PREFIX_DISABLED == MDC_YES) {
    strcpy(prefix,""); return;
  }
    
  if (n < 1000) {
    sprintf(cprefix,"m%03d-",n);
  }else{
     t = n - 1000;
    v1 = t / 1296;
    v2 = (t % 1296) / 36;
    v3 = (t % 1296) % 36;


    if (n >= 34696) {
      MdcPrntWarn("%d-th conversion creates overlapping filenames", n);
      if (MDC_FILE_OVERWRITE == MDC_NO) return;
    }

    /* first  char */
    c1 = A + v1;                   /* A...Z */
    /* second char */
    if (v2 < 10) c2 = Zero + v2;   /* 0...9 */
    else c2 = A + v2 - 10;         /* A...Z */
    /* third  char */
    if (v3 < 10) c3 = Zero + v3;   /* 0...9 */
    else c3 = A + v3 - 10;         /* A...Z */

    sprintf(cprefix,"m%c%c%c-",(char)c1,(char)c2,(char)c3);
  }

  if (MDC_FILE_SPLIT != MDC_NO) {
    /* special naming for splitted files */
    switch (MDC_FILE_SPLIT) {
      case MDC_SPLIT_PER_FRAME:
          sprintf(prefix,"%sf%04u-",cprefix,MdcGetNrSplit() + 1);
          break;
      case MDC_SPLIT_PER_SLICE:
          sprintf(prefix,"%ss%04d-",cprefix,MdcGetNrSplit() + 1);
          break;
    }
  }else if (MDC_FILE_STACK != MDC_NO) {
    /* special naming for stacked files */
    switch (MDC_FILE_STACK) {
      case MDC_STACK_SLICES:
          sprintf(prefix,"%sstacks-",cprefix);
          break;
      case MDC_STACK_FRAMES:
          sprintf(prefix,"%sstackf-",cprefix);
          break;
    }
  }else{
    /* default naming */
          strcpy(prefix,cprefix);
  }
}

int MdcGetPrefixNr(FILEINFO *fi, int nummer)
{
  int prefixnr;

  prefixnr = MDC_PREFIX_ACQ == MDC_YES ? fi->nr_acquisition :
            (MDC_PREFIX_SER == MDC_YES ? fi->nr_series : nummer) ;

  return(prefixnr);
}

void MdcNewName(char dest[], char *src, char *ext)
{
  strcpy(dest,prefix);
  MdcNewExt( dest, src, ext);
}

char *MdcAliasName(FILEINFO *fi, char alias[])
{
  char unknown[]="unknown";
  char *c, *patient, *patient_id, *study;
  Int16 year, month, day;
  Int16 hour, minute, second;
  Int32 series, acquisition, instance;

  patient    = strlen(fi->patient_name) ? fi->patient_name : unknown;
  patient_id = strlen(fi->patient_id)   ? fi->patient_id   : unknown;
  study      = strlen(fi->study_id)     ? fi->study_id     : unknown;

  year  = fi->study_date_year;
  month = fi->study_date_month;
  day   = fi->study_date_day;
  hour  = fi->study_time_hour;
  minute= fi->study_time_minute;
  second= fi->study_time_second; 

  switch (fi->iformat) {
    case MDC_FRMT_ACR:
    case MDC_FRMT_DICM: /* UID's */
      series = (fi->nr_series > 0) ? fi->nr_series : 0;
      acquisition = (fi->nr_acquisition > 0) ? fi->nr_acquisition : 0;
      instance = (fi->nr_instance > 0) ? fi->nr_instance : 0;

      sprintf(alias,"%s+%s+%hd%02hd%02hd+%02hd%02hd%02hd+%010d+%010d+%010d.ext"
                   ,patient,study
                   ,year,month,day
                   ,hour,minute,second
                   ,series,acquisition,instance);
      break;
    case MDC_FRMT_ANLZ: /* patient_id */
      sprintf(alias,"%s+%s+%hd%02hd%02hd+%02hd%02hd%02hd.ext"
                   ,patient_id,study
                   ,year,month,day
                   ,hour,minute,second);

      break;
    default: 
      sprintf(alias,"%s+%s+%hd%02hd%02hd+%02hd%02hd%02hd.ext"
                   ,patient,study
                   ,year,month,day
                   ,hour,minute,second);
  }

  /* change to lower and replace spaces */
  c=alias; while (*c) { *c=tolower((int)*c); if (isspace((int)*c)) *c='_'; c++;}

  return(alias);

}

/* make alias in opath, splitted ipath assumed */
void MdcEchoAliasName(FILEINFO *fi)
{
  MDC_ALIAS_NAME = MDC_YES; prefix[0]='\0';

  MdcDefaultName(fi,fi->iformat,fi->opath,fi->ifname);

  fprintf(stdout,"%s\n",fi->opath);
}

void MdcDefaultName(FILEINFO *fi, int format, char dest[], char *src)
{
  char alias[MDC_MAX_PATH];

  if (MDC_ALIAS_NAME == MDC_YES) src = MdcAliasName(fi,alias);

  switch (format) {
   case MDC_FRMT_RAW  : MdcNewName(dest,src,FrmtExt[MDC_FRMT_RAW]);    break;
   case MDC_FRMT_ASCII: MdcNewName(dest,src,FrmtExt[MDC_FRMT_ASCII]);  break;
#if MDC_INCLUDE_ACR
   case MDC_FRMT_ACR  : MdcNewName(dest,src,FrmtExt[MDC_FRMT_ACR]);    break;
#endif
#if MDC_INCLUDE_GIF
   case MDC_FRMT_GIF  : MdcNewName(dest,src,FrmtExt[MDC_FRMT_GIF]);    break;
#endif
#if MDC_INCLUDE_INW
   case MDC_FRMT_INW  : MdcNewName(dest,src,FrmtExt[MDC_FRMT_INW]);    break;
#endif
#if MDC_INCLUDE_ECAT
   case MDC_FRMT_ECAT6: MdcNewName(dest,src,FrmtExt[MDC_FRMT_ECAT6]);  break;
  #if MDC_INCLUDE_TPC
   case MDC_FRMT_ECAT7: MdcNewName(dest,src,FrmtExt[MDC_FRMT_ECAT7]);  break;
  #endif
#endif
#if MDC_INCLUDE_INTF
   case MDC_FRMT_INTF : MdcNewName(dest,src,FrmtExt[MDC_FRMT_INTF]);   break;
#endif
#if MDC_INCLUDE_ANLZ
   case MDC_FRMT_ANLZ : MdcNewName(dest,src,FrmtExt[MDC_FRMT_ANLZ]);   break;
#endif
#if MDC_INCLUDE_DICM
   case MDC_FRMT_DICM : MdcNewName(dest,src,FrmtExt[MDC_FRMT_DICM]);   break;
#endif
#if MDC_INCLUDE_PNG
   case MDC_FRMT_PNG  : MdcNewName(dest,src,FrmtExt[MDC_FRMT_PNG]);    break;
#endif
#if MDC_INCLUDE_CONC
   case MDC_FRMT_CONC : MdcNewName(dest,src,FrmtExt[MDC_FRMT_CONC]);   break;
#endif
#if MDC_INCLUDE_NIFTI
   case MDC_FRMT_NIFTI: MdcNewName(dest,src,FrmtExt[MDC_FRMT_NIFTI]);  break;
#endif
   default            : MdcNewName(dest,src,FrmtExt[MDC_FRMT_NONE]);   break; 
  } 

}

void MdcRenameFile(char *name)
{
  char *pbegin = NULL, *pend = NULL;

  MdcPrintLine('-',MDC_FULL_LENGTH);
  MdcPrntScrn("\tRENAME FILE\n");
  MdcPrintLine('-',MDC_FULL_LENGTH);

  pbegin = MdcGetLastPathDelim(name);    /* point to basename */

  if (pbegin == NULL) pbegin = name;
  else pbegin = pbegin + 1;

  strcpy(mdcbufr,pbegin);
  pend   = (char *)strrchr(mdcbufr,'.'); /* without extension */
  if (pend != NULL) pend[0] = '\0';

  MdcPrntScrn("\n\tOld Filename: %s\n",mdcbufr);
  MdcPrntScrn("\n\tNew Filename: ");
  MdcGetStrLine(mdcbufr,MDC_MAX_PATH-1,stdin);
  mdcbufr[MDC_MAX_PATH]='\0'; 
  MdcRemoveEnter(mdcbufr);
  strcpy(name,mdcbufr);

  MdcPrintLine('-',MDC_FULL_LENGTH);

}

/* always check for both path delimiters */
char *MdcGetLastPathDelim(char *path)
{
  char *p=NULL;

  if (path == NULL) return NULL;

  p = (char *)strrchr(path,'/');
  if (p != NULL) return(p);

  p = (char *)strrchr(path,'\\');
  return(p);

}


void MdcMySplitPath(char path[], char **dir, char **fname) 
{
  char *p=NULL;

  p = MdcGetLastPathDelim(path);       /* last path delim becomes '\0'       */

  if ( p == NULL ) {
    *fname=&path[0];
    *dir=NULL;
  } else {
    *p='\0';
    *dir=&path[0];
    *fname=p+1;
  }

}

void MdcMyMergePath(char path[], char *dir, char **fname) 
                                      /* first '\0' becomes path delim again */
{
  char *p;

  if ( dir != NULL ) {
    p=(char *)strchr(path,'\0');
    if ( p != NULL ) *p=MDC_PATH_DELIM_CHR;
  }

  *fname = &path[0];

}

void MdcFillImgPos(FILEINFO *fi, Uint32 nr, Uint32 plane, float translation)
{
  IMG_DATA *id = &fi->image[nr];

  /* according to device coordinates */

  switch (fi->pat_slice_orient) {

   case MDC_SUPINE_HEADFIRST_TRANSAXIAL: 
   case MDC_PRONE_HEADFIRST_TRANSAXIAL :
   case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
   case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
   case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
   case MDC_PRONE_FEETFIRST_TRANSAXIAL :
   case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
   case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
    id->image_pos_dev[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_dev[1]=-(id->pixel_ysize*(float)id->height);
    id->image_pos_dev[2]=-((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_SUPINE_HEADFIRST_SAGITTAL   :
   case MDC_PRONE_HEADFIRST_SAGITTAL    :
   case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
   case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
   case MDC_SUPINE_FEETFIRST_SAGITTAL   :
   case MDC_PRONE_FEETFIRST_SAGITTAL    :
   case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
   case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
    id->image_pos_dev[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_dev[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_dev[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_SUPINE_HEADFIRST_CORONAL    :
   case MDC_PRONE_HEADFIRST_CORONAL     :
   case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
   case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
   case MDC_SUPINE_FEETFIRST_CORONAL    :
   case MDC_PRONE_FEETFIRST_CORONAL     :
   case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
   case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
    id->image_pos_dev[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_dev[1]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_dev[2]=-(id->pixel_ysize*(float)id->height);
    break;
   default                              : { } /* do nothing */

  }

  /* according to the patient coordinate system */

  switch (fi->pat_slice_orient) {
   case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=-(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[2]=-((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_SUPINE_HEADFIRST_SAGITTAL   :
    id->image_pos_pat[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_SUPINE_HEADFIRST_CORONAL    :
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=-(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[2]=+((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_SUPINE_FEETFIRST_SAGITTAL   :
    id->image_pos_pat[0]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_SUPINE_FEETFIRST_CORONAL    :
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_PRONE_HEADFIRST_TRANSAXIAL :
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[2]=-((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_PRONE_HEADFIRST_SAGITTAL    :
    id->image_pos_pat[0]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_PRONE_HEADFIRST_CORONAL     :
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_PRONE_FEETFIRST_TRANSAXIAL :
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[2]=+((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_PRONE_FEETFIRST_SAGITTAL    : 
    id->image_pos_pat[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_PRONE_FEETFIRST_CORONAL     :
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
    id->image_pos_pat[0]=+(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
    id->image_pos_pat[0]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
    id->image_pos_pat[0]=+(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
    id->image_pos_pat[0]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
    id->image_pos_pat[0]=-(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
    id->image_pos_pat[0]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[1]=+((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
    id->image_pos_pat[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=-(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
    id->image_pos_pat[0]=-(id->pixel_ysize*(float)id->height);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+((id->slice_spacing*(float)(plane+1))+translation);
    break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
    id->image_pos_pat[0]=+(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
    id->image_pos_pat[0]=-((id->slice_spacing*(float)(plane+1))+translation);
    id->image_pos_pat[1]=-(id->pixel_xsize*(float)id->width);
    id->image_pos_pat[2]=+(id->pixel_ysize*(float)id->height);
    break;
   default                              : { } /* do nothing */
  }
}

void MdcFillImgOrient(FILEINFO *fi, Uint32 nr)
{
  IMG_DATA *id = &fi->image[nr];

  /* according to device coordinate system */

  switch (fi->pat_slice_orient) {
     case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
     case MDC_PRONE_HEADFIRST_TRANSAXIAL :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
     case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
     case MDC_PRONE_FEETFIRST_TRANSAXIAL :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
         id->image_orient_dev[0]=+1.0;    id->image_orient_dev[3]=+0.0;
         id->image_orient_dev[1]=-0.0;    id->image_orient_dev[4]=+1.0;
         id->image_orient_dev[2]=+0.0;    id->image_orient_dev[5]=-0.0;
         break; 

     case MDC_SUPINE_HEADFIRST_SAGITTAL   :
     case MDC_PRONE_HEADFIRST_SAGITTAL    :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
     case MDC_SUPINE_FEETFIRST_SAGITTAL   :
     case MDC_PRONE_FEETFIRST_SAGITTAL    :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
         id->image_orient_dev[0]=+0.0;    id->image_orient_dev[3]=+0.0; 
         id->image_orient_dev[1]=+1.0;    id->image_orient_dev[4]=-0.0;
         id->image_orient_dev[2]=-0.0;    id->image_orient_dev[5]=-1.0;
         break;

     case MDC_SUPINE_HEADFIRST_CORONAL    :
     case MDC_PRONE_HEADFIRST_CORONAL     :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
     case MDC_SUPINE_FEETFIRST_CORONAL    :
     case MDC_PRONE_FEETFIRST_CORONAL     :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
         id->image_orient_dev[0]=+1.0;    id->image_orient_dev[3]=+0.0; 
         id->image_orient_dev[1]=-0.0;    id->image_orient_dev[4]=-0.0;
         id->image_orient_dev[2]=+0.0;    id->image_orient_dev[5]=-1.0;
         break;

     default                              : { } 
  } 

  /* according to patient coordinate system */

  switch (fi->pat_slice_orient) {

   case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
         id->image_orient_pat[0]=+1.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=-0.0;    id->image_orient_pat[4]=+1.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=-0.0;
         break;
   case MDC_SUPINE_HEADFIRST_SAGITTAL   :
         id->image_orient_pat[0]=+0.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=+1.0;    id->image_orient_pat[4]=-0.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=-1.0;
         break;
   case MDC_SUPINE_HEADFIRST_CORONAL    :
         id->image_orient_pat[0]=+1.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=-0.0;    id->image_orient_pat[4]=-0.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=-1.0;
         break;
   case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
         id->image_orient_pat[0]=-1.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=+0.0;    id->image_orient_pat[4]=+1.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=-0.0;
         break;
   case MDC_SUPINE_FEETFIRST_SAGITTAL   :
         id->image_orient_pat[0]=+0.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=+1.0;    id->image_orient_pat[4]=+0.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=+1.0;
         break;
   case MDC_SUPINE_FEETFIRST_CORONAL    :
         id->image_orient_pat[0]=-1.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=+0.0;    id->image_orient_pat[4]=+0.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=+1.0;
         break;
   case MDC_PRONE_HEADFIRST_TRANSAXIAL :
         id->image_orient_pat[0]=-1.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=+0.0;    id->image_orient_pat[4]=-1.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=+0.0;
         break;
   case MDC_PRONE_HEADFIRST_SAGITTAL    :
         id->image_orient_pat[0]=-0.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=-1.0;    id->image_orient_pat[4]=-0.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=-1.0;
         break;
   case MDC_PRONE_HEADFIRST_CORONAL     :
         id->image_orient_pat[0]=-1.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=+0.0;    id->image_orient_pat[4]=-0.0;
         id->image_orient_pat[2]=-0.0;    id->image_orient_pat[5]=-1.0;
         break;
   case MDC_PRONE_FEETFIRST_TRANSAXIAL :
         id->image_orient_pat[0]=+1.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=-0.0;    id->image_orient_pat[4]=-1.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=+0.0;
         break;
   case MDC_PRONE_FEETFIRST_SAGITTAL    :
         id->image_orient_pat[0]=-0.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=-1.0;    id->image_orient_pat[4]=+0.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=+1.0;
         break;
   case MDC_PRONE_FEETFIRST_CORONAL     :
         id->image_orient_pat[0]=+1.0;    id->image_orient_pat[3]=-0.0;
         id->image_orient_pat[1]=-0.0;    id->image_orient_pat[4]=+0.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=+1.0;
         break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
         id->image_orient_pat[0]=+1.0;    id->image_orient_pat[3]=+0.0;
         id->image_orient_pat[1]=-0.0;    id->image_orient_pat[4]=+1.0;
         id->image_orient_pat[2]=+0.0;    id->image_orient_pat[5]=-0.0;
         break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
   case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
   case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
   case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
   case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
   case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
   case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
   case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
   case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
   case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
   case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
   /* FIXME */
   /* MdcPrntWarn("Extra code needed in %s at line %d\n", __FILE__, __LINE__);*/
   default                              : { } /* do nothing */
  }
}

int MdcGetOrthogonalInt(float f)
{
  int i; 

  if (f == 0.0) i = 0;
  else if (f == 1.0) i = 1;
  else if (f == -1.0) i = -1;
  else i = (f < 0) ? (int)(f - 0.5) : (int)(f + 0.5);

  return(i);

}

Int8 MdcGetPatSliceOrient(FILEINFO *fi, Uint32 i)
{
  IMG_DATA *id = &fi->image[i]; 
  int   i0,i1,i4,i5;
  int   slice_orientation=MDC_UNKNOWN;
  int   patient_orientation=MDC_UNKNOWN;
  int   patient_rotation=MDC_UNKNOWN;
  Int8  pat_slice_orient=MDC_UNKNOWN;

  i0 = MdcGetOrthogonalInt(id->image_orient_pat[0]);
  i1 = MdcGetOrthogonalInt(id->image_orient_pat[1]);
  i4 = MdcGetOrthogonalInt(id->image_orient_pat[4]);
  i5 = MdcGetOrthogonalInt(id->image_orient_pat[5]);

  /* A) image orientation combined with patient position */
  if (strstr(fi->pat_pos,"Unknown") == NULL) {

    /* patient orientation */
    if (strstr(fi->pat_pos,"HF") != NULL) {
      patient_orientation = MDC_HEADFIRST;
    }else if (strstr(fi->pat_pos,"FF") != NULL) {
      patient_orientation = MDC_FEETFIRST;
    }

    /* patient rotation */
    if (strstr(fi->pat_pos,"S") != NULL) {
      patient_rotation = MDC_SUPINE;
    }else if (strstr(fi->pat_pos,"P") != NULL) {
      patient_rotation = MDC_PRONE;
    }else if (strstr(fi->pat_pos, "DR") != NULL) {
      patient_rotation = MDC_DECUBITUS_RIGHT;
    }else if (strstr(fi->pat_pos, "DL") != NULL) {
      patient_rotation = MDC_DECUBITUS_LEFT;
    }

    /* slice orientation */
    if       ((i0 == +1 || i0 == -1) && (i4 == +1 || i4 == -1)) {
      slice_orientation = MDC_TRANSAXIAL;
    }else if ((i1 == +1 || i1 == -1) && (i5 == +1 || i5 == -1)) {
      slice_orientation = MDC_SAGITTAL;
    }else if ((i0 == +1 || i0 == -1) && (i5 == +1 || i5 == -1)) {
      slice_orientation = MDC_CORONAL;
    }

    /* combined result */
    switch (patient_rotation) {
      case MDC_SUPINE:
          switch (patient_orientation) {
            case MDC_HEADFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient = MDC_SUPINE_HEADFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient = MDC_SUPINE_HEADFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient = MDC_SUPINE_HEADFIRST_CORONAL;
                      break;
                }
                break;
            case MDC_FEETFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient = MDC_SUPINE_FEETFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient = MDC_SUPINE_FEETFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient = MDC_SUPINE_FEETFIRST_CORONAL;
                      break;
                }
                break;
          }
          break;
      case MDC_PRONE:
          switch (patient_orientation) {
            case MDC_HEADFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient = MDC_PRONE_HEADFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient = MDC_PRONE_HEADFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient = MDC_PRONE_HEADFIRST_CORONAL;
                      break;
                }
                break;
            case MDC_FEETFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient = MDC_PRONE_FEETFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient = MDC_PRONE_FEETFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient = MDC_PRONE_FEETFIRST_CORONAL;
                      break;
                }
                break;
          }
          break;
      case MDC_DECUBITUS_RIGHT:
          switch (patient_orientation) {
            case MDC_HEADFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL;
                      break;
                }
                break;
            case MDC_FEETFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient=MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL;
                      break;
                }
                break;
          }
          break;
      case MDC_DECUBITUS_LEFT:
          switch (patient_orientation) {
            case MDC_HEADFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL;
                      break;
                }
                break;
            case MDC_FEETFIRST:
                switch (slice_orientation) {
                  case MDC_TRANSAXIAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL;
                      break;
                  case MDC_SAGITTAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL;
                      break;
                  case MDC_CORONAL:
                      pat_slice_orient=MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL;
                      break;
                }
                break;
          }
          break;
    }

    if (pat_slice_orient != MDC_UNKNOWN) return(pat_slice_orient);

  }

  /* B) image orientation alone */

  if ((i0 == +1)   && (i4 == +1))   return MDC_SUPINE_HEADFIRST_TRANSAXIAL;
  if ((i0 == -1)   && (i4 == +1))   return MDC_SUPINE_FEETFIRST_TRANSAXIAL;
  if ((i0 == -1)   && (i4 == -1))   return MDC_PRONE_HEADFIRST_TRANSAXIAL;
  if ((i0 == +1)   && (i4 == -1))   return MDC_PRONE_FEETFIRST_TRANSAXIAL;
  /* FIXME - doesn't handle DECUBITUS positions */

  if ((i1 == +1)   && (i5 == -1))   return MDC_SUPINE_HEADFIRST_SAGITTAL;
  if ((i1 == +1)   && (i5 == +1))   return MDC_SUPINE_FEETFIRST_SAGITTAL;
  if ((i1 == -1)   && (i5 == -1))   return MDC_PRONE_HEADFIRST_SAGITTAL;
  if ((i1 == -1)   && (i5 == +1))   return MDC_PRONE_FEETFIRST_SAGITTAL;
  /* FIXME - doesn't handle DECUBITUS positions */

  if ((i0 == +1)   && (i5 == -1))   return MDC_SUPINE_HEADFIRST_CORONAL;
  if ((i0 == -1)   && (i5 == +1))   return MDC_SUPINE_FEETFIRST_CORONAL;
  if ((i0 == -1)   && (i5 == -1))   return MDC_PRONE_HEADFIRST_CORONAL;
  if ((i0 == +1)   && (i5 == +1))   return MDC_PRONE_FEETFIRST_CORONAL;
  /* FIXME - doesn't handle DECUBITUS positions */

  return(MDC_UNKNOWN);
}

Int8 MdcTryPatSliceOrient(char *pat_orient)
{
  char buffer[MDC_MAXSTR], *p1, *p2;
  Int8 orient1=MDC_UNKNOWN, orient2=MDC_UNKNOWN;

  MdcStringCopy(buffer,pat_orient,(Uint32)strlen(pat_orient));

  p1 = buffer;
  p2 = strrchr(buffer, '\\');

  if (p2 == NULL) return MDC_UNKNOWN;

  p2[0] = '\0'; p2+=1;

  if      (strchr(p1,'L') != NULL) orient1 = MDC_LEFT;
  else if (strchr(p1,'R') != NULL) orient1 = MDC_RIGHT;
  else if (strchr(p1,'A') != NULL) orient1 = MDC_ANTERIOR;
  else if (strchr(p1,'P') != NULL) orient1 = MDC_POSTERIOR;
  else if (strchr(p1,'H') != NULL) orient1 = MDC_HEAD;
  else if (strchr(p1,'F') != NULL) orient1 = MDC_FEET;

  if      (strchr(p2,'L') != NULL) orient2 = MDC_LEFT;
  else if (strchr(p2,'R') != NULL) orient2 = MDC_RIGHT;
  else if (strchr(p2,'A') != NULL) orient2 = MDC_ANTERIOR;
  else if (strchr(p2,'P') != NULL) orient2 = MDC_POSTERIOR;
  else if (strchr(p2,'H') != NULL) orient2 = MDC_HEAD;
  else if (strchr(p2,'F') != NULL) orient2 = MDC_FEET;


  if (orient1 == MDC_LEFT      && orient2 == MDC_POSTERIOR)
    return MDC_SUPINE_HEADFIRST_TRANSAXIAL;
  if (orient1 == MDC_POSTERIOR && orient2 == MDC_FEET)
    return MDC_SUPINE_HEADFIRST_SAGITTAL;
  if (orient1 == MDC_LEFT      && orient2 == MDC_FEET)
    return MDC_SUPINE_HEADFIRST_CORONAL;

  if (orient1 == MDC_RIGHT     && orient2 == MDC_POSTERIOR)
    return MDC_SUPINE_FEETFIRST_TRANSAXIAL;
  if (orient1 == MDC_POSTERIOR && orient2 == MDC_HEAD)
    return MDC_SUPINE_FEETFIRST_SAGITTAL;
  if (orient1 == MDC_RIGHT     && orient2 == MDC_HEAD)
    return MDC_SUPINE_FEETFIRST_CORONAL;

  if (orient1 == MDC_RIGHT     && orient2 == MDC_ANTERIOR)
    return MDC_PRONE_HEADFIRST_TRANSAXIAL;
  if (orient1 == MDC_ANTERIOR  && orient2 == MDC_FEET)
    return MDC_PRONE_HEADFIRST_SAGITTAL;
  if (orient1 == MDC_RIGHT     && orient2 == MDC_FEET)
    return MDC_PRONE_HEADFIRST_CORONAL;

  if (orient1 == MDC_LEFT      && orient2 == MDC_ANTERIOR)
    return MDC_PRONE_FEETFIRST_TRANSAXIAL;
  if (orient1 == MDC_ANTERIOR  && orient2 == MDC_HEAD)
    return MDC_PRONE_FEETFIRST_SAGITTAL;
  if (orient1 == MDC_LEFT      && orient2 == MDC_HEAD)
    return MDC_PRONE_FEETFIRST_CORONAL;

  if (orient1 == MDC_POSTERIOR  && orient2 == MDC_RIGHT)
    return MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL;
  if (orient1 == MDC_RIGHT      && orient2 == MDC_FEET)
    return MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL;
  if (orient1 == MDC_POSTERIOR  && orient2 == MDC_FEET)
    return MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL;

  if (orient1 == MDC_ANTERIOR   && orient2 == MDC_RIGHT)
    return MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL;
  if (orient1 == MDC_RIGHT      && orient2 == MDC_HEAD)
    return MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL;
  if (orient1 == MDC_ANTERIOR   && orient2 == MDC_HEAD)
    return MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL;

  if (orient1 == MDC_ANTERIOR && orient2 == MDC_LEFT)
    return MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL;
  if (orient1 == MDC_LEFT     && orient2 == MDC_FEET)
    return MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL;
  if (orient1 == MDC_ANTERIOR && orient2 == MDC_FEET)
    return MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL;

  if (orient1 == MDC_POSTERIOR && orient2 == MDC_LEFT)
    return MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL;
  if (orient1 == MDC_LEFT       && orient2 == MDC_FEET)
    return MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL;
  if (orient1 == MDC_POSTERIOR  && orient2 == MDC_FEET)
    return MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL;


  return MDC_UNKNOWN;
}

/* Formats that only support one rescale factor but no slope/intercept */
/* sometimes can lose quantitation: during rescale, the rescaled_fctr  */
/* is put to one. Warn the user about the loss of quantitation         */
/* example: negatives and force Uint8 pixel output                     */
Int8 MdcCheckQuantitation(FILEINFO *fi)
{
   IMG_DATA *id;
   Uint32 i;

   if (MDC_QUANTIFY || MDC_CALIBRATE) {

     for (i=0; i<fi->number; i++) {

        id = &fi->image[0];

        if (id->rescaled && (id->rescaled_fctr != id->rescaled_slope)) {
          MdcPrntWarn("Quantitation was lost");
          return(MDC_YES);
        }
     }
   }

   return(MDC_NO);

}

/* return heart rate in beats per minute */
float MdcGetHeartRate(GATED_DATA *gd, Int16 type)
{
   float heart_rate = 0.;

   if (gd->study_duration > 0.) { 
     switch (type) {
       case MDC_HEART_RATE_ACQUIRED:
         /* note: [ms] -> [min] */
         heart_rate = (gd->cycles_acquired * 60.0f * 1000.0f) / gd->study_duration;
         break;
       case MDC_HEART_RATE_OBSERVED:
         /* note: [ms] -> [min] */
         heart_rate = (gd->cycles_observed * 60.0f * 1000.0f) / gd->study_duration;
         break;
     }
   } 

   return(heart_rate);
}

