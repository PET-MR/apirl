/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-split.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : split file as specified                                  *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcGetSplitAcqType()    - Get new acquisition type       *
 *                MdcGetNrSplit()         - Get current split index        *
 *                MdcGetSplitBaseName()   - Get basename without prefix    *
 *                MdcUpdateSplitPrefix()  - Update prefix for filename     *
 *                MdcCopySlice()          - Copy specified slice in new FI *
 *                MdcCopyFrame()          - Copy specified frame in new FI *
 *                MdcSplitSlices()        - Write each slice to a file     *
 *                MdcSplitFrames()        - Write each frame to a file     *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-split.c,v 1.29 2010/08/28 23:44:23 enlf Exp $
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

#include "medcon.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

static Uint32 mdc_nrsplit=0;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

Int16 MdcGetSplitAcqType(FILEINFO *fi)
{
  Int16 type = MDC_ACQUISITION_TOMO; /* default */

  if (fi->planar && (fi->acquisition_type == MDC_ACQUISITION_STATIC))
    type = MDC_ACQUISITION_STATIC;

  if (fi->planar && (fi->acquisition_type == MDC_ACQUISITION_DYNAMIC))
    type = MDC_ACQUISITION_DYNAMIC;

  return(type);

}

Uint32 MdcGetNrSplit(void)
{
  return(mdc_nrsplit);
}

char *MdcGetSplitBaseName(char *path)
{
  char *p, *bname;

  /* terminate path */
  p = MdcGetLastPathDelim(path);
  if (p != NULL) {
    p[0]='\0'; bname = p + 1;
  }else{
    bname = path;
  }

  /* get pure basename without prefix we need to update */
  if (bname[0]=='m' && bname[4]=='-' && bname[10]=='-' &&
     (bname[5]=='s' || bname[5]=='f')) {
     bname += 11;
  }

  return(bname);

}

void MdcUpdateSplitPrefix(char *dpath, char *spath, char *bname, int nr )
{
  MdcPrefix(nr);
  strcpy(dpath,spath);
  strcat(dpath,MDC_PATH_DELIM_STR);
  strcat(dpath,prefix);
  strcat(dpath,bname);
}

/* copy a single image slice (zero-based number) */
char *MdcCopySlice(FILEINFO *ofi, FILEINFO *ifi, Uint32 slice0)
{
  char *msg;
  IMG_DATA *idin, *idout;
  DYNAMIC_DATA *dd;
  Uint32 i;

  /* copy original FILEINFO struct */
  msg = MdcCopyFI(ofi,ifi,MDC_NO,MDC_NO); if (msg != NULL) return(msg);

  /* preserve dynamic data of single slice */ 
  idin = &ifi->image[slice0];
  if (!MdcGetStructDD(ofi,1)) return("Couldn't malloc DYNAMIC_DATA struct");
  dd = &ofi->dyndata[0]; 
  dd->nr_of_slices = 1;
  dd->time_frame_start    = idin->slice_start;
  dd->time_frame_duration = MdcSingleImageDuration(ifi,idin->frame_number-1);
  /* MARK: frame_delay as alternative for slice_start
  dd->time_frame_delay    = idin->slice_start; 
   */

  /* single slice parameters */
  ofi->dim[0]=3; ofi->pixdim[0]=3.; ofi->dim[3] = 1; ofi->pixdim[3] = 1.;
  for(i=4; i<MDC_MAX_DIMS; i++) { ofi->dim[i]=1; ofi->pixdim[i]=1.; }
  ofi->acquisition_type = MdcGetSplitAcqType(ifi);

  /* get new IMG_DATA struct for slice */
  ofi->image = NULL;
  if (!MdcGetStructID(ofi,1)) 
    return("Couldn't malloc new IMG_DATA struct"); 

  /* copy IMG_DATA struct */
  idin = &ifi->image[slice0];
  idout= &ofi->image[0];

  msg = MdcCopyID(idout,idin,MDC_YES); if (msg != NULL) return(msg);

  idout->frame_number = 1; /* single frame */
 
  /* integrity check of FILEINFO struct */
  if ( (msg=MdcCheckFI(ofi)) != NULL) return(msg);

  return(NULL);

}

/* copy a (time) frame of images (zero-based number) */
char *MdcCopyFrame(FILEINFO *ofi, FILEINFO *ifi, Uint32 frame0)
{
  char *msg;
  IMG_DATA *idin, *idout;
  DYNAMIC_DATA *dd;
  Uint32 i, begin, slices;

  /* copy FILEINFO struct */
  msg = MdcCopyFI(ofi,ifi,MDC_NO,MDC_NO); if (msg != NULL) return(msg);

  /* preserve corresponding dynamic data */
  if ((ifi->dynnr > 0) && (ifi->dyndata != NULL)) {
    if (frame0 < ifi->dynnr) {
      if (!MdcGetStructDD(ofi,1))
        return("Couldn't malloc DYNAMIC_DATA struct");
      MdcCopyDD(&ofi->dyndata[0],&ifi->dyndata[frame0]);
    }
  }

  /* get begin and total slices of frame */
  if (ifi->planar && (ifi->acquisition_type == MDC_ACQUISITION_DYNAMIC)) {
    dd = &ifi->dyndata[frame0];
    slices = (frame0<ifi->dynnr) ? dd->nr_of_slices : ifi->dim[3];
    for (begin=0, i=0; i<frame0; i++) begin += ifi->dyndata[i].nr_of_slices;
  }else{
    slices = (Uint32)ifi->dim[3];
    begin  = slices * frame0;
  }

  /* set single frame parameters */
  ofi->dim[0] = 3; ofi->pixdim[0]=3.;
  ofi->dim[3] = (Int16)slices;
  for(i=4; i<MDC_MAX_DIMS; i++) {
     ofi->dim[i]=1;
     ofi->pixdim[i]=1.;
  }

  MdcDebugPrint("output slices = %d",slices);
  
  ofi->acquisition_type = MdcGetSplitAcqType(ifi);

  /* disable ACQ_DATA structs */
  /* ofi->acqnr = 0; ofi->acqdata = NULL; */

  /* get new IMG_DATA structs for slices */
  ofi->image = NULL;
  if (!MdcGetStructID(ofi,slices)) 
    return("Couldn't malloc new IMG_DATA structs"); 

  /* copy IMG_DATA information */
  for (i=0; i < slices; i++) {

    /* copy IMG_DATA struct */
    idin = &ifi->image[begin+i];
    idout= &ofi->image[i];
    msg = MdcCopyID(idout,idin,MDC_YES); if (msg != NULL) return(msg);
    idout->frame_number = 1; /* single frame */
  }

  /* integrity check of FILEINFO struct */
  if ( (msg=MdcCheckFI(ofi)) != NULL) return(msg);

  return(NULL);

}

char *MdcSplitSlices(FILEINFO *fi, int format, int prefixnr)
{
  FILEINFO *ofi;
  Uint32 nr_of_slices;
  Int32 instance=0, series=0;
  char *msg, *tpath=NULL, *bname=NULL;

  /* alloc temp struct, path */
  ofi = (FILEINFO *)malloc(sizeof(FILEINFO));
  if (ofi == NULL) return("Couldn't malloc output struct");

  tpath = (char *)malloc(MDC_MAX_PATH);
  if (tpath == NULL) return("Couldn't malloc tpath");
  if (XMDC_GUI == MDC_NO) {
    MdcGetSafeString(tpath,fi->ifname,strlen(fi->ifname),MDC_MAX_PATH);
  }else{
    /* terminate path and get basename */
    MdcGetSafeString(tpath,fi->ofname,strlen(fi->ofname),MDC_MAX_PATH);
    bname = MdcGetSplitBaseName(tpath);
  }

  /* preserve & initialize series number */
  series = fi->nr_series; fi->nr_series = (Int32)prefixnr + 1;

  /* preserve & initialize instance number */
  instance = fi->nr_instance; fi->nr_instance = 0;

  /* split up all slices */
  nr_of_slices = fi->number;
  for (mdc_nrsplit=0; mdc_nrsplit < nr_of_slices; mdc_nrsplit++) {

     /* increment instance for each slice */
     fi->nr_instance = (Int32)mdc_nrsplit + 1;

     msg = MdcCopySlice(ofi,fi,mdc_nrsplit);
     if (msg != NULL) {
       fi->nr_instance = instance;
       MdcCleanUpFI(ofi); MdcFree(ofi);
       MdcFree(tpath);
       return("Failure to copy slice");
     }

     /* prepare filename */
     if (XMDC_GUI == MDC_NO) {
       strcpy(ofi->ipath,tpath); ofi->ifname = ofi->ipath;
     }else{
       MdcUpdateSplitPrefix(ofi->opath,tpath,bname,prefixnr);
       ofi->ofname = ofi->opath;
     }

     if (MdcWriteFile(ofi, format, prefixnr, NULL) != MDC_OK) {
       fi->nr_instance = instance;
       MdcCleanUpFI(ofi); MdcFree(ofi);
       MdcFree(tpath);
       return("Failure to write splitted slice");
     } 

     MdcCleanUpFI(ofi);
  }

  /* free mallocs */
  MdcFree(ofi);
  MdcFree(tpath);

  /* restore series */
  fi->nr_series = series;

  /* restore instance */
  fi->nr_instance = instance;

  return(NULL);

}

char *MdcSplitFrames(FILEINFO *fi, int format, int prefixnr)
{
  FILEINFO *ofi;
  Int32 instance=0, series=0;
  Uint32 i, nr_of_frames=1;
  char *msg, *tpath=NULL, *bname=NULL, *p=NULL;

  /* alloc temp struct, path */
  ofi = (FILEINFO *)malloc(sizeof(FILEINFO));
  if (ofi == NULL) return("Couldn't malloc output struct");

  tpath = (char *)malloc(MDC_MAX_PATH);
  if (tpath == NULL) return("Couldn't malloc tpath");
  if (XMDC_GUI == MDC_NO) {
    MdcGetSafeString(tpath,fi->ifname,strlen(fi->ifname),MDC_MAX_PATH);
  }else{
    MdcGetSafeString(tpath,fi->ofname,strlen(fi->ofname),MDC_MAX_PATH);
    p = MdcGetLastPathDelim(tpath);
    if (p != NULL) {
      p[0]='\0'; bname = p + 1;
    }else{
      bname = tpath;
    }
    /* get pure basename without prefix we need to update */
    if (bname[0]=='m' && bname[4]=='-' && bname[10]=='-' &&
       (bname[5]=='s' || bname[5]=='f')) {
       bname += 11;
    }
  }

  /* preserve & initialize series number */
  series = fi->nr_series; fi->nr_series = (Int32)prefixnr + 1;

  /* preserve & initialize instance number */
  instance = fi->nr_instance; fi->nr_instance = 0;

  if (fi->planar && (fi->acquisition_type == MDC_ACQUISITION_DYNAMIC)) {
    nr_of_frames = fi->dynnr;
  }else{
    for (i=4; i<MDC_MAX_DIMS; i++)  nr_of_frames *= (Uint32)fi->dim[i];
  }

  /* split up all frames */
  for (mdc_nrsplit=0; mdc_nrsplit < nr_of_frames; mdc_nrsplit++) {
 
     /* increment instance for each frame */
     fi->nr_instance = (Int32)mdc_nrsplit + 1;

     msg = MdcCopyFrame(ofi,fi,mdc_nrsplit);
     if (msg != NULL) {
       fi->nr_instance = instance;
       MdcCleanUpFI(ofi); MdcFree(ofi);
       MdcFree(tpath);
       return("Failure to copy frame");
     }

     /* prepare filename */
     if (XMDC_GUI == MDC_NO) {
       strcpy(ofi->ipath,tpath); ofi->ifname = ofi->ipath;
     }else{
       MdcUpdateSplitPrefix(ofi->opath,tpath,bname,prefixnr);
       ofi->ofname = ofi->opath;
     }

     if (MdcWriteFile(ofi, format, prefixnr, NULL) != MDC_OK) {
       fi->nr_instance = instance;
       MdcCleanUpFI(ofi); MdcFree(ofi);
       MdcFree(tpath);
       return("Failure to write splitted frame");
     } 

     MdcCleanUpFI(ofi);
  }

  /* free mallocs */
  MdcFree(ofi);
  MdcFree(tpath);

  /* restore series */
  fi->nr_series = series;

  /* restore instance */
  fi->nr_instance = instance;

  return(NULL);

}
