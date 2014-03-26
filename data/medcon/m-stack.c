/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-stack.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : stack files as specified                                 *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcGetNormSliceSpacing() - Get spacing between slices    *
 *                MdcStackSlices()         - Stack single slice image files*
 *                MdcStackFrames()         - Stack multi slice volume files*
 *                MdcStackFiles()          - Main stack routine            *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-stack.c,v 1.40 2010/08/28 23:44:23 enlf Exp $
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

#include <stdio.h>
#include <math.h>

#include "medcon.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

static FILEINFO infi, outfi;

static int mdc_nrstack=0;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

float MdcGetNormSliceSpacing(IMG_DATA *id1, IMG_DATA *id2)
{
  /* slice_spacing = sqrt( (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2 ) */
  float value, slice_spacing;
  float dx, dy, dz;

  slice_spacing = id1->slice_spacing;

  dx = id1->image_pos_pat[0] - id2->image_pos_pat[0];
  dy = id1->image_pos_pat[1] - id2->image_pos_pat[1];
  dz = id1->image_pos_pat[2] - id2->image_pos_pat[2];

  value = (float)sqrt((double)(dx*dx + dy*dy + dz*dz));

  if (fabs(slice_spacing - value) <= MDC_FLT_EPSILON)  {
    /* insignificant difference, use original header value */
    slice_spacing = id1->slice_spacing;
  }else{
    /* significant, use calculated value from image_pos_pat[] info */
    slice_spacing = (float)value;
  }

  return(slice_spacing);

}

/* tomo  : stack single slice image files into one 3D volume file  (2D+ -> 3D)*/
/* planar: stack planar slice image files into one single    file             */
char *MdcStackSlices(void)
{
  FILEINFO *ifi, *ofi;
  IMG_DATA *id1, *id2;
  DYNAMIC_DATA *dd1, *dd2;
  Uint32 d, nr_of_images;
  int HAS_DYNAMIC_DATA = MDC_NO;
  char *msg=NULL;
  int *total   = mdc_arg_total; /* total arguments of files & conversions */
  int *convs   = mdc_arg_convs; /* counter for each conversion format     */
  char **files = mdc_arg_files; /* array of pointers to input filenames   */
  int i, convert, c;
  float time_frame_duration=0.;

  ifi = &infi; ofi= &outfi;

  /* initialize output FILEINFO */
  MdcInitFI(ofi,"stack3d");
  nr_of_images = total[MDC_FILES];

  if ((ifi->dynnr > 0) && (ifi->dyndata != NULL)) HAS_DYNAMIC_DATA = MDC_YES;

  /* read and stack the several single slice files */
  for (i=0; i<total[MDC_FILES]; i++) {

     /* open file */
     if (MdcOpenFile(ifi,files[i]) != MDC_OK) {
       MdcCleanUpFI(ofi); return("stack slices : Failure to open file");
     }

     /* read the file */
     if (MdcReadFile(ifi,i,NULL) != MDC_OK) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       return("stack slices : Failure to read file");
     }

     if (i == 0) {
       /* copy FILEINFO stuff from 1st image */
       MdcCopyFI(ofi,ifi,MDC_NO,MDC_NO);

       /* set some specific parameters */
       MdcSplitPath(ofi->ipath,ofi->idir,ofi->ifname);
       ofi->dim[0] = 3;
       ofi->dim[1] = ifi->dim[1];
       ofi->dim[2] = ifi->dim[2];
       ofi->dim[3] = nr_of_images;
       ofi->pixdim[0] = 3.;
       ofi->pixdim[1] = ifi->pixdim[1];
       ofi->pixdim[2] = ifi->pixdim[2];

       if (ofi->planar == MDC_NO) ofi->acquisition_type = MDC_ACQUISITION_TOMO;

       if (!MdcGetStructDD(ofi,1)) {
         MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
         return("stack slices : Couldn't alloc output DYNAMIC_DATA structs");
       }else{
         ofi->dyndata[0].nr_of_slices = nr_of_images;
       }

       if (!MdcGetStructID(ofi,nr_of_images)) {
         MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
         return("stack slices : Couldn't alloc output ING_DATA structs");
       }

       /* remember time_frame_duration */
       if (HAS_DYNAMIC_DATA == MDC_YES) {
         time_frame_duration = ifi->dyndata[0].time_frame_duration;
       }

     }else{

       if (HAS_DYNAMIC_DATA == MDC_YES) {
         dd1 = &ofi->dyndata[0];
         dd2 = &ifi->dyndata[0];

         /* check time_frame_duration differences */
         if (time_frame_duration != dd2->time_frame_duration) {
           MdcPrntWarn("stack slices : Different image durations found");
         }

         /* planar = increment total time_frame_duration */
         if (ofi->planar == MDC_YES) {
           dd1->time_frame_duration += dd2->time_frame_duration;
         }

       }

     }

     /* sanity checks */
     for (d=3; d < MDC_MAX_DIMS; d++) if (ifi->dim[d] > 1 ) {
        MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
        return("stack slices : Only single slice (one image) files supported");
     }

     if (ifi->dim[3] == 0) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       return("stack slices : File without image found");
     }

     /* copy IMG_DATA info */
     msg = MdcCopyID(&ofi->image[i],&ifi->image[0],MDC_YES);
     if (msg != NULL) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       sprintf(mdcbufr,"stack slices : %s",msg); return(mdcbufr);
     }

     /* small checks for file integrity */
     if (i > 0) {
       id1 = &ifi->image[0];   /* current  slice */
       id2 = &ofi->image[i-1]; /* previous slice */

       if (ifi->pat_slice_orient != ofi->pat_slice_orient) {
         MdcPrntWarn("stack slices : Different 'patient_slice_orient' found");
       }

       if ((id1->width != id2->width) || (id1->height != id2->height)) {
         MdcPrntWarn("stack slices : Different image dimensions found");
       }
       if (id1->slice_width != id2->slice_width) {
         MdcPrntWarn("stack slices : Different slice thickness found");
       }
       if (id1->slice_spacing != id2->slice_spacing) {
         MdcPrntWarn("stack slices : Different slice spacing found");
       }
       if (id1->type != id2->type) {
         MdcPrntWarn("stack slices : Different pixel type found");
       }
     }

     MdcCleanUpFI(ifi);

  }

  /* check all the images */
  msg = MdcImagesPixelFiddle(ofi);
  if (msg != NULL) {
    MdcCleanUpFI(ofi);
    sprintf(mdcbufr,"stack slices : %s",msg); return(mdcbufr);
  }

  if (ofi->planar == MDC_NO) {
    /* check for orthogonal slices */
    switch (ofi->pat_slice_orient) {
      case MDC_SUPINE_HEADFIRST_SAGITTAL            :
      case MDC_SUPINE_FEETFIRST_SAGITTAL            :
      case MDC_PRONE_HEADFIRST_SAGITTAL             :
      case MDC_PRONE_FEETFIRST_SAGITTAL             :
      case MDC_SUPINE_HEADFIRST_CORONAL             :
      case MDC_SUPINE_FEETFIRST_CORONAL             :
      case MDC_PRONE_HEADFIRST_CORONAL              :
      case MDC_PRONE_FEETFIRST_CORONAL              :
      case MDC_SUPINE_HEADFIRST_TRANSAXIAL          :
      case MDC_SUPINE_FEETFIRST_TRANSAXIAL          :
      case MDC_PRONE_HEADFIRST_TRANSAXIAL           :
      case MDC_PRONE_FEETFIRST_TRANSAXIAL           : 
      case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL   :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL   :
      case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL    :
      case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL    :
      case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL    :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL    :
      case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL     :
      case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL     :
      case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL :
      case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL  :
      case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL  : break;
      default:
        MdcPrntWarn("stack slices : Probably file with Non-Orthogonal slices");
    }
  }

  /* correct slice_spacing */
  for (i=1; i<nr_of_images; i++) {
     id1 = &ofi->image[i];
     id2 = &ofi->image[i-1];
     id1->slice_spacing=MdcGetNormSliceSpacing(id1,id2);
  }
  /* and also for the first image */
  if (nr_of_images > 1) {
    ofi->image[0].slice_spacing = ofi->image[1].slice_spacing;
  }

  /* if requested, reverse slices */
  if (MDC_SORT_REVERSE == MDC_YES) {
    msg = MdcSortReverse(ofi);
    if (msg != NULL) return(msg); 
  }

  /* write the file */
  if (total[MDC_CONVS] > 0) {
    /* go through conversion formats */
    for (c=1; c<MDC_MAX_FRMTS; c++) {
      convert = convs[c];
      /* write output format when selected */
      while (convert -- > 0) {
        if (MdcWriteFile(ofi, c, mdc_nrstack++, NULL) != MDC_OK) {
          MdcCleanUpFI(ofi);
          return("stack slices : Failure to write file");
        }
      }
    }
  }

  MdcCleanUpFI(ofi);

  return(NULL);

}

/* tomo  : stack volumes at different time frames into one 4D file (3D+ -> 4D)*/
/* planar: stack planar dynamic files into one planar dynamic file            */
char *MdcStackFrames(void)
{

  FILEINFO *ifi, *ofi;
  Uint32 d, nr_of_frames, nr_of_images=0;
  char *msg = NULL;
  int  *total  = mdc_arg_total; /* total arguments of files & conversions */
  int  *convs  = mdc_arg_convs; /* counter for each conversion format     */
  char **files = mdc_arg_files; /* array of pointers to input filenames   */
  int  i, j, f, convert, c;

  ifi = &infi; ofi = &outfi;

  /* initialize output FILEINFO */
  MdcInitFI(ofi,"stack4d");
  nr_of_frames = total[MDC_FILES];

  for (i=0, j=0, f=0; f < total[MDC_FILES]; f++) {

     /* open file */
     if (MdcOpenFile(ifi,files[f]) != MDC_OK) {
       MdcCleanUpFI(ofi);
       return("stack frames : Failure to open file");
     }

     /* read the file */
     if (MdcReadFile(ifi,f,NULL) != MDC_OK) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       return("stack frames : Failure to read file");
     }
     MdcCloseFile(ifi->ifp); /* no further need */

     /* sanity checks */
     for (d=4; d<MDC_MAX_DIMS; d++) if (ifi->dim[d] > 1) {
        MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
        return("stack frames : Only tomo volumes or planar dynamic supported");
     }
     if ((ifi->dim[3] == 1) && (ifi->planar == MDC_NO)) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       return("stack frames : Use option '-stacks' for single slice files");
     }
     if (ifi->dim[3] == 0) {
       MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
       return("stack frames : File without images found");
     }

     if (f == 0) {
       /* copy FILEINFO stuff from 1st file */
       MdcCopyFI(ofi,ifi,MDC_NO,MDC_NO);

       /* 4D -> dynamic */
       ofi->acquisition_type = MDC_ACQUISITION_DYNAMIC;

       /* get appropriate structs */
       if (!MdcGetStructDD(ofi,nr_of_frames)) {
         MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
         return("stack frames : Couldn't alloc output DYNAMIC_DATA structs");
       }

       /* set some specific parameters */
       if (ofi->planar == MDC_YES) {
         /* planar: asymmetric */
         nr_of_images= ifi->number; /* increment */
         ofi->dim[0] = 3;
         ofi->dim[1] = ifi->dim[1];
         ofi->dim[2] = ifi->dim[2];
         ofi->dim[3] = nr_of_images;
         ofi->pixdim[0] = 3.;
         ofi->pixdim[1] = ifi->pixdim[1];
         ofi->pixdim[2] = ifi->pixdim[2];
         ofi->pixdim[3] = ifi->pixdim[3];
       }else{
         /* tomo  : symmectric */
         nr_of_images= ifi->number * nr_of_frames;
         ofi->dim[0] = 4;
         ofi->dim[1] = ifi->dim[1];
         ofi->dim[2] = ifi->dim[2];
         ofi->dim[3] = ifi->dim[3];
         ofi->dim[4] = nr_of_frames;
         ofi->pixdim[0] = 4.;
         ofi->pixdim[1] = ifi->pixdim[1];
         ofi->pixdim[2] = ifi->pixdim[2];
         ofi->pixdim[3] = ifi->pixdim[3];
         ofi->pixdim[4] = ofi->dyndata[0].time_frame_duration; /* tomo */
       }

       /* malloc all IMG_DATA structs */
       if (!MdcGetStructID(ofi,nr_of_images)) {
         MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
         return("stack frames : Couldn't alloc output IMG_DATA structs");
       }

     }else{

       if (ofi->planar == MDC_YES) {
         nr_of_images += ifi->number;
         ofi->dim[3] = nr_of_images;
         /* malloc IMG_DATA structs current frame */
         if (!MdcGetStructID(ofi,nr_of_images)) {
           MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
           return("stack frames : Couldn't alloc planar IMG_DATA structs");
         }
       }

       /* copy DYNAMIC_DATA struct when available   */
       /* f=0 already copied at initial MdcCopyFI() */
       if ((ifi->dynnr > 0) && (ifi->dyndata != NULL)) {
         MdcCopyDD(&ofi->dyndata[f],&ifi->dyndata[0]);
       }

       /* suspectable differences */
       if (ifi->pat_slice_orient != ofi->pat_slice_orient) {
         MdcPrntWarn("stack frames : Different 'patient_slice_orient' found");
       }
       if (ifi->planar != ofi->planar) {
         MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
         return("stack frames : wrongful mixture of tomo and planar frames");
       }
     }

     for (i=0; i<ifi->dim[3]; i++, j++) {
        /* copy IMG_DATA info */
        msg = MdcCopyID(&ofi->image[j],&ifi->image[i],MDC_YES);
        if (msg != NULL) {
          MdcCleanUpFI(ofi); MdcCleanUpFI(ifi);
          sprintf(mdcbufr,"stack frames : %s",msg); return(mdcbufr);
        }
     }

     MdcCleanUpFI(ifi);

  }

  /* check all the images */
  msg = MdcImagesPixelFiddle(ofi);
  if (msg != NULL) {
    MdcCleanUpFI(ofi);
    sprintf(mdcbufr,"stack frames : %s",msg); return(mdcbufr);
  }

  if (ofi->planar == MDC_NO) {
    /* check for orthogonal slices */
    switch (ofi->pat_slice_orient) {
      case MDC_SUPINE_HEADFIRST_SAGITTAL            :
      case MDC_SUPINE_FEETFIRST_SAGITTAL            :
      case MDC_PRONE_HEADFIRST_SAGITTAL             :
      case MDC_PRONE_FEETFIRST_SAGITTAL             :
      case MDC_SUPINE_HEADFIRST_CORONAL             :
      case MDC_SUPINE_FEETFIRST_CORONAL             :
      case MDC_PRONE_HEADFIRST_CORONAL              :
      case MDC_PRONE_FEETFIRST_CORONAL              :
      case MDC_SUPINE_HEADFIRST_TRANSAXIAL          :
      case MDC_SUPINE_FEETFIRST_TRANSAXIAL          :
      case MDC_PRONE_HEADFIRST_TRANSAXIAL           :
      case MDC_PRONE_FEETFIRST_TRANSAXIAL           : 
      case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL   :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL   :
      case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL    :
      case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL    :
      case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL    :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL    :
      case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL     :
      case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL     :
      case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL :
      case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL :
      case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL  :
      case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL  : break;
      default:
        MdcPrntWarn("stack frames : Probably file with Non-Orthogonal slices");
    }
  }

  /* write the file */
  if (total[MDC_CONVS] > 0) {
    /* go through conversion formats */
    for (c=1; c<MDC_MAX_FRMTS; c++) {
      convert = convs[c];
      /* write output format when selected */
      while (convert -- > 0) {
        if (MdcWriteFile(ofi, c, mdc_nrstack++, NULL) != MDC_OK) {
          MdcCleanUpFI(ofi);
          return("stack frames : Failure to write file");
        }
      }
    }
  }

  MdcCleanUpFI(ofi);

  return(NULL);

}

char *MdcStackFiles(Int8 stack)
{
  char *msg=NULL;

  if (MDC_CONVERT != MDC_YES)
    return("In order to stack specify an output format");

  if (mdc_arg_total[MDC_FILES] == 1) 
    return("In order to stack at least two files are required");

  switch (stack) {
    case MDC_STACK_SLICES: msg = MdcStackSlices();
        break;
    case MDC_STACK_FRAMES: msg = MdcStackFrames();
        break;
  }

  return(msg);
}
