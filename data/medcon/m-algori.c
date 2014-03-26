/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-algori.c                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : endian/image algorithms                                  *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcRealloc()            - if (NULL) malloc else realloc  *
 *                MdcCeilPwr2()           - Find least power of two >=     *
 *                MdcRotateAngle()        - Rotate angle in degrees        *
 *                MdcDoSwap()             - Test to swap                   *
 *                MdcHostBig()            - Test if host is bigendian      *
 *                MdcSwapBytes()          - Swap the bytes                 *
 *                MdcForceSwap()          - Forced bytes swapping          *
 *                MdcIEEEfl_to_VAXfl()    - Change hostfloat to VAX float  *
 *                MdcVAXfl_to_IEEEfl()    - Change VAX float to hostfloat  *
 *                MdcType2Bytes()         - Pixel data type in bytes       *
 *                MdcType2Bits()          - Pixel data type in bits        *
 *                MdcTypeIntMax()         - Give maximum of integer type   *
 *                MdcSingleImageDuration()- Get duration of a single image *
 *                MdcImagesPixelFiddle()  - Process all pixels & images    *
 *                MdcGetDoublePixel()     - Get pixel from memory buffer   *
 *                MdcPutDoublePixel()     - Put pixel to memory buffer     *
 *                MdcDoSimpleCast()       - Test cast sufficient to rescale*
 *                MdcGetResizedImage()    - Make image of same size        *
 *                MdcGetDisplayImage()    - Get  image to display          *
 *                MdcMakeBIT8_U()         - Make an Uint8 image            *
 *                MdcGetImgBIT8_U()       - Get  an Uint8 image            *
 *                MdcMakeBIT16_S()        - Make an Int16 image            *
 *                MdcGetImgBIT16_S()      - Get  an Int16 image            *
 *                MdcMakeBIT32_S()        - Make an Int32 image            *
 *                MdcGetImgBIT32_S()      - Get  an Int32 image            *
 *                MdcMakeFLT32()          - Make a  float image            *
 *                MdcGetImgFLT32()        - Get  a  float image            *
 *                MdcMakeImgSwapped()     - Make an endian swapped image   *
 *                MdcGetImgSwapped()      - Get  an endian swapped image   *
 *                MdcUnpackBIT12()        - Unpack 12 bit into Uint16      *
 *                MdcHashDJB2()           - Get hash using djb2 method     *
 *                MdcHashSDBM()           - Get hash using sdbm method     *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-algori.c,v 1.86 2010/08/28 23:44:23 enlf Exp $ 
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
#define __USE_ISOC99 1
#include <math.h>
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

#define maxmin(x, max, min)          { maxmin_t=(x);                     \
                                       if (maxmin_t > max) max=maxmin_t; \
                                       if (maxmin_t < min) min=maxmin_t; \
                                     }

static Uint8 MDC_ALLOW_CAST = MDC_YES;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/


void *MdcRealloc(void *p, Uint32 bytes)
{
  void *ptmp;

  if (p == NULL) {
    ptmp = malloc(bytes);
  }else{
    ptmp = realloc(p, bytes);
  }

  return(ptmp);
}

/* Find least power of 2 greater than or equal to x */
/* Hacker's Delight - Henry S. Warren, Jr.          */
/* ISBN 0-201-91465-4 / pg. 48                      */
Uint32 MdcCeilPwr2(Uint32 x)
{
  x = x - 1;
  x = x | (x >>  1);
  x = x | (x >>  2);
  x = x | (x >>  4);
  x = x | (x >>  8);
  x = x | (x >> 16);
  return(x + 1);
}

float MdcRotateAngle(float angle, float rotate)
{
  float new_angle;

  new_angle = (float)fmod((double)rotate - (double)angle + 360., 360.);

  return(new_angle);
}

/* the machine-endian soup */
/*
READING:   host     infile    swap
-----------------------------------
           big0      big0      0
           big0      little1   1
           little1   big0      1 
           little1   little1   0 
----------------------------------
READING:   host  XOR file  =  swap


WRITING:   host     outfile   swap 
----------------------------------
           big0      big0      0
           little1   big0      1

           big0      little1   1 
           little1   little1   0 
----------------------------------
WRITING    host  XOR file  =  swap
*/

int MdcDoSwap(void) 
{
  return(MDC_HOST_ENDIAN^MDC_FILE_ENDIAN); 
}

int MdcHostBig(void)
{
  if (MDC_HOST_ENDIAN == MDC_BIG_ENDIAN) return 1;
  return 0;
}

void MdcSwapBytes(Uint8 *ptr, int bytes)
{
  int i, j;

  if ( MdcDoSwap() ) 
    for (i=0,j=bytes-1;i < (bytes/2); i++, j--) {
       ptr[i]^=ptr[j]; ptr[j]^=ptr[i]; ptr[i]^=ptr[j];
    }
}

void MdcForceSwap(Uint8 *ptr, int bytes)
{
  int i, j;

  for (i=0,j=bytes-1;i < (bytes/2); i++, j--) {
     ptr[i]^=ptr[j]; ptr[j]^=ptr[i]; ptr[i]^=ptr[j];
  }
}

void MdcIEEEfl_to_VAXfl(float *f)
{ 
  Uint16 exp;
  union { 
          Uint16 t[2];
          float t4;
  } test;

  test.t4 = *f;

  if (test.t4 != 0.0) {
    if (!MdcHostBig()) { /* swap words */
      Uint16 temp;
      memcpy((void *)&temp,(void *)&test.t[0],2);
      memcpy((void *)&test.t[0],(void *)&test.t[1],2);
      memcpy((void *)&test.t[1],(void *)&temp,2);
    }

    exp =  ((test.t[0] & 0x7f00) + 0x0100) & 0x7f00;
    test.t[0] = (test.t[0] & 0x80ff) + exp;

    MdcSwapBytes((Uint8 *)&test.t[0],2);
    MdcSwapBytes((Uint8 *)&test.t[1],2);

  }

  memcpy((void *)f,(void *)&test.t4,4);

}

void MdcVAXfl_to_IEEEfl(float *f)
{
  Uint16  t1, t2;
  union {
          Uint16 n[2];
          float  n4;
  } number;

  union {
          Uint32 t3;
          float  t4;
  } test;

  number.n4 = *f;

  if (MdcHostBig()) {
    Uint16 temp;
    temp = number.n[0]; number.n[0]=number.n[1]; number.n[1]=temp;
  } 

  MdcSwapBytes((Uint8 *)&number.n4,4);

  if ((number.n[0] != 0) || (number.n[1] != 0) ) { 
    t1 = number.n[0] & 0x80ff;
    t2 = (((number.n[0])&0x7f00)+0xff00)&0x7f00;
    test.t3 = (t1+t2)<<16;
    test.t3 = test.t3+number.n[1];

    number.n4 = test.t4;
  }

  memcpy((void *)f,(void *)&number.n4,4);

}

int MdcFixFloat(float *ref)
{
  float value = *ref;
  int fixed = 0;

#ifdef HAVE_ISNAN
  if (isnan(value)) { value = 0.; fixed = 1; }
#endif
#ifdef HAVE_ISINF
  if (isinf(value)) { value = 0.; fixed = 1; }
#endif

  *ref = value;

  return(fixed);
}

int MdcFixDouble(double *ref)
{
  double value = *ref;
  int fixed = 0;

#ifdef HAVE_ISNAN
  if (isnan(value)) { value = 0.; fixed = 1; }
#endif
#ifdef HAVE_ISINF
  if (isinf(value)) { value = 0.; fixed = 1; }
#endif

  *ref = value;

  return(fixed);
}

int MdcType2Bytes(int type)
{
  int bytes = 0; 

  switch (type) {

    case  BIT1  :
    case  BIT8_S:
    case  BIT8_U: bytes = 1; break;

    case BIT16_S:
    case BIT16_U: bytes = 2; break;

    case COLRGB : bytes = 3; break;

    case BIT32_S:
    case BIT32_U:
    case FLT32  :
    case VAXFL32: bytes = 4; break;
   
    case ASCII  : /* read as double */
#ifdef HAVE_8BYTE_INT
    case BIT64_S:
    case BIT64_U:
#endif
    case FLT64  : bytes = 8; break;
  }
 
  return(bytes);

}



int MdcType2Bits(int type)
{
  int bits = 0; 

  switch (type) {

    case  BIT1  : bits = 1; break;

    case  BIT8_S:
    case  BIT8_U: bits = 8; break;

    case BIT16_S:
    case BIT16_U: bits = 16; break;

    case COLRGB : bits = 24; break;

    case BIT32_S:
    case BIT32_U:
    case FLT32  :
    case VAXFL32: bits = 32; break;

    case ASCII  : /* read as double */
#ifdef HAVE_8BYTE_INT
    case BIT64_S:
    case BIT64_U:
#endif
    case FLT64  : bits = 64; break;
  }
 
  return(bits);

}

double MdcTypeIntMax(int type)
{
  switch (type) {

    case BIT8_S : return(127.);
    case BIT8_U : return(255.);
    case BIT16_S: return(32767.);
    case BIT16_U: return(65535.);
    case BIT32_S: return(2147483647.);
    case BIT32_U: return(4294967295.);
#ifdef HAVE_8BYTE_INT
    case BIT64_S: return(9223372036854775807.);
    case BIT64_U: return(18446744073709551615.);
#endif

  }

  return(0.0);

}

float MdcSingleImageDuration(FILEINFO *fi, Uint32 frame)
{
  DYNAMIC_DATA *dd;
  float duration, slices;

  if ((fi->dynnr == 0) || (fi->dyndata == NULL)) return(0.);

  if (frame >= fi->dynnr) return(0.);

  dd = &fi->dyndata[frame];

  if (dd->nr_of_slices == 0) return 0.;

  slices = (float)dd->nr_of_slices;

  /* planar -> each slice  separate: Tslice = Tframe/Nslices */  
  /* tomo   -> all  slices at once : Tslice = Tframe         */
  duration  = dd->time_frame_duration;                   /* no frame delay  */
  duration -= ((slices - 1) * dd->delay_slices);         /* no slice delays */
  if (fi->planar) duration /= slices;                    /* time per slice  */

  return(duration); /* [ms] */
}

/*
 pixel by pixel processes: yes, THE all in one routine 
     - swap bytes 
     - make positive 
     - get global & image  variables
 check some parameters 

 WARNING: double pixel types may get corrupted quantification values
          because our quantification is stripped down to a float!!
*/
char *MdcImagesPixelFiddle(FILEINFO *fi)
{
  DYNAMIC_DATA *dd;
  IMG_DATA *id, *idprev;
  STATIC_DATA *sd;
  Uint32 f, i, j, n, s, t;
  float start, duration;
  double fmin=0., fmax=0., qfmin=0., qfmax=0.; 
  char *msg;
  int FixWarn=0;

  /* initial checks for FILEINFO integrity */
  if (fi->number == 0) return("Internal Error ## Improper fi->number value");

  /* make sure fi->dim[] are 1-based */
  for (i=0; i<MDC_MAX_DIMS; i++) if (fi->dim[i] <= 0) fi->dim[i] = 1;

  /* check number of slices */
  for (t=1, i=3; i <= fi->dim[0]; i++) {
     MdcDebugPrint("dim[] TEST : fi->dim[%d] = %u",i,fi->dim[i]);
     t *= fi->dim[i];
  }
 
  if (fi->number != t) {
    /* return("Internal Error ## Improper fi->dim values"); */

    if (((t / fi->dim[3]) > 1) && (fi->planar == MDC_NO)) {
      /* complain when non-planar multi-dimensional array was found */
      MdcPrntWarn("Internal Error ## Improper fi->dim values\n" \
                  "\t\t - falling back to one dimensional array");
    }

    fi->dim[0] = 3; fi->dim[3] = fi->number;
    for (i=4; i<MDC_MAX_DIMS; i++) fi->dim[i] = 1;
 
  }


  /* sanity check ACQ_DATA stuff */
  if (fi->acqdata == NULL) fi->acqnr = 0;

  /* sanity check GATED_DATA structs */
  if (fi->gdata == NULL) fi->gatednr = 0;

  /* sanity check DYN_DATA structs  + updates */
  if (fi->dyndata == NULL) fi->dynnr = 0;

  if (fi->dynnr > 0) {
    /* check number of slices */
    for (i=0, t=0; i<fi->dynnr; i++) { 
       t += fi->dyndata[i].nr_of_slices;
    }
    if (t != fi->number) {

      /* reset to one frame */
      if (!MdcGetStructDD(fi,1))
        return("Internal Error ## Failure to realloc DYNAMIC_DATA struct");

      fi->dyndata[0].nr_of_slices = fi->number;

      MdcPrntWarn("Internal Warning ## Bad DYNAMIC_DATA values fixed");

    }

    /* go through all frames */
    for (f=0, t=0; f<fi->dynnr; f++) {
       dd = &fi->dyndata[f];

       /* update frame_start values */
       if ((f > 0) && (dd->time_frame_start == 0.)) {
         dd->time_frame_start = fi->dyndata[f-1].time_frame_start 
                              + fi->dyndata[f-1].time_frame_delay
                              + fi->dyndata[f-1].time_frame_duration;
       }

       /* set initial slice values */
       start = dd->time_frame_start + dd->time_frame_delay;
       duration = MdcSingleImageDuration(fi,f);

       /* update frame & start for each slice */
       for (s=0; s<dd->nr_of_slices; s++, t++) {
          id = &fi->image[t];
          id->frame_number = f+1; /* must be one based */
          id->slice_start    = start;
          if (fi->planar == MDC_YES) start += (duration + dd->delay_slices);
       }        
    }
  }
 
  /* fill in orientation information */
  if (strcmp(fi->pat_pos,"Unknown") == 0)
    strcpy(fi->pat_pos,MdcGetStrPatPos(fi->pat_slice_orient));
  if (strcmp(fi->pat_orient,"Unknown") == 0)
    strcpy(fi->pat_orient,MdcGetStrPatOrient(fi->pat_slice_orient));

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Checking images:");

  /* sanity check IMG_DATA stuff */
  if (fi->image == NULL)
    return("Internal Error ## Missing IMG_DATA structs");

  for (j=0; j<fi->number; j++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);

     id = &fi->image[j];

     /* check dimension/pixeltype values */
     if ( id->width == 0 || id->height == 0 || 
          id->bits == 0  || id->type == 0   ||
          id->buf == NULL) {
       return("Internal Error ## Improper IMG_DATA values");
     }

     if (id->pixel_xsize   <= 0.0 ) id->pixel_xsize = 1.0;
     if (id->pixel_ysize   <= 0.0 ) id->pixel_ysize = 1.0;
     if (id->slice_width   <= 0.0 ) id->slice_width = 1.0;
     if (id->slice_spacing <= 0.0 ) id->slice_spacing = id->slice_width;
     if (id->ct_zoom_fctr  <= 0.0 ) id->ct_zoom_fctr= 1.0;

     id->bits = MdcType2Bits(id->type);

     if (id->type != fi->image[0].type) fi->diff_type = MDC_YES;
     if (id->quant_scale != fi->image[0].quant_scale) fi->diff_scale = MDC_YES;
     if (id->calibr_fctr != fi->image[0].calibr_fctr) fi->diff_scale = MDC_YES;
     if (id->intercept   != fi->image[0].intercept  ) fi->diff_scale = MDC_YES;
     if (j == 0) {
       fi->mwidth  = id->width;
       fi->mheight = id->height;
     }else{
       if (id->width != fi->mwidth) {
         fi->diff_size = MDC_YES; 
         if (id->width > fi->mwidth ) fi->mwidth = id->width;
       }
       if (id->height != fi->mheight) {
         fi->diff_size = MDC_YES;
         if (id->height > fi->mheight ) fi->mheight = id->height;
       }
     }
  }

  /* set some global values */
  fi->dim[1] = (Int16) fi->mwidth;
  fi->dim[2] = (Int16) fi->mheight;

  fi->bits  = fi->image[0].bits;
  fi->type  = fi->image[0].type;

  /* check for really ugly things */
  if (fi->dim[0] <= 2 ) {
    sprintf(mdcbufr,"Internal Error ## fi->dim[0]=%d",fi->dim[0]);
    return(mdcbufr);
  }else{
    for (t=1; t<=fi->dim[0]; t++) {
       if (fi->dim[t] <= 0 ) {
         sprintf(mdcbufr,"Internal Error ## fi->dim[%d]=%d",t,fi->dim[t]);
         return(mdcbufr);
       }
    }
  }

  /* fixable things */
  if (fi->pixdim[0] == 3.0 ||
      fi->pixdim[0] == 4.0 ||
      fi->pixdim[0] == 5.0 ||
      fi->pixdim[0] == 6.0 ||
      fi->pixdim[0] == 7.0 ) { 
    for (t=1; t<=(Int32)fi->pixdim[0]; t++) {
       if (fi->pixdim[t] <= 0.0 ) fi->pixdim[t] = 1.;
    }
  }else{
    fi->pixdim[0] = 3.;
    fi->pixdim[1] = 1.;
    fi->pixdim[2] = 1.;
    fi->pixdim[3] = 1.;
  }

  /* color */
  if (fi->map == MDC_MAP_PRESENT) {
    msg=MdcHandleColor(fi);
    if (msg != NULL) return(msg);
  }

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Processing images:");

  /* max/min, endian, quantitation */ 
  for (j=0; j<fi->number; j++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);

     id = &fi->image[j];
     n = id->width * id->height;
     sd = id->sdata;

     if (MDC_FORCE_RESCALE) {
       id->quant_scale = mdc_si_slope;
       id->calibr_fctr = 1.;
       id->intercept   = mdc_si_intercept; 
     }

     if (MDC_QUANTIFY) {
       id->rescale_slope = id->quant_scale;
       id->rescale_intercept = id->intercept;
     }else if (MDC_CALIBRATE) {
       id->rescale_slope = id->quant_scale * id->calibr_fctr;
       id->rescale_intercept = id->intercept;
     }else{
       id->rescale_slope = 1.;
       id->rescale_intercept = 0.;
     }

     if (fi->contrast_remapped == MDC_YES) {
       /* any rescale already done */
       id->quant_scale = 1.;
       id->calibr_fctr = 1.;
       id->intercept   = 0.;
       id->rescale_slope = 1.;
       id->rescale_intercept = 0.;
     }

     switch (id->type) {

      case BIT8_U:
       {
         Uint8 *pix = (Uint8 *) id->buf, pix0;
         Uint8 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,1);

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT8_S:
       { 
         Int8 *pix = (Int8 *) id->buf, pix0;
         Int8 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,1);
         if (!MDC_NEGATIVE && (pix0 < 0)) pix0 = 0;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            if (!MDC_NEGATIVE && (*pix < 0)) *pix = 0;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;

         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT16_U:
       {
         Uint16 *pix = (Uint16 *) id->buf, pix0;
         Uint16 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,2);
         MdcSwapBytes((Uint8 *)&pix0, 2);

         /* init max,min values */ 
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 2);
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT16_S:
       {
         Int16 *pix = (Int16 *) id->buf, pix0;
         Int16 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,2);
         MdcSwapBytes((Uint8 *)&pix0, 2);
         if (!MDC_NEGATIVE && (pix0 < 0)) pix0 = 0;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 2);
            if (!MDC_NEGATIVE && (*pix < 0)) *pix = 0;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT32_U:
       {
         Uint32 *pix = (Uint32 *) id->buf, pix0;
         Uint32 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,4);
         MdcSwapBytes((Uint8 *)&pix0, 4);

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 4);
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT32_S:
       {
         Int32 *pix = (Int32 *) id->buf, pix0;
         Int32 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,4);
         MdcSwapBytes((Uint8 *)&pix0, 4);
         if (!MDC_NEGATIVE && (pix0 < 0)) pix0 = 0;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 4);
            if (!MDC_NEGATIVE && (*pix < 0)) *pix = 0;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
#ifdef HAVE_8BYTE_INT
      case BIT64_U:
       { 
         Uint64 *pix = (Uint64 *) id->buf, pix0;
         Uint64 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,8);
         MdcSwapBytes((Uint8 *)&pix0, 8);

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 8);
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
      case BIT64_S:
       {
         Int64 *pix = (Int64 *) id->buf, pix0;
         Int64 max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,8);
         MdcSwapBytes((Uint8 *)&pix0, 8);
         if (!MDC_NEGATIVE && (pix0 < 0)) pix0 = 0;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 8);
            if (!MDC_NEGATIVE && (*pix < 0)) *pix = 0;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;
#endif
      case FLT32:
       {
         float *pix = (float *) id->buf, pix0;
         float max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,4);
         MdcSwapBytes((Uint8 *)&pix0, 4);
         FixWarn |= MdcFixFloat(&pix0);
         if (!MDC_NEGATIVE && (pix0 < 0.)) pix0 = 0.;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = (double) pix0;
           fi->glmax = (double) pix0;
         }

         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 4);
            FixWarn |= MdcFixFloat(pix);
            if (!MDC_NEGATIVE && (*pix < 0.)) *pix = 0.;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix;
         }
         id->max = (double) max;
         id->min = (double) min;
       }
       break;

      case FLT64:
       {
         double *pix = (double *) id->buf, pix0;
         double max, min, maxmin_t;

         /* init first pixel */
         memcpy((void *)&pix0,(void *)pix,8);
         MdcSwapBytes((Uint8 *)&pix0, 8);
         FixWarn |= MdcFixDouble(&pix0);
         if (!MDC_NEGATIVE && (pix0 < 0.)) pix0 = 0.;

         /* init max,min values */
         min = pix0;
         max = pix0;
         if (j == 0) {
           fi->glmin = pix0;
           fi->glmax = pix0;
         }

         /* go through all pixels */
         for (i=0; i<n; i++, pix++) {
            MdcSwapBytes((Uint8 *)pix, 8);
            FixWarn |= MdcFixDouble(pix);
            if (!MDC_NEGATIVE && (*pix < 0.)) *pix = 0.;
            maxmin(*pix, max, min);
            if (sd != NULL) sd->total_counts += (float)*pix; /* overflow */
         }
         id->max = max;
         id->min = min;
       }
       break;
     }

     /* no negatives -> min = 0 */
     if (!MDC_NEGATIVE && (id->min < 0.)) id->min = 0.;

     /* handle global max,min values */
     if (j == 0) {
       fi->glmin = id->min;  fi->glmax = id->max;
     }else{
       if ( id->max > fi->glmax ) fi->glmax = id->max;
       if ( id->min < fi->glmin ) fi->glmin = id->min;
     }

     /* get quantified max,min */
     id->qmin = (double)((float)id->min * id->rescale_slope);
     id->qmin += (double)id->rescale_intercept;

     id->qmax = (double)((float)id->max * id->rescale_slope);
     id->qmax += (double)id->rescale_intercept;

     /* negative slope -> reverse qmax,qmin */
     if (id->rescale_slope < 0.) {
       double x;
       x = id->qmin; id->qmin = id->qmax; id->qmax = x;
     }

     /* handle quantified global min values */
     if (j == 0) {
       fi->qglmin = id->qmin;
     }else{
       if ( id->qmin < fi->qglmin) fi->qglmin = id->qmin;
     }
     /* handle quantified global max values */ 
     if (j == 0) { 
       fi->qglmax = id->qmax;
     }else{
       if ( id->qmax > fi->qglmax ) fi->qglmax = id->qmax;
     }

     /* handle the max/min values for the frame group */
     if ( (j % fi->dim[3]) == 0 ) {
       /* a frame boundary */
       if (j == 0) { 
         /* the beginning frame group */
         fmin  = id->min;  fmax  = id->max; 
         qfmin = id->qmin; qfmax = id->qmax;
       }else{
         /* new frame group, fill in the values for previous frame */
         for (t=j - fi->dim[3]; t<j ; t++) {
            idprev = &fi->image[t];

            idprev->fmin  = fmin;  idprev->fmax  = fmax;
            idprev->qfmin = qfmin; idprev->qfmax = qfmax;
         }
         /* re-initialize the values for the new frame group */
         fmin  = id->min;  fmax  = id->max;
         qfmin = id->qmin; qfmax = id->qmax;
       }
     }else{ 
       /* inside a frame group, determine min/max values */
       if (id->min < fmin ) fmin = id->min;
       if (id->max > fmax ) fmax = id->max;

       if (id->qmin < qfmin ) qfmin = id->qmin;
       if (id->qmax > qfmax ) qfmax = id->qmax;

     } 

  }

  /* warn about fixed values */
  if (FixWarn) MdcPrntWarn("Fixed pixels with bad float value (= set to zero)");

  /* don't forget to fill in the min/max values for the last frame group */
  for (t=j - fi->dim[3]; t<j ; t++) {        
     idprev = &fi->image[t];

     idprev->fmin  = fmin;  idprev->fmax  = fmax;
     idprev->qfmin = qfmin; idprev->qfmax = qfmax;
  }

  /* MARK: Prevent strange side effect of negative & raw reading: */
  /*            raw reading => enables negative values            */
  /*       In this case min value is really the min value found   */
  /*       and not set to zero. In case min value is >0 this value*/
  /*       will be displayed in XMedCon as black which is not     */
  /*       wanted sometimes (ex. DICOM's converted to pgm files)  */
  /*       Here we check if fi->glmin > 0  and fi->qglmin > 0     */
  /*       in order to set the min values to zero                 */
  if (MDC_NEGATIVE == MDC_YES && fi->glmin > 0. && fi->qglmin > 0.) {
    fi->glmin = 0.; fi->qglmin = 0.;
    for (t = 0; t < fi->number; t++) {
       fi->image[t].min   = 0.;
       fi->image[t].qmin  = 0.;
       fi->image[t].fmin  = 0.;
       fi->image[t].qfmin = 0.;
    }
  }

  /* from here on, endianess is host based */
  MDC_FILE_ENDIAN = MDC_HOST_ENDIAN; 

  return(NULL);

} 

double MdcGetDoublePixel(Uint8 *buf, int type)
{
  double value=0.0;

  switch (type) {
    case BIT8_U:
        {
           Uint8 *pix = (Uint8 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT8_S:
        {
           Int8 *pix = (Int8 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT16_U:
        {
           Uint16 *pix = (Uint16 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT16_S:
        {
           Int16 *pix = (Int16 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT32_U:
        {
           Uint32 *pix = (Uint32 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT32_S:
        {
           Int32 *pix = (Int32 *)buf;
           value = (double)pix[0];
        }
        break;
#ifdef HAVE_8BYTE_INT
    case BIT64_U:
        {
           Uint64 *pix = (Uint64 *)buf;
           value = (double)pix[0];
        }
        break;
    case BIT64_S:
        {
           Int64 *pix = (Int64 *)buf;
           value = (double)pix[0];
        }
        break;
#endif
    case FLT32:
        {
           float *pix = (float *)buf;
           value = (double)pix[0];
        }
        break;
    case FLT64:
        {
           double *pix = (double *)buf;
           value = pix[0];
        }
        break;
  }

  return(value);

}

void MdcPutDoublePixel(Uint8 *buf, double pix, int type)
{
 unsigned int bytes = (unsigned int)MdcType2Bytes(type);

 switch (type) {
  case BIT8_S:
   {
     Int8 c = (Int8) pix;
     buf[0] = c; 
   }
   break; 
  case BIT8_U:
   {
     Uint8 c = (Uint8) pix;
     buf[0] = c;
   }
   break;
  case BIT16_S:
   {
     Int16 c = (Int16) pix;
     memcpy(buf,(Uint8 *)&c,bytes); 
   }
   break;
  case BIT16_U:
   {
     Uint16 c = (Uint16) pix;
     memcpy(buf,(Uint8 *)&c,bytes);
   }
   break;
  case BIT32_S:
   {
     Int32 c = (Int32) pix; 
     memcpy(buf,(Uint8 *)&c,bytes); 
   }
   break;
  case BIT32_U:
   {
     Uint32 c = (Uint32) pix;
     memcpy(buf,(Uint8 *)&c,bytes); 
   }
   break;
#ifdef HAVE_8BYTE_INT
  case BIT64_S:
   {
     Int64 c = (Int64) pix;
     memcpy(buf,(Uint8 *)&c,bytes);
   }
   break;
  case BIT64_U:
   {
     Uint64 c = (Uint64) pix;
     memcpy(buf,(Uint8 *)&c,bytes);
   }
   break;
#endif 
  case FLT32:
   {
     float c = (float) pix;
     memcpy(buf,(Uint8 *)&c,bytes); 
   }
   break;
  case FLT64:
   {
     double c = (double) pix;
     memcpy(buf,(Uint8 *)&c,bytes); 
   }
   break;
 }

}

int MdcDoSimpleCast(double minv, double maxv, double negmin, double posmax)
{
  Int32 casted;

  /* Rescaling to new integer values (Int32, Int16, Uint8): when original
   * values are integers and within the range of the new pixel type, a 
   * simple cast would do - without rescaling
   *                      - preserving original values 
   */

  if (MDC_ALLOW_CAST == MDC_NO) return(MDC_NO);

  /* TEST #1:  simple cast test -> original values = integer ? */
  casted = (Int32)minv; if ((double)casted != minv) return(MDC_NO);
  casted = (Int32)maxv; if ((double)casted != maxv) return(MDC_NO);

  /* TEST #2: within new pixel type range ? */
  if (minv < negmin || maxv > posmax) return(MDC_NO);

  return(MDC_YES);

}


Uint8 *MdcGetResizedImage(FILEINFO *fi,Uint8 *buffer,int type,Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint32 h, p, bytes, linesize, size;
  Uint32 lpad, rpad, tpad, bpad, linepad;
  Uint8 *lbuf=NULL, *rbuf=NULL, *linebuf=NULL, *pbuf;
  double pval;
  Uint8 *maxbuf, *obuf, *ibuf=buffer;

  if (id->type == COLRGB) {
    MdcPrntWarn("Resizing true color RGB images unsupported");
    return(NULL);
  }

  if (id->rescaled) {
    pval = id->rescaled_min;
  }else{
    pval = id->min;
  }

  bytes = MdcType2Bytes(type);

  linesize = id->width * bytes;

  size = fi->mwidth * fi->mheight * bytes;
  maxbuf = MdcGetImgBuffer(size);
  if (maxbuf == NULL) return NULL;

  obuf = maxbuf;

  /* calculate padding (left, right, top, bottom) */
  linepad = fi->mwidth;

  switch (MDC_PADDING_MODE) {

    case MDC_PAD_AROUND:
        lpad = (fi->mwidth - id->width) / 2;
        rpad = (fi->mwidth - id->width + 1) / 2;   /* +1 for int rounding */
        tpad = (fi->mheight - id->height) / 2;
        bpad = (fi->mheight - id->height + 1) / 2; /* +1 for int rounding */
        break;
    case MDC_PAD_BOTTOM_RIGHT:
        lpad = 0;
        rpad = fi->mwidth - id->width;
        tpad = 0;
        bpad = fi->mheight - id->height;
        break;
    case MDC_PAD_TOP_LEFT:
        lpad = fi->mwidth - id->width;
        rpad = 0;
        tpad = fi->mheight - id->height;
        bpad = 0;
        break;
    default: /* MDC_PAD_BOTTOM_RIGHT: */
        lpad = 0;
        rpad = fi->mwidth - id->width;
        tpad = 0;
        bpad = fi->mheight - id->height;
  }

  /* malloc & fill left, right and full line buffers */
  if (lpad > 0) {
    lbuf = malloc(bytes * lpad);
    if (lbuf == NULL) {
      MdcFree(maxbuf);
      return(NULL);
    }
     
    pbuf = lbuf;
    for (p = 0; p < lpad; p++) {
       MdcPutDoublePixel(pbuf,pval,type);
       pbuf += bytes;
    }
  }
  if (rpad > 0) {
    rbuf = malloc(bytes * rpad);
    if (rbuf == NULL) {
      MdcFree(maxbuf); MdcFree(lbuf);
      return(NULL); 
    }
     
    pbuf = rbuf;
    for (p = 0; p < rpad; p++) {
       MdcPutDoublePixel(pbuf,pval,type);
       pbuf += bytes;
    }
  }

  if ((tpad > 0) || (bpad > 0)) { 
    linebuf = malloc(bytes * linepad);
    if (linebuf == NULL) {
      MdcFree(maxbuf); MdcFree(lbuf); MdcFree(rbuf);
      return(NULL);
    }

    pbuf = linebuf;
    for (p = 0; p < linepad; p++) {
       MdcPutDoublePixel(pbuf,pval,type);
       pbuf += bytes;
    }
  }


  for (h=0; h < fi->mheight; h++) {
     if ( (h < tpad) || h >= (fi->mheight - bpad) ) {
       /* pad a full line at top or bottom */
       memcpy(obuf,linebuf,linepad*bytes);
       obuf += linepad*bytes;
     }else{
       /* copy an image line */
       if (lpad > 0) {
         /* first pad line left */
         memcpy(obuf,lbuf,lpad*bytes);
         obuf += lpad*bytes;
       }

       /* now copy line */
       memcpy(obuf,ibuf,linesize);
       obuf += linesize;
       ibuf += linesize;

       if (rpad > 0) {
         /* then pad line right */
         memcpy(obuf,rbuf,rpad*bytes);
         obuf += rpad*bytes;
       }
     }
  }

  MdcFree(lbuf);
  MdcFree(rbuf);
  MdcFree(linebuf);

  return(maxbuf);

}

/* get buffer for screen display */
Uint8 *MdcGetDisplayImage(FILEINFO *fi, Uint32 img)
{
   Uint8 *buf, RESTORE=MDC_ALLOW_CAST;
   Uint32 width, height, bytes;

   if (fi->image[img].type == COLRGB) {
     /* RGB */
     width  = fi->image[img].width;
     height = fi->image[img].height;
   
     bytes = width * height * 3;

     buf = malloc(bytes);
     if (buf != NULL) memcpy(buf,fi->image[img].buf,bytes);

   }else{
     /* indexed */

     if (fi->map == MDC_MAP_PRESENT) {
       /* color */
       MDC_ALLOW_CAST = MDC_YES;
     }else{
       /* gray */
       MDC_ALLOW_CAST = MDC_NO;
     }

     buf = MdcGetImgBIT8_U(fi,img);

     MDC_ALLOW_CAST = RESTORE;

   }

   return(buf);
}


Uint8 *MdcMakeBIT8_U(Uint8 *cbuf, FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint8 *buf=(Uint8 *)cbuf, *pixel, DO_QUANT_CALIBR;

  Uint32 i, n = id->width * id->height;
  double pixval, min, max, idmin, idmax, scale=1.0;
  float  newval;

  /* get proper maximum/minimum value */
  if (MDC_QUANTIFY || MDC_CALIBRATE) {
    DO_QUANT_CALIBR = MDC_YES;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->qfmin;
      max = id->qfmax;
    }else{
      min = fi->qglmin;
      max = fi->qglmax;
    }
  }else{
    DO_QUANT_CALIBR = MDC_NO;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->fmin;
      max = id->fmax;
    }else{
      min = fi->glmin;
      max = fi->glmax;
    }
  }
  scale = (max == min) ? 1. : 255./(max - min);

  if (MdcDoSimpleCast(min,max,0.,255.) == MDC_YES) {
    scale = 1.;  min = 0.;
  }

  switch( id->type ) {
    case BIT1: /* convert bits to byte  */
        {
          /* to avoid a premature overwrite, we must begin from the end */
          Uint8 masktable[8]={0x80,0x40,0x20,0x10,0x08,0x04,0x02,0x01};

          for (i=n; i>0; i--)
             if(buf[(i-1) >> 3] & masktable[(i-1) & 7]) buf[i-1]=0xff;
             else buf[i-1]=0x00;
        }
        break;

    default:   /* anything else to byte */
        {
          for (pixel=id->buf, i=0; i<n; i++, pixel+=MdcType2Bytes(id->type)) {
             pixval = MdcGetDoublePixel(pixel,id->type);
             if (DO_QUANT_CALIBR) {
               pixval *= (double)id->rescale_slope;
               pixval += (double)id->rescale_intercept;
             }
             newval  = (float) (scale * (pixval - min));
             buf[i]  = (Uint8) newval;
          }
        }
  }

  id->rescaled = MDC_YES;
  if (DO_QUANT_CALIBR) {
    id->rescaled_fctr = (min < 0.) ? 1. : 1./scale;
    id->rescaled_slope= 1./scale;
    id->rescaled_intercept = min; 
    idmax = id->qmax; idmin = id->qmin;
  }else{
    id->rescaled_fctr = 1.;
    id->rescaled_slope= 1.;
    id->rescaled_intercept = 0.;
    idmax = id->max;  idmin = id->min;
  }
  id->rescaled_max  = (double)((Uint8)(scale * (idmax  - min)));
  id->rescaled_min  = (double)((Uint8)(scale * (idmin  - min)));

  return(buf);

} 

/* converts to Uint8 */
Uint8 *MdcGetImgBIT8_U(FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint32 size = id->width * id->height * MdcType2Bytes(BIT8_U);
  Uint8 *buffer;

  if ( (buffer=(Uint8 *)malloc(size)) == NULL )  return NULL;

  buffer=MdcMakeBIT8_U(buffer,fi,img);

  return((Uint8 *)buffer); 
  
} 

Uint8 *MdcMakeBIT16_S(Uint8 *cbuf, FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint8 *pixel, DO_QUANT_CALIBR, DO_LINEAR_SCALE=MDC_NO;
  Int16 *buf = (Int16 *)cbuf;
  Uint32 i, n = id->width * id->height;
  double pixval, min, max, idmin, idmax, scale=1.0;
  double SMAX, UMAX, negmin, posmax;
  float newval;

  UMAX = (double)(1 << MDC_INT16_BITS_USED);     /* 16-bits: 65536 */
  SMAX = (double)(1 << (MDC_INT16_BITS_USED-1)); /* 16-bits: 32768 */

  /* get proper maximum/minimum value */
  if (MDC_QUANTIFY || MDC_CALIBRATE) {
    DO_QUANT_CALIBR = MDC_YES;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->qfmin;
      max = id->qfmax;
    }else{
      min = fi->qglmin;
      max = fi->qglmax;
    }
  }else{
    DO_QUANT_CALIBR = MDC_NO;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->fmin;
      max = id->fmax;
    }else{
      min = fi->glmin;
      max = fi->glmax;
    }
  }

  /* set limit values */
  switch (MDC_INT16_BITS_USED) {
    case 16: /* signed */
        negmin = -SMAX; posmax = SMAX - 1.; break;
    default: /* unsigned */
        negmin = 0.;    posmax = UMAX - 1.;
  }

  /* check scale type: linear / affine */
  if (DO_QUANT_CALIBR) {
    /* get scale to transform max positive value to max new type */
    /* check whether neg values scale within neg range, which    */
    /* allows linear transform, otherwise affine transform used  */
    DO_LINEAR_SCALE = ((min * posmax / max) >= negmin) ? MDC_YES : MDC_NO;
  }

  /* linear scaling, do not shift to positive range */
  if (DO_LINEAR_SCALE == MDC_YES) min = 0.;

  /* set scale value */
  scale = (max == min) ? 1. : posmax / (max - min);

  if (MdcDoSimpleCast(min,max,negmin,posmax) == MDC_YES) {
    scale = 1.; min = 0.;
  }

  /* scale pixel values */
  for (pixel=id->buf, i=0; i<n; i++, pixel+=MdcType2Bytes(id->type)) {

     /* get pixel value */
     pixval = MdcGetDoublePixel(pixel,id->type);
     if (DO_QUANT_CALIBR) { 
       pixval *= (double)id->rescale_slope;
       pixval += (double)id->rescale_intercept;
     }

     newval = (float) (scale * (pixval - min));
     buf[i] = (Int16) newval;
  }

  /* preserve rescaled values */
  id->rescaled = MDC_YES;
  if (DO_QUANT_CALIBR) {
    id->rescaled_fctr = (min < 0.) ? 1. : 1./scale;
    id->rescaled_slope= 1./scale;
    id->rescaled_intercept = min;
    idmax = id->qmax; idmin = id->qmin;
  }else{
    id->rescaled_fctr = 1.;
    id->rescaled_slope= 1.;
    id->rescaled_intercept = 0.;
    idmax = id->max;  idmin = id->min;
  }

  id->rescaled_max = (double)((Int16)(scale * (idmax - min)));
  id->rescaled_min = (double)((Int16)(scale * (idmin - min)));

  return((Uint8 *)buf);

}

/* converts to Int16 */
Uint8 *MdcGetImgBIT16_S(FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint32 bytes = id->width * id->height * MdcType2Bytes(BIT16_S);
  Uint8 *buffer;

  if ( (buffer=(Uint8 *)malloc(bytes)) == NULL ) return NULL;

  buffer=MdcMakeBIT16_S(buffer,fi,img);

  return(buffer);

}


Uint8 *MdcMakeBIT32_S(Uint8 *cbuf, FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint8 *pixel, DO_QUANT_CALIBR, DO_LINEAR_SCALE=MDC_NO, BITS = 32;
  Int32 *buf = (Int32 *)cbuf;
  Uint32 i, n = id->width * id->height;
  double pixval, min, max, idmin, idmax, scale=1.0;
  double SMAX, UMAX, negmin, posmax;
  float newval;

  UMAX = (double)(1 <<  BITS);     /* 4294967296 */
  SMAX = (double)(1 << (BITS-1));  /* 2147483648 */

  /* get proper maximum/minimum value */
  if (MDC_QUANTIFY || MDC_CALIBRATE) {
    DO_QUANT_CALIBR = MDC_YES;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->qfmin;
      max = id->qfmax;
    }else{
      min = fi->qglmin;
      max = fi->qglmax;
    }
  }else{
    DO_QUANT_CALIBR = MDC_NO;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->fmin;
      max = id->fmax;
    }else{
      min = fi->glmin;
      max = fi->glmax;
    }
  }

  /* set limit values */
  negmin = -SMAX; posmax = SMAX - 1.;

  /* check scale type: linear / affine */
  if (DO_QUANT_CALIBR) {
    /* get scale to transform max positive value to max new type */
    /* check whether neg values scale within neg range, which    */
    /* allows linear transform, otherwise affine transform used  */
    DO_LINEAR_SCALE = ((min * posmax / max) >= negmin) ? MDC_YES : MDC_NO;
  }

  /* linear scaling, do not shift to positive range */
  if (DO_LINEAR_SCALE == MDC_YES) min = 0.;

  /* set scale value */
  scale = (max == min) ? 1. : posmax / (max - min);

  if (MdcDoSimpleCast(min,max,-SMAX,SMAX-1.) == MDC_YES) {
    scale = 1.; min = 0.;
  }

  /* scale pixel values */
  for (pixel=id->buf, i=0; i<n; i++, pixel+=MdcType2Bytes(id->type)) {

     /* get pixel value */
     pixval = MdcGetDoublePixel(pixel,id->type);
     if (DO_QUANT_CALIBR) {
       pixval *= (double)id->rescale_slope;
       pixval += (double)id->rescale_intercept;
     }

     newval = (float) (scale * (pixval - min));
     buf[i] = (Int32) newval;
  }

  /* preserve rescaled value */
  id->rescaled = MDC_YES;
  if (DO_QUANT_CALIBR) {
    id->rescaled_fctr = ( min < 0. ) ? 1. : 1./scale;
    id->rescaled_slope= 1./scale;
    id->rescaled_intercept = min;
    idmax = id->qmax; idmin = id->qmin;
  }else{
    id->rescaled_fctr = 1.;
    id->rescaled_slope= 1.;
    id->rescaled_intercept = 0.;
    idmax = id->max;  idmin = id->min;
  }

  id->rescaled_max = (double)((Int32)(scale * (idmax - min)));
  id->rescaled_min = (double)((Int32)(scale * (idmin - min)));

  return((Uint8 *)buf);

}

/* converts to Int32 */
Uint8 *MdcGetImgBIT32_S(FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint32 size = id->width * id->height * MdcType2Bytes(BIT32_S); 
  Uint8 *buffer;

  if ( (buffer=(Uint8 *)malloc(size)) == NULL ) return NULL;

  buffer=MdcMakeBIT32_S(buffer,fi,img);

  return(buffer);

}

Uint8 *MdcMakeFLT32(Uint8 *cbuf, FILEINFO *fi, Uint32 img)
{

  IMG_DATA *id = &fi->image[img];
  Uint8 *pixel, DO_QUANT_CALIBR, DO_CAST=MDC_NO;
  float *buf = (float *)cbuf, newval;
  Uint32 i, n = id->width * id->height;
  double pixval, min, max, scale=1.0;
  double smin = 0.; /* shift to positive values (rescale) */


  /* get proper maximum/minimum value */
  if (MDC_QUANTIFY || MDC_CALIBRATE) {
    /* do the real quantification */
    DO_QUANT_CALIBR = MDC_YES; 
    min = id->qmin;
    max = id->qmax;            

    if (id->type == FLT64) { 
      /* probably be too big for float. if global too */
      /* big, don't quantify an image but do a simple */
      /* downscaling to float and warn the user!      */ 
      if (fi->qglmax > 3.40282347e+38) {
        MdcPrntWarn("Values `double' too big for `quantified float'");
        DO_QUANT_CALIBR = MDC_NO;
        if (MDC_NORM_OVER_FRAMES) {
          min = id->fmin;
          max = id->fmax;
        }else{
          min = fi->glmin;
          max = fi->glmax;
        }
      }
    }
  }else{
    DO_QUANT_CALIBR = MDC_NO;
    if (MDC_NORM_OVER_FRAMES) {
      min = id->fmin;
      max = id->fmax;
    }else{
      min = fi->glmin;
      max = fi->glmax;
    }
  }

  if (DO_QUANT_CALIBR) {
    scale = (double)id->rescale_slope; /* anything else fits in float */
  }else{
    /* try preserving pixel values with simple cast */
    if (id->type <= FLT32 ) { 
      scale = 1.; DO_CAST = MDC_YES; /* ok, integers */
    }else if ( id->type == FLT64 && fabs(fi->glmax) < 3.40282347e+38 
                                 && fabs(fi->glmin) > 1e-37 ) {
      scale = 1.; DO_CAST = MDC_YES; /* ok, doubles fit in float */
    }else{ /* need rescaling: 0 -> MAX_FLOAT*/
      scale = (max == min) ? 1. : 3.40282347e+38 / (max - min);
      smin = min; min = 0.; DO_CAST = MDC_NO;
    }
  }

  for (pixel=id->buf, i=0; i<n; i++, pixel+=MdcType2Bytes(id->type)) {

     pixval = MdcGetDoublePixel(pixel,id->type);
     newval = (float) (scale * (pixval - smin));
     if (DO_QUANT_CALIBR) newval += id->rescale_intercept;
     buf[i] = newval; 
  }

  id->rescaled = MDC_YES;
  if (DO_QUANT_CALIBR) {
    id->rescaled_fctr = 1.; /* got the real quantified values this time! */
    id->rescaled_slope= 1.;
    id->rescaled_intercept = 0.;
    id->rescaled_max  = max; 
    id->rescaled_min  = min;
  }else if (DO_CAST == MDC_NO) {
      id->rescaled_fctr = 1.;
      id->rescaled_slope= 1.;
      id->rescaled_intercept = 0.;
      id->rescaled_max  = 3.40282347e+38;
      id->rescaled_min  = 0.;
  }else{
      id->rescaled = MDC_NO;
  }

  return((Uint8 *)buf);

}
  


/* converts from FLT64 to FLT32                              */
/* or in case of quantification all other types into a FLT32 */
Uint8 *MdcGetImgFLT32(FILEINFO *fi, Uint32 img) 
{
  IMG_DATA *id = &fi->image[img];
  Uint32 bytes  = id->width * id->height * MdcType2Bytes(FLT32);
  Uint8 *buffer = NULL;

  if ( (buffer=(Uint8 *)malloc(bytes)) == NULL ) return NULL;

  buffer=MdcMakeFLT32(buffer,fi,img);

  if (buffer == NULL) return NULL;

  return(buffer);

}

Uint8 *MdcMakeImgSwapped(Uint8 *cbuf, FILEINFO *fi, Uint32 img, 
                         Uint32 width, Uint32 height, int type)
{
  IMG_DATA *id = &fi->image[img];
  Uint8 *pixel=NULL;
  int i, pixbytes;

  /* giving a width, heigth & type, allows to use this function directly */
  /* for swapping of none IMG_DATA image buffers                         */

  if ((type == BIT8_U) || (type == BIT8_S)) return(cbuf); /* no swap needed */

  if (width  == 0) width  = id->width;
  if (height == 0) height = id->height;
  if (type   <= 0) type   = id->type;

  pixbytes = MdcType2Bytes(type);

  for (i=0; i<width*height; i++) {
    pixel = cbuf + (i * pixbytes);
    MdcForceSwap(pixel,pixbytes);
  }

  return(cbuf); 

}


Uint8 *MdcGetImgSwapped(FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  Uint32 bytes  = id->width * id->height * MdcType2Bytes(id->type);
  Uint8 *buffer = NULL;

  if ( (buffer=(Uint8 *)malloc(bytes)) == NULL ) return NULL;

  memcpy(buffer,id->buf,bytes);

  buffer=MdcMakeImgSwapped(buffer,fi,img,0,0,0);

  return(buffer);

}

/* unpack BIT12_U pixels into BIT16_U */
/* 2 pix 12bit =     [0xABCDEF]       */
/* 2 pix 16bit = [0x0ABD] + [0x0FCE]  */
int MdcUnpackBIT12(FILEINFO *fi, Uint32 img)
{
   IMG_DATA *id = &fi->image[img];
   Uint32 p, pixels = id->width * id->height;
   Uint16 *buf16 = NULL;
   Uint8 *buf = id->buf, b0, b1, b2;

   if ( (buf16=(Uint16 *)malloc(pixels * sizeof(Uint16))) == NULL)
     return(MDC_NO);

   for (p=0; p<pixels; p+=2) {
      b0 = buf[0], b1 = buf[1], b2 = buf[2];
      buf16[p  ] = ((b0 >> 4) << 8) + ((b0 & 0x0f) << 4) + (b1 & 0x0f);
                      /* A */          /* B */            /* D */
      MdcSwapBytes((Uint8 *)&buf16[p],2);

      buf16[p+1] = ((b2 & 0x0f) << 8) + ((b1 >> 4) << 4) + (b2 >> 4);
                      /* F */          /* C */            /* E */
      MdcSwapBytes((Uint8 *)&buf16[p+1],2);

      buf+=3;
   }

   MdcFree(id->buf); id->buf = (Uint8 *)buf16;

   id->bits = 12; id->type = BIT16_U;
 
   return(MDC_YES);

}

Uint32 MdcHashDJB2(unsigned char *str)
{
  Uint32 hash = 5381;
  int c;

  while ((c = *str++)) {
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  }

  return(hash);

}

Uint32 MdcHashSDBM(unsigned char *str)
{
  unsigned long hash = 0;
  int c;
    
  while ((c = *str++)) {
    hash = c + (hash << 6) + (hash << 16) - hash;
  }
 
  return hash;

}
