/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-transf.c                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : simple slice transformation routines                     *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcFlipImgHorizontal()     - Flip image horizontally (X) *
 *                MdcFlipImgVertical()       - Flip image vertically   (Y) *
 *                MdcFlipHorizontal()        - Flip all horizontally   (X) *
 *                MdcFlipVertical()          - Flip all vertically     (Y) *
 *                MdcSortReverse()           - Reverse   sorting           * 
 *                MdcSortCineApply()         - Apply cine sorting          *
 *                MdcSortCineUndo()          - Undo  cine sorting          *
 *                MdcMakeSquare()            - Make all square             *
 *                MdcCropImages()            - Crop image dimensions       *
 *                MdcMakeGray()              - Make all gray scale         *
 *                MdcHandleColor()           - Handle color images         *
 *                MdcContrastRemap()         - Apply contrast remapping    *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-transf.c,v 1.36 2010/08/28 23:44:23 enlf Exp $
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
                            F U N C T I O N S
****************************************************************************/

int MdcFlipImgHorizontal(IMG_DATA *id)
{
  Uint8 *pix1, *pix2, *temp;
  Uint32 h, w, bytes;

  bytes = MdcType2Bytes(id->type);
  if ((temp=(Uint8 *)malloc(bytes)) == NULL) return(MDC_BAD_ALLOC);

  for (h=0; h < id->height; h++) {
     pix1 = &id->buf[bytes * (   h   * id->width)];
     pix2 = &id->buf[bytes * (((h+1) * id->width) - 1)];
     for (w=0; w < (id->width/2); w++) {
        memcpy(temp,pix1,bytes);
        memcpy(pix1,pix2,bytes);
        memcpy(pix2,temp,bytes);
        pix1+=bytes; pix2-=bytes;
     }
  }

  MdcFree(temp);

  return(MDC_OK);

}

int MdcFlipImgVertical(IMG_DATA *id)
{
  Uint8 *pix1, *pix2, *temp;
  Uint32 h, w, bytes, linebytes;

  bytes = MdcType2Bytes(id->type);
  if ((temp=(Uint8 *)malloc(bytes)) == NULL) return(MDC_BAD_ALLOC);

  linebytes = bytes * id->width;
  for (w=0; w < linebytes; w+=bytes) {
     pix1 = &id->buf[w];
     pix2 = &id->buf[((id->height - 1) * linebytes) + w];
     for (h=0; h < (id->height/2); h++) {
        memcpy(temp,pix1,bytes);
        memcpy(pix1,pix2,bytes);
        memcpy(pix2,temp,bytes);
        pix1+=linebytes; pix2-=linebytes;
     }
  }

  MdcFree(temp);

  return(MDC_OK);

}

char *MdcFlipHorizontal(FILEINFO *fi)
{
  Uint32 i;
  int err;

  for (i=0; i < fi->number; i++) {
     err = MdcFlipImgHorizontal(&fi->image[i]);
     if (err != MDC_OK) return("FlipH - Couldn't malloc temp pixel");
  }

  return(NULL);

}

char *MdcFlipVertical(FILEINFO *fi)
{
  Uint32 i;
  int err;

  for (i=0; i < fi->number; i++) {
     err = MdcFlipImgVertical(&fi->image[i]);
     if (err != MDC_OK) return("FlipV - Couldn't malloc temp pixel");
  }

  return(NULL);

}

char *MdcSortReverse(FILEINFO *fi)
{
  IMG_DATA *id1, *id2, *tmp;
  Uint32 i, size;

  if (fi->number == 1) return(NULL);

  size = sizeof(IMG_DATA);
  if ((tmp = (IMG_DATA *)malloc(size)) == NULL)
    return("SortRev - Couldn't malloc IMG_DATA tmp");

  for (i=0; i < (fi->number/2); i++) {

     id1 = &fi->image[i];
     id2 = &fi->image[fi->number - (i + 1)];

     memcpy(tmp,id1,size);
     memcpy(id1,id2,size);
     memcpy(id2,tmp,size);

  }

  MdcFree(tmp);

  return(NULL);

}

char *MdcSortCineApply(FILEINFO *fi)
{
  IMG_DATA *tmp;
  Uint32 c, n, i, size;

  if (fi->number == fi->dim[3]) return(NULL);

  size = sizeof(IMG_DATA);
  if ((tmp = (IMG_DATA *)malloc(size * fi->number)) == NULL)
    return("SortCine - Couldn't malloc temporary IMG_DATA array");

  for (c=0,n=0,i=0; i < fi->number; i++, n+=fi->dim[3]) {
     if (n >= fi->number) { c+=1; n = c; }
     memcpy(&tmp[i],&fi->image[n],size);
  }

  for (i=0; i < fi->number; i++) {
     memcpy(&fi->image[i],&tmp[i],size);
  }

  MdcFree(tmp);

  return(NULL);

}

char *MdcSortCineUndo(FILEINFO *fi)
{
  IMG_DATA *tmp;
  Uint32 c, n, i, size;

  if (fi->dim[3] == fi->number) return(NULL);

  size = sizeof(IMG_DATA);
  if ((tmp = (IMG_DATA *)malloc(size * fi->number)) == NULL)
    return("SortNoCine - Couldn't malloc temporary IMG_DATA array");

  for (c=0,n=0,i=0; i < fi->number; i++, n+=fi->dim[3]) {
     if (n >= fi->number) { c+=1; n = c; }
     memcpy(&tmp[n],&fi->image[i],size);
  }

  for (i=0; i < fi->number; i++) {
     memcpy(&fi->image[i],&tmp[i],size);
  }

  MdcFree(tmp);

  return(NULL);

}

char *MdcMakeSquare(FILEINFO *fi, int SQR_TYPE)
{
  IMG_DATA *id;
  Uint32 i, dim;
  Uint8 *sqrbuf;

  /* get largest dim */
  dim = (fi->mwidth > fi->mheight) ? fi->mwidth : fi->mheight;

  /* dims as a power of two */
  if (SQR_TYPE == MDC_TRANSF_SQR2) dim = MdcCeilPwr2(dim);

  /* set to new dimensions */
  fi->mwidth = dim; fi->mheight = dim;
  fi->dim[1] = dim; fi->dim[2]  = dim;

  /* make square images */
  for (i=0; i<fi->number; i++) {
     id = &fi->image[i];
     sqrbuf = MdcGetResizedImage(fi,id->buf,id->type,i);
     if (sqrbuf == NULL) return("Square - Couldn't create squared image");

     id->width  = dim; id->height = dim;
     MdcFree(id->buf);
     id->buf = sqrbuf;
  }

  /* finish settings */ 
  fi->diff_size = MDC_NO;

  return(NULL);
 
}

char *MdcCropImages(FILEINFO *fi, MDC_CROP_INFO *ecrop)
{
  MDC_CROP_INFO icrop, *crop;
  FILEINFO *new, *cur=fi;
  IMG_DATA *newid, *curid;
  Uint8 *curbuf, *newbuf;
  Uint32 i, r, pixelbytes, curlinebytes, newlinebytes, newimgbytes;
  char *msg;

  /* initialize crop settings */
  if (ecrop == NULL ) {
    crop = &icrop;
    crop->xoffset = mdc_crop_xoffset;
    crop->yoffset = mdc_crop_yoffset;
    crop->width   = mdc_crop_width;
    crop->height  = mdc_crop_height;
  }else{
    crop = ecrop;
  }

  /* some sanity checks */
  if ((cur == NULL) || (crop == NULL))
    return(NULL);

  if (cur->diff_size == MDC_YES)
    return("Crop - Different sized slices unsupported");

  if ((crop->width == 0) || (crop->height == 0))
    return("Crop - Improper crop zero values");

  if ((crop->xoffset >= cur->mwidth) || (crop->yoffset >= cur->mheight))
    return("Crop - Improper crop offset values");

  /* cut off */
  if ((crop->xoffset + crop->width)  > cur->mwidth )
    crop->width  = cur->mwidth  - crop->xoffset;
  if ((crop->yoffset + crop->height) > cur->mheight)
    crop->height = cur->mheight - crop->yoffset;

  /* get temporary FILEINFO structure */
  new = (FILEINFO *)malloc(sizeof(FILEINFO));
  if (new == NULL) return("Crop - Bad malloc FILEINFO struct");
  MdcCopyFI(new,cur,MDC_NO,MDC_YES);

  /* set global parameters */
  new->number = cur->number;
  new->mwidth = crop->width;   new->dim[1] = crop->width;
  new->mheight= crop->height;  new->dim[2] = crop->height;

  if (!MdcGetStructID(new,new->number)) {
    MdcCleanUpFI(new); MdcFree(new);
    return("Crop - Bad malloc IMG_DATA structs");
  }

  /* crop image matrices */
  for (i=0; i<new->number; i++) {
     newid = &new->image[i];
     curid = &cur->image[i];

     /* copy all image data */
     MdcCopyID(newid,curid,MDC_YES);

     /* set new dimensions */
     newid->width = crop->width;
     newid->height= crop->height;
     
     /* get some bytes values */
     pixelbytes = MdcType2Bytes(newid->type);

     newlinebytes = pixelbytes   * newid->width;
     newimgbytes  = newlinebytes * newid->height;

     curlinebytes = pixelbytes   * curid->width;

     /* set buffer pointers */
     newbuf = newid->buf; 
     curbuf = curid->buf; /* init and skip */
     curbuf += crop->yoffset*curlinebytes + crop->xoffset*pixelbytes;
     for (r=0; r < newid->height; r++) {
        memcpy(newbuf,curbuf,newlinebytes);
        newbuf += newlinebytes;
        curbuf += curlinebytes;
     }

     /* realloc cropped buffer */
     newid->buf = (Uint8 *)realloc(newid->buf,newimgbytes);
     if (newid->buf == NULL) {
       MdcCleanUpFI(new); MdcFree(new);
       return("Crop - Bad realloc cropped buffer");
     }

  }

  /* check integrity */
  if ((msg = MdcImagesPixelFiddle(new)) != NULL) {
    MdcCleanUpFI(new); MdcFree(new); return(msg);
  }

  /* remove cur */
  MdcCleanUpFI(cur);

  /* copy new -> cur */
  MdcCopyFI(cur,new,MDC_NO,MDC_YES);

  /* just rehang image pointer */
  cur->number = new->number;
  cur->image  = new->image;

  /* and mask new image pointer */
  new->number = 0;
  new->image  = NULL;

  /* now safely remove new */
  MdcCleanUpFI(new); MdcFree(new);

  return(NULL); 

}

char *MdcMakeGray(FILEINFO *fi)
{
  IMG_DATA *id;
  Uint32 i, p, pixels;
  Uint8 *img8, rd=0, gr=0, bl=0, v;


  /* no color file */
  if (fi->map != MDC_MAP_PRESENT) return(NULL);

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.0f,"Grayscaling images: ");

  for (i=0; i<fi->number; i++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1.0f/(float)fi->number,NULL);

     id = &fi->image[i];

     pixels = id->width * id->height;
     img8 = malloc(pixels);
     if (img8 == NULL) return("Couldn't malloc gray buffer");

     for (p=0; p<pixels; p++) {

        if (id->type == COLRGB) {
          /* rgb */
          rd = id->buf[p * 3 + 0];
          gr = id->buf[p * 3 + 1];
          bl = id->buf[p * 3 + 2];
        }else if (id->type == BIT8_U) {
          /* indexed */
          v = id->buf[p];
          rd = fi->palette[v * 3 + 0];
          gr = fi->palette[v * 3 + 1];
          bl = fi->palette[v * 3 + 2];
        }

        img8[p] = (Uint8) MdcGRAY(rd,gr,bl);
     }

     /* free color images */
     MdcFree(id->buf);

     /* replace with gray */
     id->buf = img8;

     id->type = BIT8_U; id->bits = 8;
     
  }

  MdcGetColorMap(MDC_COLOR_MAP,fi->palette);
  fi->map = MDC_COLOR_MAP;
  fi->type = BIT8_U; fi->bits = 8;

  return(NULL);
}

char *MdcHandleColor(FILEINFO *fi)
{
  char *msg=NULL; 

  if (MDC_MAKE_GRAY == MDC_YES) {
    msg = MdcMakeGray(fi);
  }else if (MDC_COLOR_MODE == MDC_COLOR_INDEXED) {
    msg = MdcReduceColor(fi);
  }

  return(msg);
}

/* see also DICOM standards: PS 3.3 - 2001 Page 505 */
char *MdcContrastRemap(FILEINFO *fi)
{
  IMG_DATA *id;
  double wc, ww, rs, ri;
  double xval, yval;
  double ymax, ymin;
  Uint8 *pix;
  Uint32 i, p, n;
  Int16 *newbuf, newtype=BIT16_S, pix16;
  Int16 max=0, min=0, glmax=0, glmin=0;

  /* handle window centre/width */
  if (MDC_FORCE_CONTRAST == MDC_YES) {
    /* apply user specified values */
    wc = (double)mdc_cw_centre;
    ww = (double)mdc_cw_width;
  }else{
    /* apply file specified values */
    wc = (double)fi->window_centre;
    ww = (double)fi->window_width;
  }

  if (ww == 0.) return(NULL);

  for (i=0; i < fi->number; i++) {
     id = &fi->image[i];

     if (id->type == COLRGB) continue;

     newbuf = (Int16 *)malloc(id->width*id->height*MdcType2Bytes(newtype));
     if (newbuf == NULL) return("Couldn't malloc contrast remaped image");

     /* get slope/intercept, even without quantitation */
     rs = (double)id->quant_scale;
     ri = (double)id->intercept;

     /* prevent division by zero */
     if (rs == 0.) rs = 1.;

     /* rescale window towards pixel values */
     wc = (wc - ri) / rs;
     ww = (ww / rs);

     /* prepare range value: [0 -> +max] */
     ymin = 0.;
     ymax = (float)MDC_MAX_BIT16_S;

     n = id->width * id->height;
     for (pix=id->buf, p=0; p < n; p++, pix+=MdcType2Bytes(id->type)) {

        /* get pixel value */
        xval = MdcGetDoublePixel(pix,id->type);

        /* apply window centre/width */
        if ( xval <= wc - 0.5 - ((ww-1.)/2.)) {
          yval = ymin;
        }else if ( xval > wc - 0.5 + ((ww-1.)/2.)) {
          yval = ymax;
        }else{
          yval = ((((xval-(wc-0.5))/(ww-1.))+0.5) * (ymax-ymin)) + ymin;
        }

        /* save in new type */
        pix16 = (Int16) yval;

        /* keep new image max,min */
        if (p == 0) {
          /* init for each image */
          max = pix16; min = pix16;
        }else{
          if (pix16 > max) max = pix16;
          if (pix16 < min) min = pix16;
        }

        /* keep new global max,min */
        if ((i == 0) && (p == 0)) {
          /* init for first image */
          glmax = pix16; glmin = pix16;
        }else{
          if (pix16 > glmax) glmax = pix16;
          if (pix16 < glmin) glmin = pix16;
        }

        /* put value in new buffer */
        newbuf[p] = (Int16)yval;
     }

     /* replace with new image buffer */
     MdcFree(id->buf); id->buf = (Uint8 *)newbuf;

     /* replace image values */
     id->max = id->qmax = max;
     id->min = id->qmin = min;
     id->fmax = id->qfmax = max;
     id->fmin = id->qfmin = min;
     id->rescale_slope = 1.;
     id->rescale_intercept = 1.;
     id->quant_scale = 1.;
     id->calibr_fctr = 1.;
     id->intercept = 0.;
     id->bits = MdcType2Bits(newtype);
     id->type = newtype;
  }

  /* replace global values */
  fi->glmax = fi->qglmax = glmax;
  fi->glmin = fi->qglmin = glmin;
  fi->contrast_remapped = MDC_YES;
  fi->window_centre = 0.;
  fi->window_width  = 0.;
  fi->bits = MdcType2Bits(newtype);
  fi->type = newtype;

  return(NULL);
}
