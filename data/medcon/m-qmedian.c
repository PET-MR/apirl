/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-qmedian.c                                                   *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : do color reduction from RGB to 8 bit (median cut/dither) *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcReduceColor() - RGB to indexed for all FI images      *
 *                MdcRgb2Indexed() - RBG to indexed for one image buffer   *
 *                                                                         *
 * Notes        : routines addapted from 'tiffmedian.c' found in libtiff   *
 *                                                                         *
 *                see also http://www.libtiff.org/                         *
 *                                                                         *
 * Original Copyright Notice:                                              *
 *                                                                         *
 * Copyright (c) 1988-1997 Sam Leffler                                     *
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.                          *
 *                                                                         *
 * Permission to use, copy, modify, distribute, and sell this software and *
 * its documentation for any purpose is hereby granted without fee,        * 
 * provided that (i) the above copyright notices and this permission notice*
 * appear in all copies of the software and related documentation, and     *
 * (ii) the names of Sam Leffler and Silicon Graphics may not be used in   *
 * any advertising or publicity relating to the software without the       *
 * specific, prior written permission of Sam Leffler and Silicon Graphics. *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-qmedian.c,v 1.19 2010/08/28 23:44:23 enlf Exp $
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

/*
 * Notes:
 *
 * [1] Floyd-Steinberg dither:
 *  I should point out that the actual fractions we used were, assuming
 *  you are at X, moving left to right:
 *
 *                  X     7/16
 *           3/16   5/16  1/16    
 *
 *  Note that the error goes to four neighbors, not three.  I think this
 *  will probably do better (at least for black and white) than the
 *  3/8-3/8-1/4 distribution, at the cost of greater processing.  I have
 *  seen the 3/8-3/8-1/4 distribution described as "our" algorithm before,
 *  but I have no idea who the credit really belongs to.

 *  Also, I should add that if you do zig-zag scanning (see my immediately
 *  previous message), it is sufficient (but not quite as good) to send
 *  half the error one pixel ahead (e.g. to the right on lines you scan
 *  left to right), and half one pixel straight down.  Again, this is for
 *  black and white;  I've not tried it with color.
 *  -- 
 *                                          Lou Steinberg
 *
 * [2] Color Image Quantization for Frame Buffer Display, Paul Heckbert,
 *     SIGGRAPH '82 proceedings, pp. 297-307
 */

#define  MAX_CMAP_SIZE  256

#define  COLOR_DEPTH  8
#define  MAX_COLOR  256

#define  B_DEPTH    5    /* # bits/pixel to use */
#define  B_LEN    (1L<<B_DEPTH)

#define  C_DEPTH    2
#define  C_LEN    (1L<<C_DEPTH)  /* # cells/color to use */

#define  COLOR_SHIFT  (COLOR_DEPTH-B_DEPTH)

typedef  struct colorbox {
  struct  colorbox *next, *prev;
  int  rmin, rmax;
  int  gmin, gmax;
  int  bmin, bmax;
  int  total;
} Colorbox;

typedef struct {
  int  num_ents;
  int  entries[MAX_CMAP_SIZE][2];
} C_cell;

Uint16  rm[MAX_CMAP_SIZE], gm[MAX_CMAP_SIZE], bm[MAX_CMAP_SIZE];
int  bytes_per_pixel;
int  num_colors;
int  histogram[B_LEN][B_LEN][B_LEN];
Colorbox *freeboxes;
Colorbox *usedboxes;
C_cell  **ColorCells;
Uint32  rowsperstrip = (Uint32) -1;
Uint16  compression = (Uint16) -1;
Uint16  bitspersample = 1;
Uint16  samplesperpixel;
Uint32  imagewidth;
Uint32  imagelength;
Uint16  predictor = 0;

static  void get_histogram(Uint8 *pRGB, Colorbox*, Uint32 n);
static  void splitbox(Colorbox*);
static  void shrinkbox(Colorbox*);
static  char *map_colortable(void);
static  char *quant(Uint8 *srcRGB, Uint8 *dest8);
static  char *quant_fsdither(Uint8 *srcRGB, Uint8 *dest8);
static  Colorbox* largest_box(void);

/****************************************************************************
                             F U N C T I O N S
 ****************************************************************************/

char *MdcReduceColor(FILEINFO *fi)
{
  IMG_DATA *id;
  Uint32 n;
  int i;
  Colorbox *box_list, *ptr;
  Uint8 *dest8;
  char *msg; 


  if (fi->diff_type == MDC_YES) 
    return("Reduce color unsupported for different types");

  if (fi->diff_size == MDC_YES)
    return("Reduce color unsupported for different sizes");

  if (fi->type != COLRGB) return(NULL);

  /*
   * STEP 0: initialize some values
   */
  num_colors = MAX_CMAP_SIZE;

  imagewidth  = fi->mwidth;
  imagelength = fi->mheight;

  for (i=0; i<MAX_CMAP_SIZE; i++) {
     rm[i]=0; bm[i]=0; gm[i]=0;
  }

  /*
   * STEP 1:  create empty boxes
   */
  usedboxes = NULL;
  box_list = freeboxes = (Colorbox *)malloc(num_colors*sizeof(Colorbox));
  if (box_list == NULL) return("Unable to malloc box_list");
  freeboxes[0].next = &freeboxes[1];
  freeboxes[0].prev = NULL;
  for (i = 1; i < num_colors-1; ++i) {
    freeboxes[i].next = &freeboxes[i+1];
    freeboxes[i].prev = &freeboxes[i-1];
  }
  freeboxes[num_colors-1].next = NULL;
  freeboxes[num_colors-1].prev = &freeboxes[num_colors-2];

  /*
   * STEP 2: get histogram, initialize first box
   */
  ptr = freeboxes;
  freeboxes = ptr->next;
  if (freeboxes)
    freeboxes->prev = NULL;
  ptr->next = usedboxes;
  usedboxes = ptr;
  if (ptr->next)
    ptr->next->prev = ptr;

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Reducing colors: ");

  for (n=0; n<fi->number; n++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,.5/(float)fi->number,NULL);

     get_histogram(fi->image[n].buf, ptr, n);

  }

  /*
   * STEP 3: continually subdivide boxes until no more free
   * boxes remain or until all colors assigned.
   */
  while (freeboxes != NULL) {
    ptr = largest_box();
    if (ptr != NULL)
      splitbox(ptr);
    else
      freeboxes = NULL;
  }

  /*
   * STEP 4: assign colors to all boxes
   */
  for (i = 0, ptr = usedboxes; ptr != NULL; ++i, ptr = ptr->next) {
    rm[i] = ((ptr->rmin + ptr->rmax) << COLOR_SHIFT) / 2;
    gm[i] = ((ptr->gmin + ptr->gmax) << COLOR_SHIFT) / 2;
    bm[i] = ((ptr->bmin + ptr->bmax) << COLOR_SHIFT) / 2;
  }

  /* We're done with the boxes now */
  MdcFree(box_list);
  freeboxes = usedboxes = NULL;

  /*
   * STEP 5: scan histogram and map all values to closest color
   */
  /* 5a: create cell list as described in Heckbert[2] */
  ColorCells = (C_cell **)malloc(C_LEN*C_LEN*C_LEN*sizeof(C_cell*));
  if (ColorCells == NULL) return("Unable to malloc ColorCells");
  memset(ColorCells, 0, C_LEN*C_LEN*C_LEN*sizeof(C_cell*));
  /* 5b: create mapping from truncated pixel space to color
     table entries */
  msg = map_colortable();
  if (msg != NULL) {
    MdcFree(ColorCells);
    return(msg);
  }

  /*
   * STEP 6: scan image, match input values to table entries
   */
  for (n=0; n<fi->number; n++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,.5/(float)fi->number,NULL);

     id = &fi->image[n];

     dest8 = (Uint8 *)malloc(id->width * id->height);
     if (dest8 == NULL) return("Unable to malloc indexed buffer");

    if (MDC_DITHER_COLOR == MDC_YES)
      msg = quant_fsdither(id->buf, dest8);
    else
      msg = quant(id->buf,dest8);

    if (msg != NULL) return(msg);

    MdcFree(id->buf);

    id->buf = dest8;
    id->type = BIT8_U; id->bits = 8;

  }

  fi->map = MDC_MAP_PRESENT;
  fi->type = BIT8_U; fi->bits = 8;

  /*
   * copy reduced colormap
   */

  for (i = 0; i < MAX_CMAP_SIZE; ++i) {
    fi->palette[i*3 + 0] = rm[i];
    fi->palette[i*3 + 1] = gm[i];
    fi->palette[i*3 + 2] = bm[i];
  }

  return (NULL);

}

char *MdcRgb2Indexed(Uint8 *srcRGB, Uint8 *dest8, Uint32 width, Uint32 height, Uint8 *palette, int dither)
{
  int i;
  Colorbox *box_list, *ptr;
  char *msg; 

  /*
   * STEP 0: initialize some values
   */
  num_colors = MAX_CMAP_SIZE;

  imagewidth  = width;
  imagelength = height;

  for (i=0; i<MAX_CMAP_SIZE; i++) {
     rm[i]=0; bm[i]=0; gm[i]=0;
  }

  /*
   * STEP 1:  create empty boxes
   */
  usedboxes = NULL;
  box_list = freeboxes = (Colorbox *)malloc(num_colors*sizeof(Colorbox));
  if (box_list == NULL) return("Unable to malloc box_list");
  freeboxes[0].next = &freeboxes[1];
  freeboxes[0].prev = NULL;
  for (i = 1; i < num_colors-1; ++i) {
    freeboxes[i].next = &freeboxes[i+1];
    freeboxes[i].prev = &freeboxes[i-1];
  }
  freeboxes[num_colors-1].next = NULL;
  freeboxes[num_colors-1].prev = &freeboxes[num_colors-2];

  /*
   * STEP 2: get histogram, initialize first box
   */
  ptr = freeboxes;
  freeboxes = ptr->next;
  if (freeboxes)
    freeboxes->prev = NULL;
  ptr->next = usedboxes;
  usedboxes = ptr;
  if (ptr->next)
    ptr->next->prev = ptr;
  get_histogram(srcRGB, ptr, 0);

  /*
   * STEP 3: continually subdivide boxes until no more free
   * boxes remain or until all colors assigned.
   */
  while (freeboxes != NULL) {
    ptr = largest_box();
    if (ptr != NULL)
      splitbox(ptr);
    else
      freeboxes = NULL;
  }

  /*
   * STEP 4: assign colors to all boxes
   */
  for (i = 0, ptr = usedboxes; ptr != NULL; ++i, ptr = ptr->next) {
    rm[i] = ((ptr->rmin + ptr->rmax) << COLOR_SHIFT) / 2;
    gm[i] = ((ptr->gmin + ptr->gmax) << COLOR_SHIFT) / 2;
    bm[i] = ((ptr->bmin + ptr->bmax) << COLOR_SHIFT) / 2;
  }

  /* We're done with the boxes now */
  MdcFree(box_list);
  freeboxes = usedboxes = NULL;

  /*
   * STEP 5: scan histogram and map all values to closest color
   */
  /* 5a: create cell list as described in Heckbert[2] */
  ColorCells = (C_cell **)malloc(C_LEN*C_LEN*C_LEN*sizeof(C_cell*));
  if (ColorCells == NULL) return("Unable to malloc ColorCells");
  memset(ColorCells, 0, C_LEN*C_LEN*C_LEN*sizeof(C_cell*));
  /* 5b: create mapping from truncated pixel space to color
     table entries */
  msg = map_colortable();
  if (msg != NULL) {
    MdcFree(ColorCells);
    return(msg);
  }

  /*
   * STEP 6: scan image, match input values to table entries
   */
  if (dither)
    msg = quant_fsdither(srcRGB,dest8);
  else
    msg = quant(srcRGB,dest8);

  /*
   * copy reduced colormap
   */

  for (i = 0; i < MAX_CMAP_SIZE; ++i) {
    palette[i*3 + 0] = rm[i];
    palette[i*3 + 1] = gm[i];
    palette[i*3 + 2] = bm[i];
  }

  return (msg);

}

static void get_histogram(Uint8 *pRGB, Colorbox* box, Uint32 n)
{
  register Uint8 *inptr;
  register int red, green, blue;
  register Uint32 j, i;
  Uint8 *inputline;


  /* init at first image only */
  if (n == 0) {
    register int *ptr = &histogram[0][0][0];

    for (i = B_LEN*B_LEN*B_LEN; i-- > 0;) *ptr++ = 0;

    box->rmin = box->gmin = box->bmin = 999;
    box->rmax = box->gmax = box->bmax = -1;
    box->total = imagewidth * imagelength;

  }

  for (i = 0; i < imagelength; i++) {
     inputline = &pRGB[i*imagewidth*3];
     inptr = inputline;
     for (j = imagewidth; j-- > 0;) {
        red   = *inptr++ >> COLOR_SHIFT;
        green = *inptr++ >> COLOR_SHIFT;
        blue  = *inptr++ >> COLOR_SHIFT;
        if (red < box->rmin)
          box->rmin = red;
        if (red > box->rmax)
          box->rmax = red;
        if (green < box->gmin)
          box->gmin = green;
        if (green > box->gmax)
          box->gmax = green;
        if (blue < box->bmin)
          box->bmin = blue;
        if (blue > box->bmax)
          box->bmax = blue;
        histogram[red][green][blue]++;
    }
  }
}

static Colorbox *largest_box(void)
{
  register Colorbox *p, *b;
  register int size;

  b = NULL;
  size = -1;
  for (p = usedboxes; p != NULL; p = p->next)
    if ((p->rmax > p->rmin || p->gmax > p->gmin ||
        p->bmax > p->bmin) &&  p->total > size)
            size = (b = p)->total;
  return (b);
}

static void splitbox(Colorbox* ptr)
{
  int    hist2[B_LEN];
  int    first=0, last=0;
  register Colorbox  *new;
  register int  *iptr, *histp;
  register int  i, j;
  register int  ir,ig,ib;
  register int sum, sum1, sum2;
  enum { RED, GREEN, BLUE } axis;

  /*
   * See which axis is the largest, do a histogram along that
   * axis.  Split at median point.  Contract both new boxes to
   * fit points and return
   */
  i = ptr->rmax - ptr->rmin;
  if (i >= ptr->gmax - ptr->gmin  && i >= ptr->bmax - ptr->bmin)
    axis = RED;
  else if (ptr->gmax - ptr->gmin >= ptr->bmax - ptr->bmin)
    axis = GREEN;
  else
    axis = BLUE;
  /* get histogram along longest axis */
  switch (axis) {
    case RED:
        histp = &hist2[ptr->rmin];
        for (ir = ptr->rmin; ir <= ptr->rmax; ++ir) {
           *histp = 0;
           for (ig = ptr->gmin; ig <= ptr->gmax; ++ig) {
              iptr = &histogram[ir][ig][ptr->bmin];
              for (ib = ptr->bmin; ib <= ptr->bmax; ++ib)
                 *histp += *iptr++;
           }
           histp++;
        }
        first = ptr->rmin;
        last = ptr->rmax;
        break;
    case GREEN:
        histp = &hist2[ptr->gmin];
        for (ig = ptr->gmin; ig <= ptr->gmax; ++ig) {
           *histp = 0;
           for (ir = ptr->rmin; ir <= ptr->rmax; ++ir) {
              iptr = &histogram[ir][ig][ptr->bmin];
              for (ib = ptr->bmin; ib <= ptr->bmax; ++ib)
                 *histp += *iptr++;
           }
           histp++;
        }
        first = ptr->gmin;
        last = ptr->gmax;
        break;
    case BLUE:
        histp = &hist2[ptr->bmin];
        for (ib = ptr->bmin; ib <= ptr->bmax; ++ib) {
           *histp = 0;
           for (ir = ptr->rmin; ir <= ptr->rmax; ++ir) {
              iptr = &histogram[ir][ptr->gmin][ib];
              for (ig = ptr->gmin; ig <= ptr->gmax; ++ig) {
                 *histp += *iptr;
                 iptr += B_LEN;
              }
           }
           histp++;
        }
        first = ptr->bmin;
        last = ptr->bmax;
        break;
  }
  /* find median point */
  sum2 = ptr->total / 2;
  histp = &hist2[first];
  sum = 0;
  for (i = first; i <= last && (sum += *histp++) < sum2; ++i)
     ;
  if (i == first)
    i++;

  /* Create new box, re-allocate points */
  new = freeboxes;
  freeboxes = new->next;
  if (freeboxes)
    freeboxes->prev = NULL;
  if (usedboxes)
    usedboxes->prev = new;
  new->next = usedboxes;
  usedboxes = new;

  histp = &hist2[first];
  for (sum1 = 0, j = first; j < i; j++)
     sum1 += *histp++;
  for (sum2 = 0, j = i; j <= last; j++)
      sum2 += *histp++;
  new->total = sum1;
  ptr->total = sum2;

  new->rmin = ptr->rmin;
  new->rmax = ptr->rmax;
  new->gmin = ptr->gmin;
  new->gmax = ptr->gmax;
  new->bmin = ptr->bmin;
  new->bmax = ptr->bmax;
  switch (axis) {
    case RED:
        new->rmax = i-1;
        ptr->rmin = i;
        break;
    case GREEN:
        new->gmax = i-1;
        ptr->gmin = i;
        break;
    case BLUE:
        new->bmax = i-1;
        ptr->bmin = i;
        break;
  }
  shrinkbox(new);
  shrinkbox(ptr);
}

static void shrinkbox(Colorbox* box)
{
  register int *histp, ir, ig, ib;

  if (box->rmax > box->rmin) {
    for (ir = box->rmin; ir <= box->rmax; ++ir)
       for (ig = box->gmin; ig <= box->gmax; ++ig) {
          histp = &histogram[ir][ig][box->bmin];
          for (ib = box->bmin; ib <= box->bmax; ++ib)
             if (*histp++ != 0) {
               box->rmin = ir;
               goto have_rmin;
             }
       }
  have_rmin:
    if (box->rmax > box->rmin)
      for (ir = box->rmax; ir >= box->rmin; --ir)
        for (ig = box->gmin; ig <= box->gmax; ++ig) {
          histp = &histogram[ir][ig][box->bmin];
          ib = box->bmin;
          for (; ib <= box->bmax; ++ib)
            if (*histp++ != 0) {
              box->rmax = ir;
              goto have_rmax;
            }
              }
  }
have_rmax:
  if (box->gmax > box->gmin) {
    for (ig = box->gmin; ig <= box->gmax; ++ig)
      for (ir = box->rmin; ir <= box->rmax; ++ir) {
        histp = &histogram[ir][ig][box->bmin];
              for (ib = box->bmin; ib <= box->bmax; ++ib)
        if (*histp++ != 0) {
          box->gmin = ig;
          goto have_gmin;
        }
      }
  have_gmin:
    if (box->gmax > box->gmin)
      for (ig = box->gmax; ig >= box->gmin; --ig)
        for (ir = box->rmin; ir <= box->rmax; ++ir) {
          histp = &histogram[ir][ig][box->bmin];
          ib = box->bmin;
          for (; ib <= box->bmax; ++ib)
            if (*histp++ != 0) {
              box->gmax = ig;
              goto have_gmax;
            }
              }
  }
have_gmax:
  if (box->bmax > box->bmin) {
    for (ib = box->bmin; ib <= box->bmax; ++ib)
      for (ir = box->rmin; ir <= box->rmax; ++ir) {
        histp = &histogram[ir][box->gmin][ib];
              for (ig = box->gmin; ig <= box->gmax; ++ig) {
          if (*histp != 0) {
            box->bmin = ib;
            goto have_bmin;
          }
          histp += B_LEN;
              }
            }
  have_bmin:
    if (box->bmax > box->bmin)
      for (ib = box->bmax; ib >= box->bmin; --ib)
        for (ir = box->rmin; ir <= box->rmax; ++ir) {
          histp = &histogram[ir][box->gmin][ib];
          ig = box->gmin;
          for (; ig <= box->gmax; ++ig) {
            if (*histp != 0) {
              box->bmax = ib;
              goto have_bmax;
            }
            histp += B_LEN;
          }
              }
  }
have_bmax:
  ;
}

static C_cell *create_colorcell(int red, int green, int blue)
{
  register int ir, ig, ib, i;
  register C_cell *ptr;
  int mindist, next_n;
  register int tmp, dist, n;

  ir = red >> (COLOR_DEPTH-C_DEPTH);
  ig = green >> (COLOR_DEPTH-C_DEPTH);
  ib = blue >> (COLOR_DEPTH-C_DEPTH);
  ptr = (C_cell *)malloc(sizeof (C_cell));
  if (ptr == NULL) return(NULL);
  *(ColorCells + ir*C_LEN*C_LEN + ig*C_LEN + ib) = ptr;
  ptr->num_ents = 0;

  /*
   * Step 1: find all colors inside this cell, while we're at
   *     it, find distance of centermost point to furthest corner
   */
  mindist = 99999999;
  for (i = 0; i < num_colors; ++i) {
    if (rm[i]>>(COLOR_DEPTH-C_DEPTH) != ir  ||
        gm[i]>>(COLOR_DEPTH-C_DEPTH) != ig  ||
        bm[i]>>(COLOR_DEPTH-C_DEPTH) != ib)
      continue;
    ptr->entries[ptr->num_ents][0] = i;
    ptr->entries[ptr->num_ents][1] = 0;
    ++ptr->num_ents;
          tmp = rm[i] - red;
          if (tmp < (MAX_COLOR/C_LEN/2))
      tmp = MAX_COLOR/C_LEN-1 - tmp;
          dist = tmp*tmp;
          tmp = gm[i] - green;
          if (tmp < (MAX_COLOR/C_LEN/2))
      tmp = MAX_COLOR/C_LEN-1 - tmp;
          dist += tmp*tmp;
          tmp = bm[i] - blue;
          if (tmp < (MAX_COLOR/C_LEN/2))
      tmp = MAX_COLOR/C_LEN-1 - tmp;
          dist += tmp*tmp;
          if (dist < mindist)
      mindist = dist;
  }

  /*
   * Step 3: find all points within that distance to cell.
   */
  for (i = 0; i < num_colors; ++i) {
    if (rm[i] >> (COLOR_DEPTH-C_DEPTH) == ir  &&
        gm[i] >> (COLOR_DEPTH-C_DEPTH) == ig  &&
        bm[i] >> (COLOR_DEPTH-C_DEPTH) == ib)
      continue;
    dist = 0;
          if ((tmp = red - rm[i]) > 0 ||
        (tmp = rm[i] - (red + MAX_COLOR/C_LEN-1)) > 0 )
      dist += tmp*tmp;
          if ((tmp = green - gm[i]) > 0 ||
        (tmp = gm[i] - (green + MAX_COLOR/C_LEN-1)) > 0 )
      dist += tmp*tmp;
          if ((tmp = blue - bm[i]) > 0 ||
        (tmp = bm[i] - (blue + MAX_COLOR/C_LEN-1)) > 0 )
      dist += tmp*tmp;
          if (dist < mindist) {
      ptr->entries[ptr->num_ents][0] = i;
      ptr->entries[ptr->num_ents][1] = dist;
      ++ptr->num_ents;
          }
  }

  /*
   * Sort color cells by distance, use cheap exchange sort
   */
  for (n = ptr->num_ents - 1; n > 0; n = next_n) {
    next_n = 0;
    for (i = 0; i < n; ++i)
      if (ptr->entries[i][1] > ptr->entries[i+1][1]) {
        tmp = ptr->entries[i][0];
        ptr->entries[i][0] = ptr->entries[i+1][0];
        ptr->entries[i+1][0] = tmp;
        tmp = ptr->entries[i][1];
        ptr->entries[i][1] = ptr->entries[i+1][1];
        ptr->entries[i+1][1] = tmp;
        next_n = i;
            }
  }
  return (ptr);
}

static char *map_colortable(void)
{
  register int *histp = &histogram[0][0][0];
  register C_cell *cell;
  register int j, tmp, d2, dist;
  int ir, ig, ib, i;

  for (ir = 0; ir < B_LEN; ++ir)
    for (ig = 0; ig < B_LEN; ++ig)
      for (ib = 0; ib < B_LEN; ++ib, histp++) {
        if (*histp == 0) {
          *histp = -1;
          continue;
        }
        cell = *(ColorCells +
            (((ir>>(B_DEPTH-C_DEPTH)) << C_DEPTH*2) +
            ((ig>>(B_DEPTH-C_DEPTH)) << C_DEPTH) +
            (ib>>(B_DEPTH-C_DEPTH))));
        if (cell == NULL )
          cell = create_colorcell(
              ir << COLOR_SHIFT,
              ig << COLOR_SHIFT,
              ib << COLOR_SHIFT);
        if (cell == NULL) return("Unable to malloc colorcell");
        dist = 9999999;
        for (i = 0; i < cell->num_ents &&
            dist > cell->entries[i][1]; ++i) {
          j = cell->entries[i][0];
          d2 = rm[j] - (ir << COLOR_SHIFT);
          d2 *= d2;
          tmp = gm[j] - (ig << COLOR_SHIFT);
          d2 += tmp*tmp;
          tmp = bm[j] - (ib << COLOR_SHIFT);
          d2 += tmp*tmp;
          if (d2 < dist) {
            dist = d2;
            *histp = j;
          }
        }
      }

  return(NULL);
}

/*
 * straight quantization.  Each pixel is mapped to the colors
 * closest to it.  Color values are rounded to the nearest color
 * table entry.
 */
static char *quant(Uint8 *srcRGB, Uint8 *dest8)
{
  Uint8 *inputline;
  register Uint8 *outptr, *inptr;
  register Uint32 i, j;
  register int red, green, blue;

  for (i = 0; i < imagelength; i++) {
    inputline = &srcRGB[i*3*imagewidth];
    inptr = inputline;
    outptr = dest8+(i*imagewidth);
    for (j = 0; j < imagewidth; j++) {
      red = *inptr++ >> COLOR_SHIFT;
      green = *inptr++ >> COLOR_SHIFT;
      blue = *inptr++ >> COLOR_SHIFT;
      *outptr++ = (Uint8)histogram[red][green][blue];
    }
  }

  return(NULL);
}

#define  SWAP(type,a,b)  { type p; p = a; a = b; b = p; }

#define  GetComponent(raw, cshift, c)        \
  cshift = raw;            \
  if (cshift < 0)            \
    cshift = 0;          \
  else if (cshift >= MAX_COLOR)        \
    cshift = MAX_COLOR-1;        \
  c = cshift;            \
  cshift >>= COLOR_SHIFT;

static char *quant_fsdither(Uint8 *srcRGB, Uint8 *dest8)
{
  Uint8 *inputline, *inptr;
  short *thisline, *nextline;
  register Uint8 *outptr;
  register short *thisptr, *nextptr;
  register Uint32 i, j;
  Uint32 imax, jmax;
  int lastline, lastpixel;

  imax = imagelength - 1;
  jmax = imagewidth - 1;
  thisline = (short *)malloc(imagewidth * 3 * sizeof (short));
  if (thisline == NULL) return("Unable to malloc thisline");
  nextline = (short *)malloc(imagewidth * 3 * sizeof (short));
  if (nextline == NULL) {
    MdcFree(nextline);
    return("Unable to malloc nextline");
  }

  inputline = srcRGB; inptr = inputline;
  nextptr = nextline;
  for (j = 0; j < imagewidth; ++j) {
     *nextptr++ = *inptr++;
     *nextptr++ = *inptr++;
     *nextptr++ = *inptr++;
  }
  for (i = 1; i < imagelength; ++i) {
    SWAP(short *, thisline, nextline);
    lastline = (i == imax);
    inputline = &srcRGB[i*imagewidth*3]; inptr = inputline;
    nextptr = nextline;
    for (j = 0; j < imagewidth; ++j) {                      
       *nextptr++ = *inptr++;        
       *nextptr++ = *inptr++;           
       *nextptr++ = *inptr++;                          
    }
    thisptr = thisline;
    nextptr = nextline;
    outptr = dest8 + i*imagewidth;
    for (j = 0; j < imagewidth; ++j) {
      int red, green, blue;
      register int oval, r2, g2, b2;

      lastpixel = (j == jmax);
      GetComponent(*thisptr++, r2, red);
      GetComponent(*thisptr++, g2, green);
      GetComponent(*thisptr++, b2, blue);
      oval = histogram[r2][g2][b2];
      if (oval == -1) {
        int ci;
        register int cj, tmp, d2, dist;
        register C_cell  *cell;

        cell = *(ColorCells +
            (((r2>>(B_DEPTH-C_DEPTH)) << C_DEPTH*2) +
            ((g2>>(B_DEPTH-C_DEPTH)) << C_DEPTH ) +
            (b2>>(B_DEPTH-C_DEPTH))));
        if (cell == NULL)
          cell = create_colorcell(red,
              green, blue);
        if (cell == NULL) {
          MdcFree(thisline); MdcFree(nextline);
          return("Unable to malloc colorcell");
        }
        dist = 9999999;
        for (ci = 0; ci < cell->num_ents && dist > cell->entries[ci][1]; ++ci) {
          cj = cell->entries[ci][0];
          d2 = (rm[cj] >> COLOR_SHIFT) - r2;
          d2 *= d2;
          tmp = (gm[cj] >> COLOR_SHIFT) - g2;
          d2 += tmp*tmp;
          tmp = (bm[cj] >> COLOR_SHIFT) - b2;
          d2 += tmp*tmp;
          if (d2 < dist) {
            dist = d2;
            oval = cj;
          }
        }
        histogram[r2][g2][b2] = oval;
      }
      *outptr++ = (Uint8)oval;
      red -= rm[oval];
      green -= gm[oval];
      blue -= bm[oval];
      if (!lastpixel) {
        thisptr[0] += blue * 7 / 16;
        thisptr[1] += green * 7 / 16;
        thisptr[2] += red * 7 / 16;
      }
      if (!lastline) {
        if (j != 0) {
          nextptr[-3] += blue * 3 / 16;
          nextptr[-2] += green * 3 / 16;
          nextptr[-1] += red * 3 / 16;
        }
        nextptr[0] += blue * 5 / 16;
        nextptr[1] += green * 5 / 16;
        nextptr[2] += red * 5 / 16;
        if (!lastpixel) {
          nextptr[3] += blue / 16;
                nextptr[4] += green / 16;
                nextptr[5] += red / 16;
        }
        nextptr += 3;
      }
    }
  }

  MdcFree(thisline);
  MdcFree(nextline);
  return(NULL);
}
