/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-color.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : make color palettes                                      *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcLoadLUT()       - Load LUT file into RGB array        *
 *                MdcGrayScale()     - Make gray palette                   *
 *                MdcInvertedScale() - Make inverted gray palette          *
 *                MdcRainbowScale()  - Make rainbow palette                *
 *                MdcCombinedScale() - Make combined palette               *
 *                MdcHotmetalScale() - Make hotmetal palette               *
 *                MdcGetColorMap()   - Get the specified palette           *
 *                MdcSetPresentMap() - Preserve colormap in colored file   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-color.c,v 1.19 2010/08/28 23:44:23 enlf Exp $
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
#include <string.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#ifndef _WIN32
#include <strings.h>
#endif
#endif
#include "m-defs.h"
#include "m-global.h"
#include "m-color.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

static Uint8 loaded_map[768], LOADED = MDC_NO;
static Uint8 present_map[768];            /* map from file */

/* cti source */
struct {int n,r,g,b,dr,dg,db; }
  bitty[] = {  {32,0,0,0,2,0,4},          /* violet to indigo */
               {32,64,0,128,-2,0,4},      /* indigo to blue */
               {32,0,0,255,0,8,-8},       /* blue to green */
               {64,0,255,0,4,0,0},        /* green to yellow */
               {32,255,255,0,0,-2,0},     /* yellow to orange */
               {64,255,192,0,0,-3,0} };   /* orange to red */


/****************************************************************************
                            F U N C T I O N S
****************************************************************************/
int  MdcLoadLUT(const char *lutname)
{
  FILE *fp;
  int s;
 
  LOADED = MDC_NO;

  if ((fp=fopen(lutname,"rb")) == NULL) return(MDC_NO);

  LOADED = MDC_YES;

  /* get red values */
  for (s=0; s<768; s+=3) loaded_map[s] = (Uint8)fgetc(fp);
  /* get green values */
  for (s=1; s<768; s+=3) loaded_map[s] = (Uint8)fgetc(fp);
  /* get blue values */
  for (s=2; s<768; s+=3) loaded_map[s] = (Uint8)fgetc(fp);

  fclose(fp);

  return(MDC_YES);
}

void MdcGrayScale(Uint8 *palette)
{
  int i;
  Uint8 gray;
   
  for (i=0; i<256; i++) {
     gray = (Uint8)i;
     palette[i*3]=palette[i*3+1]=palette[i*3+2]=gray;
  }

}
 
void MdcInvertedScale(Uint8 *palette)
{
  int i;
  Uint8 gray;
   

  for (i=0; i<256; i++) {
     gray = 255 - (Uint8)i;
     palette[i*3]=palette[i*3+1]=palette[i*3+2]=gray;
  }

}

void MdcRainbowScale(Uint8 *palette)
{   
  int p=0,i,j,r,g,b;
        
  for (j=0;j<6;j++) {
     palette[p++]=r=bitty[j].r;
     palette[p++]=g=bitty[j].g;
     palette[p++]=b=bitty[j].b;
     for (i=1;i<bitty[j].n;i++) {
        r+=bitty[j].dr; palette[p++]=r;
        g+=bitty[j].dg; palette[p++]=g;
        b+=bitty[j].db; palette[p++]=b; 
     } 
  }
}

void MdcCombinedScale(Uint8 *palette)
{
  int t=0,p=0,i,j,r,g,b;

  /* lower 128 = gray    levels */
  for (i=0; i<256; i+=2) {
     palette[t*3]=palette[t*3+1]=palette[t*3+2]=(Uint8)i; t+=1;
  }

  /* upper 128 = rainbow levels */
  for (j=0;j<6;j++) {
     r=bitty[j].r;
     g=bitty[j].g;
     b=bitty[j].b;
     if (p++ % 2 && p <= 256) {
       palette[t*3]  =(Uint8)r;
       palette[t*3+1]=(Uint8)g;
       palette[t*3+2]=(Uint8)b;
       t+=1;
     }
     for (i=1;i<bitty[j].n;i++) {
        r+=bitty[j].dr;
        g+=bitty[j].dg;
        b+=bitty[j].db;
        if (p++ % 2 && p <= 256) {
          palette[t*3]  =(Uint8)r;
          palette[t*3+1]=(Uint8)g;
          palette[t*3+2]=(Uint8)b;
          t+=1;
        }
     }
  }
}

void MdcHotmetalScale(Uint8 *palette)
{
  int i, p=0;
  float intensity, delta_intensity;
 
  intensity = 0.0;
  delta_intensity = 1.0f/182.0f;
  for (p=0, i=0;i<182;i++,p+=3) {               /* red */
     palette[p]=(Uint8)255*intensity;
     intensity+=delta_intensity;
  }
  for (i=182;i<256;i++,p+=3) palette[p]=255;
  for (i=0,p=1;i<128;i++,p+=3) palette[p]=0;   /* green */
  intensity = 0.0;
  delta_intensity = 1.0f/91.0f;
  for (i=128;i<219; i++,p+=3) {
     palette[p]=(Uint8)255*intensity;
     intensity+=delta_intensity; 
  }
  for (i=219;i<256;i++,p+=3) palette[p]=255;   
  for (i=0,p=2;i<192;i++,p+=3) palette[p]=0;   /* blue */
  intensity=0.0;
  delta_intensity = 1.0/64;
  for (i=192;i<256;i++,p+=3) {
     palette[p]=(Uint8)255*intensity;
     intensity += delta_intensity; 
  }
}

void MdcGetColorMap(int map, Uint8 palette[])
{

  switch (map) {
    case MDC_MAP_PRESENT : memcpy(palette,present_map,768);
                      break;
    case MDC_MAP_GRAY    : MdcGrayScale(palette); 
                      break;
    case MDC_MAP_INVERTED: MdcInvertedScale(palette);
                      break;
    case MDC_MAP_RAINBOW : MdcRainbowScale(palette);
                      break;
    case MDC_MAP_COMBINED: MdcCombinedScale(palette);
                      break;
    case MDC_MAP_HOTMETAL: MdcHotmetalScale(palette);
                      break;
    case MDC_MAP_LOADED  : 
                      if (LOADED == MDC_YES) memcpy(palette,loaded_map,768);
                      break;
    default: MdcGrayScale(palette);
  }

}

int MdcSetPresentMap(Uint8 palette[])
{
  memcpy(present_map,palette,768);

  return(MDC_MAP_PRESENT);
}
