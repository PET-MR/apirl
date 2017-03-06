/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-fancy.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : Nice output, edit strings & print defaults               *
 *                                                                         * 
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcPrintLine()          - Print a full or half line      *
 *                MdcPrintChar()          - Print a char                   *
 *                MdcPrintStr()           - Print a string                 *
 *                MdcPrintBoxLine()       - Print horizontal line of a box *
 *                MdcPrintYesNo()         - Print Yes, No or Unknown       *
 *                MdcPrintImageLayout()   - Print layout of a raw image    *
 *                MdcPrintValue()         - Print numeric value            *
 *                MdcLowStr()             - Make string lower case         *
 *                MdcUpStr()              - Make string upper case         *
 *                MdcKillSpaces()         - Remove first/last spaces       *
 *                MdcRemoveAllSpaces()    - Remove all spaces from string  *
 *                MdcRemoveEnter()        - Remove <enter> from string     *
 *                MdcGetStrLine()         - Get string skipping comment    *
 *                MdcGetStrInput()        - Get string from input with '\n'*
 *                MdcGetSubStr()          - Get substr between separators  *
 *                MdcGetSafeString()      - Copy & add terminating char    *
 *                MdcPutDefault()         - Get (default) answer           *
 *                MdcGetRange()           - Get a range from the a list    *
 *                MdcHandleEcatList()     - Get a list in ecat style       *
 *                MdcHandleNormList()     - Get a list in normal style     *
 *                MdcHandlePixelList()    - Get a list of pixels           *
 *                MdcGetStrAcquisition()  - Get string for acquisition type*
 *                MdcGetStrRawConv()      - Get string of raw type         *
 *                MdcGetStrEndian()       - Get string of endian type      *
 *                MdcGetStrCompression()  - Get string of compression type *
 *                MdcGetStrPixelType()    - Get string of pixel type       *
 *                MdcGetStrColorMap()     - Get string of colormap         *
 *                MdcGetStrYesNo()        - Get string "yes" or "no"       *
 *                MdcGetStrSlProjection() - Get string slice projection    *
 *                MdcGetStrPatSlOrient()  - Get string patient/slice orient*
 *                MdcGetStrPatPos()       - Get string patient position    *
 *                MdcGetStrPatOrient()    - Get string patient orientation *
 *                MdcGetStrSliceOrient()  - Get string slice   orientation *
 *                MdcGetStrRotation()     - Get string rotation direction  *
 *                MdcGetStrMotion()       - Get string detector motion     *
 *                MdcGetStrModality()     - Get string modality            *
 *                MdcGetStrGSpectNesting()- Get string GSPECT nesting      *
 *                MdcGetStrHHMMSS()       - Get string hrs:mins:secs       *
 *                MdcGetIntModality()     - Get int modality type          *
 *                MdcGetIntSliceOrient()  - Get int slice orientation      *
 *                MdcGetLibLongVersion()  - Get string of library version  *
 *                MdcGetLibShortVersion() - Get string of short   version  *
 *                MdcCheckStrSize()       - Check if we can add a string   *
 *                MdcMakeScanInfoStr()    - Make string with scan info     *
 *                MdcIsDigit()            - Test if char is a digit        *
 *                MdcWaitForEnter()       - Wait until <enter> key press   *
 *                MdcGetSelectionType()   - Get select type (norm,ecat,...)*
 *                MdcFlushInput()         - Flush the input stream         *
 *                MdcWhichDecompress()    - Give supported decompression   *
 *                MdcWhichCompression()   - Give compression type of file  *
 *                MdcAddCompressionExt()  - Add  compression extension     *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-fancy.c,v 1.61 2010/08/28 23:44:23 enlf Exp $
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
#include <string.h>
#include <stdlib.h>
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


#include "m-fancy.h"

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

void MdcPrintLine(char c, int length)
{
  int i;

  for (i=0; i<length; i++) MdcPrntScrn("%c",c);
  MdcPrntScrn("\n");

}

void MdcPrintChar(int c)
{
  if (c == '\0') MdcPrntScrn("<null>");
  else if (c==9 || c==13 || c==10) putchar(c);
  else if (c >= 32) putchar(c);
  else if (c==EOF) MdcPrntScrn("<EOF>");
  else MdcPrntScrn("<%u>",c);
}

void MdcPrintStr(char *str)
{
  int t=(int)strlen(str);

  if ( t == 0 ) MdcPrntScrn("<null>");
  else MdcPrntScrn("%s",str);

  MdcPrntScrn("\n");

}

void MdcPrintBoxLine(char c, int t)
{
  int i;

  MdcPrntScrn("\t\t#");
  for (i=-1;i<=t;i++) MdcPrntScrn("%c",c);
  MdcPrntScrn("#\n");
  
}

void MdcPrintYesNo(int value )
{
  switch ( value ) {
    case MDC_NO : MdcPrntScrn("(= No)\n");      break;
    case MDC_YES: MdcPrntScrn("(= Yes)\n");     break;
    default     : MdcPrntScrn("(= Unknown)\n"); break;
  }
}

void MdcPrintImageLayout(FILEINFO *fi, Uint32 gen, Uint32 img
                                     , Uint32 *abs, int repeat)
{
  IMG_DATA *id;
  Uint32 i;

  MdcPrintLine('-',MDC_FULL_LENGTH);
  MdcPrntScrn("\t\t\tSUMMARY OF IMAGE LAYOUT\n");
  MdcPrintLine('-',MDC_FULL_LENGTH);

  if ((gen==0) && (img==0))  MdcPrintBoxLine('-',MDC_BOX_SIZE);
  if (gen!=0) {
    MdcPrintBoxLine('-',MDC_BOX_SIZE);
    MdcPrntScrn("\t\t| General Header   |  \t(%u)\n",gen);
    MdcPrintBoxLine('-',MDC_BOX_SIZE);
  }
  for (i=0; i<fi->number; i++) {
     id = &fi->image[i];
     if ( ((i==0) && (img>0)) || (repeat) ) {
       if ( ! ((i==0) && (gen>0)) )MdcPrintBoxLine('-',MDC_BOX_SIZE); 
       MdcPrntScrn("\t\t| Image   Header   |  \t(%u)\n",img);
       MdcPrintBoxLine('-',MDC_BOX_SIZE);
     }
     MdcPrntScrn("\t\t! Image #%-4u ",i+1);
     if (fi->endian != MDC_HOST_ENDIAN) MdcPrntScrn("swap !"); 
     else MdcPrntScrn("     !");
     MdcPrntScrn("\t(%ux%ux%u)",id->width,id->height,MdcType2Bytes(id->type));
     if (abs!=NULL) if (abs[i] > 0) MdcPrntScrn("\tOFFSET: %u",abs[i]);
     MdcPrntScrn("\n");

  }
  MdcPrintBoxLine('-',MDC_BOX_SIZE);
}

int MdcPrintValue(FILE *fp, Uint8 *pvalue, Uint16 type)
{
  
  switch (type) {
   case BIT8_S:
    {
      Int8 *val = (Int8 *) pvalue;
      fprintf(fp,"%hd",val[0]);
    }
    break;
   case BIT8_U:
    {
      Uint8 *val = (Uint8 *) pvalue;
      fprintf(fp,"%hu",val[0]);
    }
    break;
   case BIT16_S:
    {
      Int16 *val = (Int16 *) pvalue;
      fprintf(fp,"%hd",val[0]);
    }
    break;
   case BIT16_U:
    {
      Uint16 *val = (Uint16 *) pvalue;
      fprintf(fp,"%hu",val[0]);
    }
    break;
   case BIT32_S:
    {
      Int32 *val = (Int32 *) pvalue;
      fprintf(fp,"%d",val[0]);
    }
    break;
   case BIT32_U:
    {
      Uint32 *val = (Uint32 *) pvalue;
      fprintf(fp,"%d",val[0]);
    }
    break;
#ifdef HAVE_8BYTE_INT
   case BIT64_S:
    {
      Int64 *val = (Int64 *) pvalue;
      fprintf(fp,"%lld",val[0]);
    }
    break;
   case BIT64_U:
    {
      Uint64 *val = (Uint64 *) pvalue;
      fprintf(fp,"%llu",val[0]);
    }
    break;
#endif
   case FLT32:
    {
      float *val = (float *) pvalue;
      fprintf(fp,"%+e",val[0]);
    }
    break;
   case FLT64:
    {
      double *val = (double *) pvalue;
      fprintf(fp,"%+e",val[0]);
    }
    break;
  }

  return(ferror(fp));
}

void MdcLowStr(char *str)
{ 
  char *c;

  c=str;

  while(*c) { *c=tolower((int)*c); c++; }

}

void MdcUpStr(char *str)
{
  char *c;

  c=str;

  while(*c) { *c=toupper((int)*c); c++; }

}

void MdcKillSpaces(char string[]) /* kill first and last spaces */
{
  int i=0, shift=0, length;

  length = (int)strlen(string);

  if (length > 0) { 
    /* kill the first spaces */
    while (isspace((int)string[i])) { 
         if (i < length) {
           i+=1; shift+=1; 
         }else break;
    }
    if (shift) for (i=0; i<=length; i++) string[i] = string[i+shift];

    /* kill the last  spaces */
    length = (int)strlen(string);
    if (length > 0) { 
      i = length - 1;
      while (isspace((int)string[i])) { 
           if (i > 0 ) {
             string[i] = '\0'; i-=1;
           }else break;
      }
    }
  } 
}

void MdcRemoveAllSpaces(char string[]) /* remove all spaces */
{
  int i=0, j=0, length;

  length = (int)strlen(string);

  while (i < length) {
       if (isspace((int)string[i])) {
         i+=1;
       }else{
         string[j++] = string[i++];
       }
  }

  string[j]='\0';

}


void MdcRemoveEnter(char string[])
{
  char *p;

  p = strchr(string,'\r'); if (p != NULL) p[0] = '\0';
  p = strchr(string,'\n'); if (p != NULL) p[0] = '\0';

} 

void MdcGetStrLine(char string[], int maxchars, FILE *fp)
{
  /* skip comment lines beginning with '#' */
  do {
    if (fgets(string,maxchars,fp) == NULL) return;
  }while (string[0] == '#');
}

void MdcGetStrInput(char string[], int maxchars)
{ 
  MdcGetStrLine(string,maxchars,stdin);
}

int MdcGetSubStr(char *dest, char *src, int dmax, char sep, int n)
{
  Uint32 i, b, cnt=1, length, sublength=0;

  length = (Uint32)strlen(src);

  if (length == 0) return(MDC_NO);

  /* get begin substr */
  for (b=0; b<length; b++) {
     if (src[b] == sep) cnt+=1;
     if (cnt == n) break;
  }

  /* n-th substr not found */
  if (cnt != n) return(MDC_NO);

  /* get length substr */
  b+=1;
  for (i=b; i<length; i++) {
     if (src[i] == sep) break;
     sublength+=1;
  }

  if ((sublength == 0) || (sublength >= (Uint32)dmax)) return(MDC_NO);

  strncpy(dest,&src[b],sublength);

  dest[sublength] = '\0'; 

  MdcKillSpaces(dest);

  return(MDC_YES);
}
 
void MdcGetSafeString(char *dest, char *src, Uint32 length, Uint32 maximum)
{
   Uint32 MAX = maximum - 1; /* let's be really safe */

   if (length < MAX) {
     memcpy(dest,src,length);
     dest[length]='\0';
   }else{
     memcpy(dest,src,MAX);
     dest[MAX]='\0';
   }
}

int MdcUseDefault(const char string[])
{
  /* <enter> = default */
  if (string[0] == '\n' || string[0] == '\r') return(1);
  return(0);
}
                    /* string[] = MDC_2KB_OFFSET               */
int MdcPutDefault(char string[]) /* 1=default  or 0=no default */
{
  MdcGetStrLine(string,MDC_2KB_OFFSET-1,stdin);
  if (MdcUseDefault(string)) return(1);
  MdcKillSpaces(string);
  return(0);

}

int MdcGetRange(const char *item, Uint32 *from, Uint32 *to, Uint32 *step)
{
  Uint32 a1, a2, t;

  /* read range values */
  if (strchr(item,':') != 0 ) {
    /* interval */
    sscanf(item,"%u:%u:%u",&a1,&t,&a2);
  }else if ( strstr(item,"...") != 0 ) {
    /* range v1 */
    sscanf(item,"%u...%u",&a1,&a2); t=1;
  }else if ( strstr(item,"-") != 0 ) {
    /* range v2 */
    sscanf(item,"%u-%u",&a1,&a2); t=1;
  }else{
    /* single */
    sscanf(item,"%u",&a1); a2=a1;   t=1;
  }

  /* some sanity checks */ 
  if (t == 0) t = 1;

  *from = a1; *to = a2; *step = t;

  return(MDC_OK);
}

char *MdcHandleEcatList(char *list, Uint32 **dims, Uint32 max)
{
  int ITEM_FOUND=MDC_NO, REVERSED, HANDLE;
  Uint32 a1, a2, t, i, l, length;
  char *p, *item;

  length = (Uint32)strlen(list);

  /* <enter> default = all */
  if (MdcUseDefault(list)) {
    for (i=1; i<=max; i++) (*dims)[i]=MDC_YES;
    (*dims)[0]=max;
    return(NULL);
  }

  /* loop through string with entire list */
  for (p=list, item=list, l=0; l<=length; l++) {

     /* separate items: begins at digit, ends at space (or \t, \n, ...) */

     if (ITEM_FOUND == MDC_NO) {

       if (isdigit((int)p[l])) { item=&p[l]; ITEM_FOUND=MDC_YES; }

     }else if (isspace((int)p[l]) || p[l]=='\0') {

       p[l]='\0';

       if (MdcGetRange(item,&a1,&a2,&t) != MDC_OK)
         return("Error reading range item");

       if (a1 > max) a1 = max;
       if (a2 > max) a2 = max;

       if ( (a1==0) || (a2==0) ) {
         for (i=1; i<=max; i++) (*dims)[i]=MDC_YES;
         (*dims)[0]=max;
         break;
       }

       /* reversed range ? */
       REVERSED = (a1 > a2) ? MDC_YES : MDC_NO;

       /* initialize and get image numbers */
       i = a1; HANDLE = MDC_YES;
       do {

         /* include image number */
         if ((*dims)[i] == MDC_NO) {
           (*dims)[i] = MDC_YES; 
           (*dims)[0] += 1;
         }

         if ((REVERSED == MDC_YES) && (i < t)) break;
          
         /* set next image number in range */
         i = (REVERSED == MDC_YES) ? (i-t) : (i+t);

         /* check end of range */
         if (REVERSED == MDC_YES) {
           if (i < a2) HANDLE = MDC_NO;
         }else{
           if (i > a2) HANDLE = MDC_NO;
         }

       }while(HANDLE == MDC_YES);

       ITEM_FOUND = MDC_NO;

     } 
  }

  return(NULL);

}

char *MdcHandleNormList(char *list,Uint32 **inrs,Uint32 *it
                                  ,Uint32 *bt,Uint32 max)
{
  int ITEM_FOUND=MDC_NO, HANDLE, REVERSED;
  Uint32 a1, a2, t, i, l, length;
  char *p, *item;

  length = (Uint32)strlen(list);

  /* <enter> = default: all */
  if (MdcUseDefault(list)) { 
    (*inrs)[1] = 0; *it = 2; return(NULL);
  }

  /* loop through string with entire list */
  for (p=list, item=list, l=0; l<=length; l++) {

     /* separate items: begins at digit, ends at space (or \t, \n, ...) */

     if (ITEM_FOUND == MDC_NO) {

       if (isdigit((int)p[l])) { item=&p[l]; ITEM_FOUND=MDC_YES; }

     }else if (isspace((int)p[l]) || p[l]=='\0') {

       p[l]='\0';

       if (MdcGetRange(item,&a1,&a2,&t) != MDC_OK)
         return("Error reading range item");

       if (a1 > max) a1 = max;
       if (a2 > max) a2 = max;

       if ( (a1==0) || (a2==0) ) {
         (*inrs)[1] = 0;
         *it = 2;
         return(NULL);
       }

       /* reversed range ? */
       REVERSED = (a1 > a2) ? MDC_YES : MDC_NO;

       /* initialize and get image numbers */
       i = a1; HANDLE = MDC_YES;
       do {

         /* store image number */
         (*inrs)[*it] = i;
         *it += 1;
         if ( (*it % MDC_BUF_ITMS) == 0 ) {
           if (((*inrs)=(Uint32 *)MdcRealloc((*inrs),
                        (*bt)*MDC_BUF_ITMS*sizeof(Uint32)))==NULL){
             return("Couldn't realloc images number buffer");
           }
           *bt += 1;
         }

         if ((REVERSED == MDC_YES) && (i < t)) break;

         /* set next image number in range */
         i = (REVERSED == MDC_YES) ? (i-t) : (i+t);

         /* check end of range */
         if (REVERSED == MDC_YES) {
           if (i < a2) HANDLE = MDC_NO;
         }else{
           if (i > a2) HANDLE = MDC_NO;
         }
         
       }while (HANDLE == MDC_YES);

       ITEM_FOUND = MDC_NO;

     }

  }

  return(NULL);

}

char *MdcHandlePixelList(char *list, Uint32 **cols, Uint32 **rows, 
                         Uint32 *it, Uint32 *bt)
{
  int ITEM_FOUND=MDC_NO;
  Uint32 r_from, r_to, r_step, c_from, c_to, c_step;
  Uint32 r, c, l, length, tmp;
  char *col, *row;
  char *p, *item;

  length = (Uint32)strlen(list);

  /* <enter> default = all */
  if (MdcUseDefault(list)) {
    (*cols)[*it] = 0;
    (*rows)[*it] = 0;
    *it+=1;
    return(NULL);
  }

  /* loop through string with entire list */
  for (p=list, item=list, l=0; l<=length; l++) {

     /* separate items: begins at digit, ends at space (or \t, \n, ...) */

     if (ITEM_FOUND == MDC_NO) {

       if (isdigit((int)p[l])) { item=&p[l]; ITEM_FOUND=MDC_YES; }

     }else if (isspace((int)p[l]) || p[l]=='\0') {

       p[l]='\0';

       col=item;
       row=strchr(item,',');

       if ( row == NULL) return("Wrong input!"); 

       *row = '\0'; row += 1;

       if (MdcGetRange(col,&c_from,&c_to,&c_step) != MDC_OK)
         return("Error reading column range");

       /* some checks */
       if (c_from == 0 || c_to == 0) {
         c_from = 0; c_to = 0;
       }else if (c_from > c_to) {
         tmp = c_from; c_from = c_to; c_to = tmp;
       } 

       if (MdcGetRange(row,&r_from,&r_to,&r_step) != MDC_OK)
         return("Error reading row range"); 

       /* some checks */
       if (r_from == 0 || r_to == 0) {
         r_from = 0; r_to = 0;
       }else if (r_from > r_to) {
         tmp = r_from; r_from = r_to; r_to = tmp;
       } 
       
       for (r=r_from; r<=r_to; r+=r_step)
       for (c=c_from; c<=c_to; c+=c_step) {
          (*cols)[*it] = c;
          (*rows)[*it] = r;
          *it+=1;
          if ( (*it % MDC_BUF_ITMS) == 0 ) {
            if ( ((*cols)=(Uint32 *)MdcRealloc((*cols),
                        (*bt)*MDC_BUF_ITMS*sizeof(Uint32))) == NULL) {
              return("Couldn't realloc pixels column buffer");
            }
            if (((*rows)=(Uint32 *)MdcRealloc((*rows),
                        (*bt)*MDC_BUF_ITMS*sizeof(Uint32))) == NULL) {
              return("Couldn't realloc pixels row buffer");
            }
          }
          *bt+=1;
       }

       ITEM_FOUND = MDC_NO;

     }
  }

  return(NULL);

}

char *MdcGetStrAcquisition(int acq_type)
{ 
  switch (acq_type) {
    case MDC_ACQUISITION_STATIC : return("Static");
                                       break;
    case MDC_ACQUISITION_DYNAMIC: return("Dynamic");
                                       break;
    case MDC_ACQUISITION_TOMO   : return("Tomographic");
                                       break;
    case MDC_ACQUISITION_GATED  : return("Gated");
                                       break;
    case MDC_ACQUISITION_GSPECT : return("GSPECT");
                                       break;
    default                     : return("Unknown");
  }
} 

char *MdcGetStrRawConv(int rawconv)
{
  switch (rawconv) {
    case MDC_NO            : return("No"); break;
    case MDC_FRMT_RAW      : return("Binary"); break;
    case MDC_FRMT_ASCII    : return("Ascii"); break;
    default                : return("Unknown");
  }
}

char *MdcGetStrEndian(int endian)
{
  switch (endian) {
    case MDC_BIG_ENDIAN   : return("Big"); break;
    case MDC_LITTLE_ENDIAN: return("Little"); break;
    default               : return("Unknown");
  }
}

char *MdcGetStrCompression(int compression)
{
  switch (compression) {
    case MDC_NO       : return("None"); break;
    case MDC_COMPRESS : return("Compress"); break;
    case MDC_GZIP     : return("Gzipped"); break;
    default           : return("Unknown");
  }
}

char *MdcGetStrPixelType(int type)
{
  switch (type) {
   case     BIT1: return("1-bit"); break;
   case   BIT8_S: return("Int8"); break;
   case   BIT8_U: return("Uint8"); break;
   case  BIT16_S: return("Int16"); break;
   case  BIT16_U: return("Uint16"); break;
   case  BIT32_S: return("Int32"); break;
   case  BIT32_U: return("Uint32"); break;
   case  BIT64_S: return("Int64"); break;
   case  BIT64_U: return("Uint64"); break;
   case    FLT32: return("IEEE float"); break;
   case    FLT64: return("IEEE double"); break;
   case    ASCII: return("ASCII"); break;
   case  VAXFL32: return("VAX  float"); break;
   case   COLRGB: return("RGB24 triplets"); break;
   default      : return("Unknown");
  }
}

char *MdcGetStrColorMap(int map)
{
  switch (map) {
   case MDC_MAP_PRESENT : return("present");     break;
   case MDC_MAP_GRAY    : return("gray normal"); break;
   case MDC_MAP_INVERTED: return("gray invers"); break;
   case MDC_MAP_RAINBOW : return("rainbow");     break;
   case MDC_MAP_COMBINED: return("combined");    break; 
   case MDC_MAP_HOTMETAL: return("hotmetal");    break;
   case MDC_MAP_LOADED  : return("loaded LUT");  break;
   default              : return("Unknown");
  }
}

char *MdcGetStrYesNo(int boolean)
{
  switch (boolean) {
    case MDC_NO : return("No");      break;
    case MDC_YES: return("Yes");     break;
    default     : return("Unknown");
  }
}

char *MdcGetStrSlProjection(int slice_projection)
{
  switch (slice_projection) {

    case MDC_TRANSAXIAL: strcpy(mdcbufr,"XY - Transaxial");
        break;
    case MDC_SAGITTAL  : strcpy(mdcbufr,"YZ - Sagittal");
        break;
    case MDC_CORONAL   : strcpy(mdcbufr,"XZ - Coronal");
        break;
    default: strcpy(mdcbufr,"Unknown");
  }

  return(mdcbufr);
}

char *MdcGetStrPatSlOrient(int patient_slice_orient)
{
  switch (patient_slice_orient) {

   case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
    strcpy(mdcbufr,"Supine;HeadFirst;Transverse");  break;
   case MDC_SUPINE_HEADFIRST_SAGITTAL   : 
    strcpy(mdcbufr,"Supine;HeadFirst;Sagittal");    break;
   case MDC_SUPINE_HEADFIRST_CORONAL    : 
    strcpy(mdcbufr,"Supine;HeadFirst;Coronal");     break;
   case MDC_SUPINE_FEETFIRST_TRANSAXIAL: 
    strcpy(mdcbufr,"Supine;FeetFirst;Transverse");  break;
   case MDC_SUPINE_FEETFIRST_SAGITTAL   : 
    strcpy(mdcbufr,"Supine;FeetFirst;Sagittal");    break;
   case MDC_SUPINE_FEETFIRST_CORONAL    : 
    strcpy(mdcbufr,"Supine;FeetFirst;Coronal");     break;
   case MDC_PRONE_HEADFIRST_TRANSAXIAL : 
    strcpy(mdcbufr,"Prone;HeadFirst;Transverse");   break;
   case MDC_PRONE_HEADFIRST_SAGITTAL    : 
    strcpy(mdcbufr,"Prone;HeadFirst;Sagittal");     break;
   case MDC_PRONE_HEADFIRST_CORONAL     : 
    strcpy(mdcbufr,"Prone;HeadFirst;Coronal");      break;
   case MDC_PRONE_FEETFIRST_TRANSAXIAL : 
    strcpy(mdcbufr,"Prone;FeetFirst;Transverse");   break;
   case MDC_PRONE_FEETFIRST_SAGITTAL    : 
    strcpy(mdcbufr,"Prone;FeetFirst;Sagittal");     break;
   case MDC_PRONE_FEETFIRST_CORONAL     :
    strcpy(mdcbufr,"Prone;FeetFirst;Coronal");      break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
    strcpy(mdcbufr,"DecubitusRight;HeadFirst;Transverse"); break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL   : 
    strcpy(mdcbufr,"DecubitusRight;HeadFirst;Sagittal");   break;
   case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL    : 
    strcpy(mdcbufr,"DecubitusRight;HeadFirst;Coronal");    break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL: 
    strcpy(mdcbufr,"DecubitusRight;FeetFirst;Transverse"); break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL   : 
    strcpy(mdcbufr,"DecubitusRight;FeetFirst;Sagittal");   break;
   case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL    : 
    strcpy(mdcbufr,"DecubitusRight;FeetFirst;Coronal");    break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL : 
    strcpy(mdcbufr,"DecubitusLeft;HeadFirst;Transverse");  break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL    : 
    strcpy(mdcbufr,"DecubitusLeft;HeadFirst;Sagittal");    break;
   case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL     : 
    strcpy(mdcbufr,"DecubitusLeft;HeadFirst;Coronal");     break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL : 
    strcpy(mdcbufr,"DecubitusLeft;FeetFirst;Transverse");  break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL    : 
    strcpy(mdcbufr,"DecubitusLeft;FeetFirst;Sagittal");    break;
   case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL     :
    strcpy(mdcbufr,"DecubitusLeft;FeetFirst;Coronal");     break;
   default                              : 
    strcpy(mdcbufr,"Unknown");

  }
  
  return(mdcbufr);
} 

char *MdcGetStrPatPos(int patient_slice_orient)
{
  switch (patient_slice_orient) {
    case MDC_SUPINE_HEADFIRST_TRANSAXIAL: 
    case MDC_SUPINE_HEADFIRST_SAGITTAL   :
    case MDC_SUPINE_HEADFIRST_CORONAL    :
        strcpy(mdcbufr,"HFS"); break;
    case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
    case MDC_SUPINE_FEETFIRST_SAGITTAL   :
    case MDC_SUPINE_FEETFIRST_CORONAL    :
        strcpy(mdcbufr,"FFS"); break;
    case MDC_PRONE_HEADFIRST_TRANSAXIAL :
    case MDC_PRONE_HEADFIRST_SAGITTAL    :
    case MDC_PRONE_HEADFIRST_CORONAL     :
        strcpy(mdcbufr,"HFP"); break;
    case MDC_PRONE_FEETFIRST_TRANSAXIAL :
    case MDC_PRONE_FEETFIRST_SAGITTAL    :
    case MDC_PRONE_FEETFIRST_CORONAL     :
        strcpy(mdcbufr,"FFP"); break;
    case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL: 
    case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL   :
    case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL    :
        strcpy(mdcbufr,"HFDR"); break;
    case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
    case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL   :
    case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL    :
        strcpy(mdcbufr,"FFDR"); break;
    case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
    case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL    :
    case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL     :
        strcpy(mdcbufr,"HFDL"); break;
    case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
    case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL    :
    case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL     :
        strcpy(mdcbufr,"FFDL"); break;
    default                              :
        strcpy(mdcbufr,"Unknown");
  }

  return(mdcbufr);

}

char *MdcGetStrPatOrient(int patient_slice_orient)
{
  switch (patient_slice_orient) {
    case MDC_SUPINE_HEADFIRST_TRANSAXIAL: strcpy(mdcbufr,"L\\P"); break;
    case MDC_SUPINE_HEADFIRST_SAGITTAL  : strcpy(mdcbufr,"P\\F"); break;
    case MDC_SUPINE_HEADFIRST_CORONAL   : strcpy(mdcbufr,"L\\F"); break;
    case MDC_SUPINE_FEETFIRST_TRANSAXIAL: strcpy(mdcbufr,"R\\P"); break;
    case MDC_SUPINE_FEETFIRST_SAGITTAL  : strcpy(mdcbufr,"P\\H"); break;
    case MDC_SUPINE_FEETFIRST_CORONAL   : strcpy(mdcbufr,"R\\H"); break;
    case MDC_PRONE_HEADFIRST_TRANSAXIAL : strcpy(mdcbufr,"R\\A"); break;
    case MDC_PRONE_HEADFIRST_SAGITTAL   : strcpy(mdcbufr,"A\\F"); break;
    case MDC_PRONE_HEADFIRST_CORONAL    : strcpy(mdcbufr,"R\\F"); break;
    case MDC_PRONE_FEETFIRST_TRANSAXIAL : strcpy(mdcbufr,"L\\A"); break;
    case MDC_PRONE_FEETFIRST_SAGITTAL   : strcpy(mdcbufr,"A\\H"); break;
    case MDC_PRONE_FEETFIRST_CORONAL    : strcpy(mdcbufr,"L\\H"); break;
    case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL: strcpy(mdcbufr,"P\\R");break;
    case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL  : strcpy(mdcbufr,"L\\F");break;
    case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL   : strcpy(mdcbufr,"P\\F");break;
    case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL: strcpy(mdcbufr,"A\\R");break;
    case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL  : strcpy(mdcbufr,"L\\H");break;
    case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL   : strcpy(mdcbufr,"A\\H");break;
    case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL : strcpy(mdcbufr,"A\\L");break;
    case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL   : strcpy(mdcbufr,"R\\F");break;
    case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL    : strcpy(mdcbufr,"A\\F");break;
    case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL : strcpy(mdcbufr,"P\\L");break;
    case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL   : strcpy(mdcbufr,"R\\H");break;
    case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL    : strcpy(mdcbufr,"P\\H");break;
    default                              : strcpy(mdcbufr,"Unknown");
  }

  return(mdcbufr);
}

char *MdcGetStrSliceOrient(int patient_slice_orient)
{
  switch (patient_slice_orient) {
     case MDC_SUPINE_HEADFIRST_TRANSAXIAL         :
     case MDC_PRONE_HEADFIRST_TRANSAXIAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
     case MDC_SUPINE_FEETFIRST_TRANSAXIAL         :
     case MDC_PRONE_FEETFIRST_TRANSAXIAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
      strcpy(mdcbufr,"Transverse"); break;
     case MDC_SUPINE_HEADFIRST_SAGITTAL         :
     case MDC_PRONE_HEADFIRST_SAGITTAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
     case MDC_SUPINE_FEETFIRST_SAGITTAL         :
     case MDC_PRONE_FEETFIRST_SAGITTAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
      strcpy(mdcbufr,"Sagittal");  break;
     case MDC_SUPINE_HEADFIRST_CORONAL         :
     case MDC_PRONE_HEADFIRST_CORONAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
     case MDC_SUPINE_FEETFIRST_CORONAL         :
     case MDC_PRONE_FEETFIRST_CORONAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
      strcpy(mdcbufr,"Coronal");   break;
     default                              : 
      strcpy(mdcbufr,"unknown");
  }

  return(mdcbufr);
}

char *MdcGetStrRotation(int rotation)
{ 
  switch (rotation) {
    case MDC_ROTATION_CW: strcpy(mdcbufr,"clockwise");         break;
    case MDC_ROTATION_CC: strcpy(mdcbufr,"counter-clockwise"); break;
    default             : strcpy(mdcbufr,"unknown");
  }

  return(mdcbufr);
}

char *MdcGetStrMotion(int motion)
{
  switch (motion) {
    case MDC_MOTION_STEP: strcpy(mdcbufr,"step and shoot");  break;
    case MDC_MOTION_CONT: strcpy(mdcbufr,"continuous");    break;
    case MDC_MOTION_DRNG: strcpy(mdcbufr,"during step");   break;
    default             : strcpy(mdcbufr,"unknown");
  }

  return(mdcbufr);
}

char *MdcGetStrModality(int modint)
{
  char *pmod;
  Uint16 umod16;

  umod16 = (Uint16)modint;

  pmod = (char *)&umod16;

  if (MdcHostBig()) {
    mdcbufr[0] = pmod[0];
    mdcbufr[1] = pmod[1];
  }else{
    mdcbufr[0] = pmod[1];
    mdcbufr[1] = pmod[0];
  }
  mdcbufr[2]='\0';

  return(mdcbufr);
}

char *MdcGetStrGSpectNesting(int nesting)
{
  switch (nesting) {
    case MDC_GSPECT_NESTING_SPECT: return("SPECT");
    case MDC_GSPECT_NESTING_GATED: return("Gated");
    default : return("unknown");
  }
}

char *MdcGetStrHHMMSS(float msecs)
{
  unsigned int s, ms, hrs, mins, secs;

  s = (unsigned int)(msecs / 1000.0f);

  ms = (unsigned int)(msecs - (s * 1000));

  hrs  = s / 3600; s -= hrs  * 3600;

  mins = s / 60;   s -= mins * 60;

  secs = s;

  if (hrs > 0) {
    sprintf(mdcbufr,"%02uh%02um%02u",hrs,mins,secs);
  }else if (mins > 0) {
    sprintf(mdcbufr,"%02um%02u",mins,secs);
  }else{
    sprintf(mdcbufr,"%02us%03u",secs,ms);
  }

  return(mdcbufr);
}

int MdcGetIntModality(char *modstr)
{
  int modint;

  modint = (modstr[0]<<8)|modstr[1];

  return(modint);
}

int MdcGetIntSliceOrient(int patient_slice_orient)
{
  int slice_orient;

  switch (patient_slice_orient) {
     case MDC_SUPINE_HEADFIRST_TRANSAXIAL         :
     case MDC_PRONE_HEADFIRST_TRANSAXIAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL :
     case MDC_SUPINE_FEETFIRST_TRANSAXIAL         :
     case MDC_PRONE_FEETFIRST_TRANSAXIAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL :
      slice_orient = MDC_TRANSAXIAL;
      break;
     case MDC_SUPINE_HEADFIRST_SAGITTAL         :
     case MDC_PRONE_HEADFIRST_SAGITTAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL :
     case MDC_SUPINE_FEETFIRST_SAGITTAL         :
     case MDC_PRONE_FEETFIRST_SAGITTAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL :
      slice_orient = MDC_SAGITTAL;
      break;
     case MDC_SUPINE_HEADFIRST_CORONAL         :
     case MDC_PRONE_HEADFIRST_CORONAL          :
     case MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL :
     case MDC_SUPINE_FEETFIRST_CORONAL         :
     case MDC_PRONE_FEETFIRST_CORONAL          :
     case MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL:
     case MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL :
      slice_orient = MDC_CORONAL;
      break;
     default                              : 
      slice_orient = MDC_TRANSAXIAL;
  }

  return(slice_orient);

}

const char *MdcGetLibLongVersion(void)
{
  return(MDC_LIBVERS);
}

const char *MdcGetLibShortVersion(void)
{
  return(MDC_VERSION);
}
  
/* returns the new stringsize value or 0 in case of error to add */
Uint32 MdcCheckStrSize(char *str_to_add, Uint32 current_size, Uint32 max)
{
  Uint32 max_value = MDC_2KB_OFFSET;
  Uint32 new_size;

  if (max != 0) max_value = max;

  new_size = current_size + (Uint32)strlen(str_to_add);

  if ( new_size >= max_value ) {
    MdcPrntWarn("Internal Problem -- Information string too small");
    return(0);
  }

  return(new_size);
} 

/* print to global `mdcbufr' array */
int MdcMakeScanInfoStr(FILEINFO *fi)
{
  char strbuf[100];
  Uint32 size=0;

  sprintf(mdcbufr,"\n\n\
******************************\n\
Short Patient/Scan Information\n\
******************************\n");
  size = (Uint32)strlen(mdcbufr);
  sprintf(strbuf,"Patient Name  : %s\n",fi->patient_name);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Patient Sex   : %s\n",fi->patient_sex);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Patient ID    : %s\n",fi->patient_id);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Patient DOB   : %s\n",fi->patient_dob);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Patient Weight: %.2f\n",fi->patient_weight);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Study Date  : %02d/%02d/%04d\n",fi->study_date_day
                                               ,fi->study_date_month
                                               ,fi->study_date_year);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Study Time  : %02d:%02d:%02d\n",fi->study_time_hour
                                               ,fi->study_time_minute
                                               ,fi->study_time_second);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Study ID    : %s\n",fi->study_id);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Study Descr : %s\n",fi->study_descr);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Acquisition Type     : %s\n",
                                 MdcGetStrAcquisition(fi->acquisition_type));
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Reconstructed        : %s\n",
                                 MdcGetStrYesNo(fi->reconstructed));
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;

  if (fi->reconstructed == MDC_YES) {
  sprintf(strbuf,"Reconstruction Method: %s\n",fi->recon_method);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Filter Type          : %s\n",fi->filter_type);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Decay Corrected      : %s\n",
                                       MdcGetStrYesNo(fi->decay_corrected));
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Flood Corrected      : %s\n",
                                       MdcGetStrYesNo(fi->flood_corrected));
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Series Description   : %s\n",fi->series_descr);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Radiopharmaceutical  : %s\n",fi->radiopharma);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  }
  sprintf(strbuf,"Isotope Code         : %s\n",fi->isotope_code);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Isotope Halflife     : %+e [sec]\n",
                                       fi->isotope_halflife);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Injected Dose        : %+e [MBq]\n",
                                       fi->injected_dose);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;
  sprintf(strbuf,"Gantry Tilt          : %+e degrees\n",
                                       fi->gantry_tilt);
  if ((size=MdcCheckStrSize(strbuf,size,0))) strcat(mdcbufr,strbuf);
  else return MDC_NO;

  return(MDC_YES);

}

int MdcIsDigit(char c) 
{
  if (c >= '0' && c <= '9') return MDC_YES;
  
  return(MDC_NO);
}

void MdcWaitForEnter(int page)
{
  if (page > 0) {
    MdcPrntScrn("\t\t*********** Press <enter> for page #%d **********",page);
  }
  if (page == 0) {
    MdcPrntScrn("\t\t********** Press <enter> for next page **********");
  }
  if (page < 0 ) {
    MdcPrntScrn("Press <enter> to continue ...");
  }
  while ( fgetc(stdin) != '\n' ) { /* wait until <enter> key pressed */ }
}

Int32 MdcGetSelectionType(void)
{
   Int32 type=-1;

   MdcPrntScrn("\n\tSelection Type:\n");
   MdcPrntScrn("\n\ttype  %d  ->  normal",MDC_INPUT_NORM_STYLE);
   MdcPrntScrn("\n\t      %d  ->  ecat\n",MDC_INPUT_ECAT_STYLE);
   MdcPrntScrn("\n\tYour choice [%d]? ",MDC_INPUT_NORM_STYLE);
   MdcGetStrLine(mdcbufr,MDC_2KB_OFFSET-1,stdin);

   type=(Int32)atol(mdcbufr);

   if (type != MDC_INPUT_ECAT_STYLE) type = MDC_INPUT_NORM_STYLE;

   return(type);
}

void MdcFlushInput(void)
{
   while( fgetc(stdin) != '\n' ) { }
}

int MdcWhichDecompress(void)
{
  if (strcmp(MDC_DECOMPRESS,"gunzip")     == 0) return(MDC_GZIP);
  if (strcmp(MDC_DECOMPRESS,"uncompress") == 0) return(MDC_COMPRESS);

  return(MDC_NO);
}

int MdcWhichCompression(const char *fname)
{
  char *ext=NULL;
  int compression = MDC_NO;

  /* get filename extension */
  if (fname != NULL) ext = strrchr(fname,'.');
  if (ext != NULL) {
    /* check for supported compression */
    switch (MdcWhichDecompress()) {

      case MDC_COMPRESS: if (strcmp(ext,".Z") == 0 )        /* only .Z files */
                           compression = MDC_COMPRESS;
                         break;
      case MDC_GZIP    : if (strcmp(ext,".gz") == 0 ) {
                           compression = MDC_GZIP;
                         }else if (strcmp(ext,".Z")  == 0 ) {
                           compression = MDC_COMPRESS;
                         }
                         break;
    }
  }

  return(compression);

}

void MdcAddCompressionExt(int ctype, char *fname)
{
   switch (ctype) {
     case MDC_COMPRESS: strcat(fname,".Z");  break;

     case MDC_GZIP    : strcat(fname,".gz"); break;
   } 
}
