/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-intf.c                                                      *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : read and write InterFile 3.3                             *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcCheckINTF()        - Check for InterFile 3.3 format   *
 *                MdcGetIntfKey()       - Get InterFile key                *
 *                MdcInitIntf()         - Init InterFile struct (defaults) *
 *                MdcIntfIsString()     - String occurency test            *
 *                MdcIsArrayKey()       - Check for array like keys {,,}   *
 *                MdcGetMaxIntArrayKey()- Get max integer from array key   *
 *                MdcGetIntKey()        - Get key with integer             *
 *                MdcGetYesNoKey()      - Get key Y or N                   *
 *                MdcGetFloatKey()      - Get key with float               *
 *                MdcGetStrKey()        - Get key with string              *
 *                MdcGetSubStrKey()     - Get string between separators    *
 *                MdcGetDateKey()       - Get key with date format         *
 *                MdcGetSplitDateKey()  - Get date in year, month, day     *
 *                MdcGetSplitTimeKey()  - Get time in hour, minute, sec    *
 *                MdcGetDataType()      - Get data type of pixels          *
 *                MdcGetProcessStatus() - Get process status               *
 *                MdcGetPatRotation()   - Get patient rotation             *
 *                MdcGetPatOrientation()- Get patient orientation          *
 *                MdcGetSliceOrient()   - Get slice orient                 *
 *                MdcGetPatSlOrient()   - Get patient slice orientation    *
 *                MdcGetPixelType()     - Get pixel data type              *
 *                MdcGetRotation()      - Get rotation direction           *
 *                MdcGetMotion()        - Get detector motion              *
 *                MdcGetGSpectNesting() - Get Gated SPECT nesting          *
 *                MdcSpecifyPixelType() - Specify pixel data type (bytes)  *
 *                MdcHandleIntfDialect()- Handle InterFile dialect headers *
 *                MdcReadIntfHeader()   - Read InterFile header            *
 *                MdcReadIntfImages()   - Read InterFile images            *
 *                MdcReadINTF()         - Read InterFile file              *
 *                MdcType2Intf()        - Translate data type to InterFile *
 *                MdcGetProgramDate()   - Get date in correct format       *
 *                MdcCheckIntfDim()     - Check supported dimensions       *
 *                MdcSetPatRotation()   - Set patient rotation string      *
 *                MdcSetPatOrientation()- Set patient orientation string   *
 *                MdcWriteGenImgData()  - Write general image data         *
 *                MdcWriteWindows()     - Write energy windows             *
 *                MdcWriteMatrixInfo()  - Write matrix info                *
 *                MdcWriteIntfStatic()  - Write a Static header            *
 *                MdcWriteIntfDynamic() - Write a Dynamic header           *
 *                MdcWriteIntfTomo()    - Write a Tomographic header       *
 *                MdcWriteIntfGated()   - Write a Gated header             *
 *                MdcWriteIntfGSPECT()  - Write a GSPECT header            *
 *                MdcWriteIntfHeader()  - Write InterFile header           *
 *                MdcWriteIntfImages()  - Write InterFile images           *
 *                MdcWriteINTF()        - Write InterFile file             * 
 *                /// Agregado por Martin Belzunce: MdcWriteIntfPET()      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-intf.c,v 1.131 2010/08/28 23:44:23 enlf Exp $
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

//#include "m-depend.h"

#include <stdio.h>
#include <stdlib.h>
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
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "medcon.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

#define MDC_IGNORE_DATA_ENCODE   1  /* 0/1 - ignore "data encode" key      */
#define MDC_IGNORE_DATA_COMPRESS 1  /* 0/1 - ignore "data compression" key */

#define MDC_INTF_SUPPORT_DIALECT 1  /* 0/1 - support dialect interfile     */

#define MDC_INTF_SUPPORT_SCALE   1  /* 0/1 - support global scale factor   */

#define MDC_INTF_SUPPORT_NUD     1  /* 0/1 - support NUD extended keys     */

#define MdcThisString(x)     MdcIntfIsString(x,0)
#define MdcThisKey(x)        MdcIntfIsString(x,1)

#define MDC_INTF_DATA_OFFSET  5120

static char keystr[MDC_INTF_MAXKEYCHARS+1];       /* all lower case       */
static char keystr_check[MDC_INTF_MAXKEYCHARS+1]; /* all lower, no spaces */
static char keystr_case[MDC_INTF_MAXKEYCHARS+1];  /* original key string  */

static Uint32 ACQI = 0;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

int MdcCheckINTF(FILEINFO *fi)
{
  if (MdcGetIntfKey(fi->ifp) != MDC_OK) return(MDC_BAD_READ);

  if (strstr(keystr_check,MDC_INTF_SIG) == NULL)  return(MDC_FRMT_NONE);

  return(MDC_FRMT_INTF);
}


/* three key string types to retrieve:    */
/* 1. original key string                 */
/* 2. case insensitive                    */
/* 3. case insensitive and without spaces */
int MdcGetIntfKey(FILE *fp)
{
  char *c, *pkeyval = NULL;

  /* assure string termination */
  memset(keystr,'\0',MDC_INTF_MAXKEYCHARS+1);

  fgets(keystr,MDC_INTF_MAXKEYCHARS,fp);
  if (ferror(fp)) return(MDC_BAD_READ);

  /* remove any comment from the key line*/
  c = strchr(keystr,';'); if (c != NULL) c[0]='\0';

  /* check for valid and safe interfile key */
  if (strstr(keystr,":=") == NULL) strcat(keystr,":=\n");

  /* [1] preserve original key string, without comments */ 
  memcpy(keystr_case,keystr,MDC_INTF_MAXKEYCHARS+1);

  /* remove spaces from key value */
  pkeyval = strstr(keystr,":=") + 2;
  MdcKillSpaces(pkeyval);

  /* [2] preserve case insensitive key string */
  MdcLowStr(keystr);

  /* [3] case insensitive and without spaces */
  strcpy(keystr_check,keystr); MdcRemoveAllSpaces(keystr_check);

  return MDC_OK;
  
}


void MdcInitIntf(MDC_INTERFILE *intf)
{
  intf->DIALECT = MDC_NO;
  intf->dim_num = 0; intf->dim_found = 0;
  intf->data_type = MDC_INTF_STATIC;
  intf->process_status = MDC_INTF_UNKNOWN;
  intf->pixel_type = BIT8_U;
  intf->width = 0; intf->height = 0;
  intf->images_per_dimension = 1;
  intf->time_slots = 0;
  intf->data_offset = 0; intf->data_blocks = 0;
  intf->imagesize = 0; intf->number_images = 0;
  intf->energy_windows = intf->frame_groups = 1;
  intf->time_windows = intf->detector_heads = 1;
  intf->pixel_xsize = 1.; intf->pixel_ysize = 1.;
  intf->slice_thickness=1.; intf->slice_thickness_mm=1.;
  intf->centre_centre_separation=1.;
  intf->study_duration=0.;
  intf->image_duration = 0.;
  intf->image_pause = 0.;
  intf->group_pause = 0.;
  intf->ext_rot = 0.;
  intf->procent_cycles_acquired = 100.;  /* default: acquired = observed */
  intf->rescale_slope = 1.; intf->rescale_intercept = 0.;
  intf->patient_rot = MDC_SUPINE;
  intf->patient_orient = MDC_HEADFIRST;
  intf->slice_orient =  MDC_TRANSAXIAL;
}

int MdcIsEmptyKeyValue(void)
{
   char *pkeyval = NULL;
 
   pkeyval = strstr(keystr_check,":=") + 2;

   if (pkeyval[0] == '\0') return(MDC_YES);

   return(MDC_NO);
}

int MdcIntfIsString(char *string, int key)
{
  char check[MDC_INTF_MAXKEYCHARS+1];

  strcpy(check,string);

  if (key) strcat(check,":="); /* add key delimiter */

  MdcRemoveAllSpaces(check);

  MdcLowStr(check);

  if( strstr(keystr_check,check) != NULL) return MDC_YES;

  return MDC_NO;
}

int MdcIsArrayKey(void)
{
  char *pkeyval;

  pkeyval = strstr(keystr_check,":=") + 2;
  pkeyval = strchr(pkeyval,'{');

  if (pkeyval != NULL) return(MDC_YES);

  return(MDC_NO);
}

int MdcGetMaxIntArrayKey(void)
{
  char *pkeyval;
  int value, max=0;

  pkeyval = strstr(keystr,":=") + 2;
  if (pkeyval == NULL) return(max);

  pkeyval = strchr(pkeyval,'{');
  while (pkeyval != '\0' ) {
    pkeyval++;
    value = atoi(pkeyval);
    if (value > max) max = value;
    pkeyval = strchr(pkeyval,','); 
  }

  return(max);
}

int MdcGetIntKey(void)
{
  return(atoi(strstr(keystr,":=") + 2));
}

int MdcGetYesNoKey(void)
{
  strcpy(mdcbufr,(strstr(keystr,":=") + 2));
  MdcKillSpaces(mdcbufr);
  if (mdcbufr[0] == 'y') return MDC_YES;
  if (mdcbufr[0] == 'n') return MDC_NO;
  return(MDC_NO);
}
  

double MdcGetFloatKey(void)
{
  double d=-1;
  char* strNum = strstr(keystr,":=") + 2;
  d = atof(strNum);
  return(d);
}

void MdcGetStrKey(char *str)
{
  memcpy(str,(strstr(keystr_case,":=") + 2),MDC_MAXSTR-1);
  str[MDC_MAXSTR-1] = '\0';
  MdcKillSpaces(str);
}

void MdcGetSubStrKey(char *str, int n)
{
  char *pkey;

  pkey = strstr(keystr_case,":=") + 2;
  MdcGetSubStr(str,pkey,MDC_MAXSTR,'/',n);
}

void MdcGetDateKey(char *str)
{
  int i, t;

  memcpy(str,(strstr(keystr_case,":=") + 2),MDC_MAXSTR-1);
  str[MDC_MAXSTR-1] = '\0';
  MdcKillSpaces(str);

  /* fix into YYYYMMDD format */
  for (t=0,i=0; i < strlen(str); i++) {
     if (str[i] != ':') str[t++] = str[i];
  }
  str[t]='\0';

}

void MdcGetSplitDateKey(Int16 *year, Int16 *month, Int16 *day)
{
  sscanf((char *)(strstr(keystr,":=")+2),"%4hd:%2hd:%2hd",year,month,day);
}

void MdcGetSplitTimeKey(Int16 *hour, Int16 *minute, Int16 *second)
{
  sscanf((char *)(strstr(keystr,":=")+2),"%2hd:%2hd:%2hd",hour,minute,second);
}

int MdcGetDataType(void)
{
  if (MdcThisString("gatedtomo"))
    return MDC_INTF_GSPECT; /* IS2 dialect, check before planar "gated" ;-) */
  if (MdcThisString("static"))
    return MDC_INTF_STATIC;
  if (MdcThisString("dynamic"))
    return MDC_INTF_DYNAMIC;
  if (MdcThisString("gated"))
    return MDC_INTF_GATED;
  if (MdcThisString("tomographic"))
    return MDC_INTF_TOMOGRAPH;
  if (MdcThisString("curve"))
    return MDC_INTF_CURVE;
  if (MdcThisString("roi"))
    return MDC_INTF_ROI;
  if (MdcThisString("gspect"))
    return MDC_INTF_GSPECT;
  if (MdcThisString("pet"))
    return MDC_INTF_DIALECT_PET;
  /// Agregado por Martín Belzunce
  if (MdcThisString("PET"))
    return MDC_INTF_PET;

  return MDC_INTF_UNKNOWN;
}

int MdcGetProcessStatus(void)
{
  if (MdcThisString("acquired"))
    return MDC_INTF_ACQUIRED;
  if (MdcThisString("reconstructed"))
    return MDC_INTF_RECONSTRUCTED;
 
  return MDC_INTF_UNKNOWN;
}

int MdcGetPatRotation(void)
{
  if (MdcThisString("supine")) return MDC_SUPINE;
  if (MdcThisString("prone"))  return MDC_PRONE;
  
  return MDC_UNKNOWN;
}

int MdcGetPatOrientation(void)
{
  if (MdcThisString("head")) return MDC_HEADFIRST;
  if (MdcThisString("feet")) return MDC_FEETFIRST;
 
  return MDC_UNKNOWN;
}

int MdcGetSliceOrient(void)
{
  if (MdcThisString("transverse")) return MDC_TRANSAXIAL;
  if (MdcThisString("sagittal"))   return MDC_SAGITTAL;
  if (MdcThisString("coronal"))    return MDC_CORONAL;

  return MDC_UNKNOWN;
}  

int MdcGetPatSlOrient(MDC_INTERFILE *intf)
{
  switch (intf->patient_rot) {
    case MDC_SUPINE: 
        switch (intf->patient_orient) {
               case MDC_HEADFIRST: 
                   switch (intf->slice_orient) {
                         case MDC_TRANSAXIAL:
                             return(MDC_SUPINE_HEADFIRST_TRANSAXIAL); break;
                         case MDC_SAGITTAL   :
                             return(MDC_SUPINE_HEADFIRST_SAGITTAL);    break;
                         case MDC_CORONAL    :
                             return(MDC_SUPINE_HEADFIRST_CORONAL);     break; 
                   }
                   break;

               case MDC_FEETFIRST: 
                   switch(intf->slice_orient) {
                         case MDC_TRANSAXIAL: 
                             return(MDC_SUPINE_FEETFIRST_TRANSAXIAL); break;
                         case MDC_SAGITTAL   :
                             return(MDC_SUPINE_FEETFIRST_SAGITTAL);    break;
                         case MDC_CORONAL    :
                             return(MDC_SUPINE_FEETFIRST_CORONAL);     break;
                   }
                   break;
        }
        break;
    case MDC_PRONE : 
        switch (intf->patient_orient) {
              case MDC_HEADFIRST:  
                  switch (intf->slice_orient) {
                         case MDC_TRANSAXIAL: 
                             return(MDC_PRONE_HEADFIRST_TRANSAXIAL);  break;
                         case MDC_SAGITTAL   :
                             return(MDC_PRONE_HEADFIRST_SAGITTAL);     break;
                         case MDC_CORONAL    :
                             return(MDC_PRONE_HEADFIRST_CORONAL);      break;
                  }
                  break;
              case MDC_FEETFIRST:
                  switch (intf->slice_orient) {
                         case MDC_TRANSAXIAL: 
                             return(MDC_PRONE_FEETFIRST_TRANSAXIAL);  break;
                         case MDC_SAGITTAL   :
                             return(MDC_PRONE_FEETFIRST_SAGITTAL);     break;
                         case MDC_CORONAL    :
                             return(MDC_PRONE_FEETFIRST_CORONAL);      break;
                  }
                  break;
        }
        break;
  }

  return(MDC_SUPINE_HEADFIRST_TRANSAXIAL); /* default for InterFile (!) */

}

int MdcGetPixelType(void)
{
  if (MdcThisString("unsigned integer")) return BIT8_U;
  if (MdcThisString("signed integer"))   return BIT8_S;
  if (MdcThisString("long float"))       return FLT64;
  if (MdcThisString("short float"))      return FLT32;
  if (MdcThisString("float"))            return FLT32;
  if (MdcThisString("bit"))              return BIT1;
  if (MdcThisString("ascii"))            return ASCII;

  return BIT8_U;
}

int MdcGetRotation(void)
{
  if (MdcThisString("ccw")) return(MDC_ROTATION_CC);
  if (MdcThisString("cw"))  return(MDC_ROTATION_CW);

  return(MDC_UNKNOWN);
}

int MdcGetMotion(void)
{
  if (MdcThisString("step"))       return(MDC_MOTION_STEP);
  if (MdcThisString("continuous")) return(MDC_MOTION_CONT);

  return(MDC_UNKNOWN);
}

int MdcGetGSpectNesting(void)
{ /* can not use MdcThisString() because "SPECT" */
  /* can be mentioned in key as well as in value */
  char *pkeyval;
  
  if ( (pkeyval = strstr(keystr,":=")) != NULL ) {
    if (strstr(pkeyval,"spect") != NULL) return(MDC_GSPECT_NESTING_SPECT);
    if (strstr(pkeyval,"gated") != NULL) return(MDC_GSPECT_NESTING_GATED);
  }

  return(MDC_GSPECT_NESTING_GATED);

}

int MdcSpecifyPixelType(MDC_INTERFILE *intf)
{
   int bytes;

   bytes = MdcGetIntKey();

   if (intf->pixel_type == BIT8_S) 
     switch (bytes) {
       case 1: break;
       case 2: intf->pixel_type = BIT16_S; break;
       case 4: intf->pixel_type = BIT32_S; break;
       case 8: intf->pixel_type = BIT64_S; break;
      default: intf->pixel_type = 0;
   }else if (intf->pixel_type == BIT8_U)
     switch (bytes) {
       case 1: break;
       case 2: intf->pixel_type = BIT16_U; break;
       case 4: intf->pixel_type = BIT32_U; break;
       case 8: intf->pixel_type = BIT64_U; break;
      default: intf->pixel_type = 0;
  }

  return intf->pixel_type;

}

char *MdcHandleIntfDialect(FILEINFO *fi, MDC_INTERFILE *intf)
{
  int d, number=1;

  /* increment number of dimensions found */
  intf->dim_found += 1;

  /* with "total number of images" key present -> already allocated */
  /* if ((fi->number != 0) && (fi->image != NULL))  return(NULL);   */

  if (intf->dim_num == intf->dim_found) {
    for (d=3; d<=intf->dim_num; d++) number *= fi->dim[d];
    if (number == 0) return("INTF Bad matrix size values (dialect)");
    if (!MdcGetStructID(fi,(Uint32)number))
      return("INTF Bad malloc IMG_DATA structs (dialect)");
  }

  return NULL;
}


char *MdcReadIntfHeader(FILEINFO *fi, MDC_INTERFILE *intf)
{
  DYNAMIC_DATA *dd=NULL;
  GATED_DATA *gd=NULL;
  STATIC_DATA *sd=NULL;
  ACQ_DATA *acq=NULL;
  IMG_DATA *id;
  FILE *fp = fi->ifp;
  Uint32 i, counter=0, total=0, img=0, uv, number=0, acqnr=0;
  char *err=NULL, *pfname=NULL;
  float v;
  int matrix_size_4=MDC_FALSE;

  if (MDC_INFO) {
    MdcPrintLine('-',MDC_HALF_LENGTH);
    MdcPrntScrn("InterFile Header\n");
    MdcPrintLine('-',MDC_HALF_LENGTH);
  }

  while (!feof(fp)) {
    if (MdcGetIntfKey(fp) != MDC_OK) return("INTF Bad read of key");
    if (err != NULL) return(err);
    if (MDC_INFO) {
      MdcPrntScrn("%s",keystr_case);
    }
    if (ferror(fp)) return("INTF Bad read header file");
    if (MdcThisString(";")) continue;
    if (MdcThisKey("version of keys")) {
      if (strstr(keystr_case,MDC_INTF_SUPP_VERS) == NULL)
        MdcPrntWarn("INTF Unexpected version of keys found");
      continue;
    }
#if ! (MDC_IGNORE_DATA_COMPRESS)
    if (MdcThisKey("data compression")) {
      if (! (MdcThisString("none") || MdcIsEmptyKeyValue()) ) 
        return("INTF Don't handle compressed images");
    }
#endif
#if ! (MDC_IGNORE_DATA_ENCODE)
    if (MdcThisKey("data encode")) {
      if (! (MdcThisString("none") || MdcIsEmptyKeyValue()) )
        return("INTF Don't handle encoded images");
    }
#endif
    if (MdcThisKey("organ")) {
      MdcGetStrKey(fi->organ_code);
      continue;
    }
    if (MdcThisKey("isotope")) {
      MdcGetSubStrKey(fi->isotope_code,1);
      MdcGetSubStrKey(fi->radiopharma,2);
      continue;
    }
    if (MdcThisKey("dose")) {
      fi->injected_dose = (float)MdcGetFloatKey();
      continue;
    }
#if MDC_INTF_SUPPORT_NUD
    if (MdcThisKey("patient weight [kg]")) {
      fi->patient_weight = (float)MdcGetFloatKey();
      continue;
    }
    if (MdcThisKey("imaging modality")) {
      MdcGetStrKey(mdcbufr); fi->modality = MdcGetIntModality(mdcbufr);
      continue;
    }
    if (MdcThisKey("activity")) {
      fi->injected_dose = MdcGetFloatKey();
      continue;
    }
    if (MdcThisKey("activity start time")) {
      MdcGetSplitTimeKey(&fi->dose_time_hour
                        ,&fi->dose_time_minute
                        ,&fi->dose_time_second);
      continue;
    }
    if (MdcThisKey("isotope half life [hours]")) {
      fi->isotope_halflife = (float)MdcGetFloatKey() * 3600.;
      continue;
    }
#endif
    if (MdcThisKey("original institution")) {
      if (MdcIsEmptyKeyValue() == MDC_NO) MdcGetStrKey(fi->institution);
      continue;
    }
    if (MdcThisKey("originating system")) {
      if (MdcIsEmptyKeyValue() == MDC_NO) MdcGetStrKey(fi->manufacturer);
      continue;
    }
    if (MdcThisKey("data starting block")) {
      intf->data_offset = MdcGetIntKey() * 2048L;
      continue;
    }
    if (MdcThisKey("data offset in bytes")) {
      intf->data_offset = MdcGetIntKey();
      continue;
    }
    if (MdcThisKey("name of data file")) {
      pfname = strstr(keystr_case,":=") + 2;
      MdcKillSpaces(pfname);
      /* protect against empty key */
      if ( strlen(pfname) > 0 ) {

        fi->ifname = (MDC_IGNORE_PATH==MDC_YES) ? MdcGetFname(pfname) : pfname;

        if ((MDC_IGNORE_PATH == MDC_NO) && 
            (MdcThisString("/") || MdcThisString("\\"))) { 
          /* use absolute path mentioned in header file */
          strcpy(fi->ipath,fi->ifname);
        }else{
          /* use relative path where header was loaded */
          if (fi->idir != NULL) {
            /* assume fi->idir = fi->ipath */
            strcat(fi->ipath,"/");
            strcat(fi->ipath,fi->ifname);
          }else{ 
            strcpy(fi->ipath,fi->ifname); 
          }
        } 
        MdcSplitPath(fi->ipath,fi->idir,fi->ifname);
      }
      continue;
    } 
    if (MdcThisKey("patient name")) {
      MdcGetStrKey(fi->patient_name); continue;
    }
    if (MdcThisKey("patient id")) {
      MdcGetStrKey(fi->patient_id); continue;
    }
    if (MdcThisKey("patient dob")) {
      MdcGetDateKey(fi->patient_dob); continue;
    } 
    if (MdcThisKey("patient sex")) {
      MdcGetStrKey(fi->patient_sex); continue;
    }

    if (MdcThisKey("study id")) {
      MdcGetStrKey(fi->study_id); continue;
    }
   
    if (MdcThisKey("exam type")) {
      MdcGetStrKey(fi->series_descr); continue;
    }
 
    if (MdcThisKey("total number of images")) {
      number = MdcGetIntKey();
      if (number == 0) return("INTF No valid images specified");
      if (!MdcGetStructID(fi,number))
        return("INTF Bad malloc IMG_DATA structs"); 
      continue;
    }
    if (MdcThisKey("imagedata byte order")) {
      if (MdcThisString("bigendian")) 
        MDC_FILE_ENDIAN = MDC_BIG_ENDIAN;
      else if (MdcThisString("littleendian")) 
        MDC_FILE_ENDIAN = MDC_LITTLE_ENDIAN;
      else 
        MDC_FILE_ENDIAN = MDC_BIG_ENDIAN;
      fi->endian = MDC_FILE_ENDIAN; 
      if (intf->DIALECT == MDC_NO) continue; /* linked with end of interfile */
    }
    if (MdcThisKey("process label")) {
      MdcGetStrKey(fi->study_descr); continue;
    }
    if (MdcThisKey("type of data")) {
      intf->data_type = MdcGetDataType(); 
      if (intf->data_type == MDC_INTF_UNKNOWN)
        intf->data_type = MDC_INTF_STATIC;  /* take this as default */
      switch (intf->data_type) {
        case MDC_INTF_DYNAMIC    : 
                          fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
                          fi->planar = MDC_YES;
                          break;
        case MDC_INTF_TOMOGRAPH  : 
                          fi->acquisition_type = MDC_ACQUISITION_TOMO;
						   fi->planar = MDC_NO;
                          break;
        case MDC_INTF_GATED      :
                          fi->acquisition_type = MDC_ACQUISITION_GATED;
                          fi->planar = MDC_YES;
                          break;
        case MDC_INTF_GSPECT     :
                          fi->acquisition_type = MDC_ACQUISITION_GSPECT;
                          break;
        case MDC_INTF_DIALECT_PET:
                          fi->acquisition_type = MDC_ACQUISITION_TOMO;
                          break;
	/// Agregado por MArtin Belzunce:
	case MDC_INTF_PET:
                          fi->acquisition_type = MDC_ACQUISITION_PET;
                          break;
        case MDC_INTF_CURVE      : /* default = Static */
        case MDC_INTF_ROI        : /* default = Static */
        case MDC_INTF_STATIC     : /* default = Static */
        default                  : 
                          fi->acquisition_type = MDC_ACQUISITION_STATIC;
                          fi->planar = MDC_YES;
      }
      if (fi->acquisition_type == MDC_ACQUISITION_GATED || 
          fi->acquisition_type == MDC_ACQUISITION_GSPECT ) {
        /* MARK: limited to one AND no info on recon yet*/
        if (!MdcGetStructGD(fi,1)) { 
          return("INTF Bad malloc GATED_DATA structs");
        }else{
          gd = &fi->gdata[0];
        }
      }
      continue;
    }
    if (MdcThisKey("study date")) {
      MdcGetSplitDateKey(&fi->study_date_year
                        ,&fi->study_date_month
                        ,&fi->study_date_day);
      continue;
    }

    if (MdcThisKey("study time")) {
      MdcGetSplitTimeKey(&fi->study_time_hour
                        ,&fi->study_time_minute
                        ,&fi->study_time_second);
      continue;
    } 
    
    if (MdcThisKey("number of energy windows")) {
      intf->energy_windows = MdcGetIntKey(); continue;
    }

    if (MdcThisKey("flood corrected")) {
      fi->flood_corrected = MdcGetYesNoKey();
      continue;
    }  
    if (MdcThisKey("decay corrected")) {
      fi->decay_corrected = MdcGetYesNoKey();
      continue;
    } 

    /* read some keys without making a distinction in type of data, thus */
    /* allowing some great flexibility in reading interfile images       */
    /* ==>> pixel/voxel/slice dimensions */
    if (MdcThisKey("matrix size [1]")) {
      intf->width = MdcGetIntKey();
      if (intf->DIALECT == MDC_YES) {
        err = MdcHandleIntfDialect(fi,intf);
        if (err != NULL) return(err);
      }else{
        for (i=img; i<fi->number; i++) { /* fill the rest too */
           fi->image[i].width = intf->width;
        }
      }
      continue;
    }
    if (MdcThisKey("matrix size [2]")) {
      intf->height = MdcGetIntKey();
      if (intf->DIALECT == MDC_YES) {
        err = MdcHandleIntfDialect(fi,intf);
        if (err != NULL) return(err);
      }else{
        for (i=img; i<fi->number; i++) { /* fill the rest too */ 
           fi->image[i].height = intf->height;
        }
      }
      continue;
    }
#if MDC_INTF_SUPPORT_DIALECT
    if (MdcThisKey("number of dimensions")) {
      intf->DIALECT = MDC_YES;
      intf->dim_num = MdcGetIntKey();
      if (intf->dim_num >= MDC_MAX_DIMS) 
        return("INTF Maximum dimensions exceeded");
      fi->dim[0] = (Uint32) intf->dim_num;
      continue;
    }
    if (MdcThisKey("matrix size [3]")) {
      fi->acquisition_type = MDC_ACQUISITION_TOMO;
      if (MdcIsArrayKey()) {
        fi->dim[3] = (Int16)MdcGetMaxIntArrayKey(); /* only symmetric */
      }else{
        fi->dim[3] = (Int16)MdcGetIntKey();
      }
      intf->number_images = fi->dim[3];
      intf->images_per_dimension = intf->number_images;

      err = MdcHandleIntfDialect(fi,intf);
      if (err != NULL) return(err);
      continue;
    }
    if (MdcThisKey("matrix size [4]")) {
      matrix_size_4=MDC_TRUE;
      fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
      if (MdcIsArrayKey()) {
         fi->dim[4] = (Int16)MdcGetMaxIntArrayKey(); /* only symmetric */
      }else{
         fi->dim[4] = (Int16)MdcGetIntKey();
      }
      intf->number_images = fi->dim[4];

      err = MdcHandleIntfDialect(fi,intf);
      if (err != NULL) return(err);
      continue;
    }
    if (MdcThisKey("number of time frames")) {
      if (matrix_size_4 == MDC_FALSE) { /* prefer "matrix size [4]" if found */
        fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
        fi->dim[4] = (Int16)MdcGetIntKey();
        intf->number_images = fi->dim[4];

        intf->dim_num = 4; 
        fi->dim[0] = (Uint32) intf->dim_num;

        err = MdcHandleIntfDialect(fi,intf);
        if (err != NULL) return(err);

        if (!MdcGetStructDD(fi,fi->dim[4]))
          return("INTF Bad malloc DYNAMIC_DATA structs");

      } 
      continue;
    }
    if (MdcThisKey("matrix size [5]")) {
      fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
      if (MdcIsArrayKey()) {
         fi->dim[5] = (Int16)MdcGetMaxIntArrayKey(); /* only symmetric */
      }else{
         fi->dim[5] = (Int16)MdcGetIntKey();
      }
      intf->number_images = fi->dim[5];

      err = MdcHandleIntfDialect(fi,intf);
      if (err != NULL) return(err);
      continue;
    }
    if (MdcThisKey("matrix size [6]")) {
      fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
      if (MdcIsArrayKey()) {
         fi->dim[6] = (Int16)MdcGetMaxIntArrayKey(); /* only symmetric */
      }else{
         fi->dim[6] = (Int16)MdcGetIntKey();
      }
      intf->number_images = fi->dim[6];

      err = MdcHandleIntfDialect(fi,intf);
      if (err != NULL) return(err);
      continue;
    }
    if (MdcThisKey("matrix size [7]")) {
      fi->acquisition_type = MDC_ACQUISITION_DYNAMIC;
      if (MdcIsArrayKey()) {
         fi->dim[7] = (Int16)MdcGetMaxIntArrayKey(); /* only symmetric */
      }else{
         fi->dim[7] = (Int16)MdcGetIntKey();
      }
      intf->number_images = fi->dim[7];

      err = MdcHandleIntfDialect(fi,intf);
      if (err != NULL) return(err);
      continue;
    }
#endif
    if (MdcThisKey("number format")) {
      intf->pixel_type = MdcGetPixelType();
      for (i=img; i<fi->number; i++) { /* fill the rest too */
         fi->image[i].type = intf->pixel_type;
         fi->image[i].bits = MdcType2Bits(fi->image[i].type);
      }
      continue;
    }
    if (MdcThisKey("number of bytes per pixel")) {
      intf->pixel_type = MdcSpecifyPixelType(intf); 
      for (i=img; i<fi->number; i++) { /* fill the rest too */
         fi->image[i].type = intf->pixel_type;
         fi->image[i].bits = MdcType2Bits(fi->image[i].type);
      }
      continue;
    }
    if (MdcThisKey("scaling factor (mm/pixel) [1]")) {
      intf->pixel_xsize = (float)MdcGetFloatKey();
      continue;
    }
    if (MdcThisKey("scaling factor (mm/pixel) [2]")) {
      intf->pixel_ysize = (float)MdcGetFloatKey();
      continue;
    }
#if MDC_INTF_SUPPORT_DIALECT
    if (MdcThisKey("scaling factor (mm/pixel) [3]")) {
      intf->slice_thickness_mm = (float)MdcGetFloatKey();
      fi->pixdim[0] = 3.;
      fi->pixdim[3] = intf->slice_thickness_mm;
      continue;
    }
#endif
    if (MdcThisKey("slice thickness (pixels)")) {
      intf->slice_thickness = MdcGetFloatKey();
      continue;
    }
    if (MdcThisKey("centre-centre slice separation (pixels)")) {
      intf->centre_centre_separation = MdcGetFloatKey();
      continue;
    }
    if (MdcThisKey("center-center slice separation (pixels)")) {
      intf->centre_centre_separation = MdcGetFloatKey();
      continue;
    }
    /* ==>> slice/patient orientations */
    if (MdcThisKey("slice orientation")) {
      intf->slice_orient = MdcGetSliceOrient();
      continue;
    }
    if (MdcThisKey("patient rotation")) {
      intf->patient_rot = MdcGetPatRotation();
      continue;
    }
    if (MdcThisKey("patient orientation")) {
      intf->patient_orient = MdcGetPatOrientation();
      continue;
    } 
#if MDC_INTF_SUPPORT_SCALE
    /* some global scale factors */
    if (MdcThisKey("quantification units")) {  /* mediman */
      v = (float)MdcGetFloatKey();
      if (v != 0.0) intf->rescale_slope = v;
      continue;  
    }
    if (MdcThisKey("rescale slope")) {     /* NUD */
      v = (float)MdcGetFloatKey();
      if (v != 0.0) intf->rescale_slope = v;
      continue;
    }
    if (MdcThisKey("rescale intercept")) { /* NUD */
      v = (float)MdcGetFloatKey();
      if (v != 0.0) intf->rescale_intercept = v;
      continue;
    }
#endif

    /* now make a distinction between each type of data */
    switch (intf->data_type) {
      case MDC_INTF_STATIC:
      case MDC_INTF_ROI:

          if (img < fi->number) sd = fi->image[img].sdata;

          if (MdcThisKey("static study (general)")) {
            if (!MdcGetStructSD(fi,fi->number))
              return("INTF Couldn't malloc STATIC_DATA structs");
            continue;
          }
          if (MdcThisKey("image number")) {
            img = MdcGetIntKey() - 1;
            continue;
          }
          if (MdcThisKey("number of images/energy window")) {
            intf->number_images = MdcGetIntKey();
            intf->images_per_dimension = intf->number_images;
            continue;
          }

          /* place to store static data info */
          if (sd != NULL) {
            if (MdcThisKey("label")) {
              MdcGetStrKey(sd->label);
              continue;
            }
            if (MdcThisKey("image duration (sec)")) {
              sd->image_duration = (float)MdcGetFloatKey() * 1000.;
              continue;
            }
            if (MdcThisKey("image start time")) {
              MdcGetSplitTimeKey(&sd->start_time_hour
                                ,&sd->start_time_minute
                                ,&sd->start_time_second);
              continue;
            } 
          }

          if ( MdcThisKey("static study (each frame)")
            || MdcThisKey("end of interfile"))
          if (img < fi->number) {
            id = &fi->image[img];
            id->type = intf->pixel_type;
            id->bits = MdcType2Bits(id->type);
            id->width = intf->width;
            id->height = intf->height;
            id->pixel_xsize = intf->pixel_xsize;
            id->pixel_ysize = intf->pixel_ysize;
          }

        break;

      case MDC_INTF_DIALECT_PET: /* probably GE vendor specific */

          if (MdcThisString("image duration (sec)")) {
            intf->image_duration=MdcGetFloatKey() * 1000.;
            if ((fi->dyndata != NULL) && (counter < fi->dim[4])) {
              dd = &(fi->dyndata[counter]);
              counter++;
              if (dd != NULL) {
                dd->time_frame_duration = intf->image_duration;
                dd->nr_of_slices = fi->dim[3];
              }
            }
            continue;
          }

          /* write info to all images in frame group # */
          if ( MdcThisKey("frame group number") 
            || MdcThisKey("end of interfile"))
          if ((counter > 0) && (counter <= total) && (img < fi->number)) { 
            for (i=0; i<intf->number_images; i++, img++) {
               if (i == fi->number) break;
               id = &fi->image[img];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;
            }

            intf->number_images=0;
            /* fill the rest too */
            for (i=img; i<fi->number; i++) {
               id = &fi->image[i];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

            }

            /* fix some DIALECT settings */
            if (intf->DIALECT == MDC_YES) {
              fi->planar = MDC_NO;
              if (dd != NULL) dd->nr_of_slices = intf->images_per_dimension;
            }
          }
        break;

      case MDC_INTF_DYNAMIC:
          if (MdcThisKey("number of frame groups")) {
            intf->frame_groups = MdcGetIntKey(); 
            total = intf->frame_groups * intf->energy_windows;
            if (total == 0) {
              MdcPrntWarn("INTF Found zero frame groups (fixed = 1)");
              total = 1;
            }
            if (!MdcGetStructDD(fi,total))
              return("INTF Bad malloc DYNAMIC_DATA structs");
            continue;
          }
          if (MdcThisKey("frame group number")) {
            counter = MdcGetIntKey();
            if ((counter > 0)&&(counter <= fi->dynnr)&&(fi->dyndata != NULL))
              dd = &fi->dyndata[counter - 1];
          }else{
           if (MdcThisKey("number of images this frame group")) {
             intf->number_images = MdcGetIntKey();
             if (intf->DIALECT == MDC_NO)
               intf->images_per_dimension = intf->number_images;
             if (dd != NULL)
               dd->nr_of_slices = intf->number_images;
             continue;
           }
           if (MdcThisKey("image duration (sec)")) {
             intf->image_duration=MdcGetFloatKey() * 1000.;
             if (dd != NULL) {
               float duration;
               duration = intf->image_duration * dd->nr_of_slices;
               dd->time_frame_duration += duration;
             }
             continue;
           }
           if (MdcThisKey("pause between images (sec)")) {
             intf->image_pause=MdcGetFloatKey() * 1000.;
             if (dd != NULL) {
               dd->delay_slices = intf->image_pause;
               dd->time_frame_duration += dd->delay_slices*(dd->nr_of_slices-1);
             }
             continue;
           }
           if (MdcThisKey("pause between frame groups (sec)")) {
             intf->group_pause=MdcGetFloatKey() * 1000.;
             if (dd != NULL) {
               dd->time_frame_delay = intf->group_pause;
             }
             continue;
           }
          }
          /* write info to all images in frame group # */
          if ( MdcThisKey("frame group number") 
            || MdcThisKey("end of interfile"))
          if ((counter > 0) && (counter <= total) && (img < fi->number)) { 
            for (i=0; i<intf->number_images; i++, img++) {
               if (i == fi->number) break;
               id = &fi->image[img];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;
            }

            intf->number_images=0;
            /* fill the rest too */
            for (i=img; i<fi->number; i++) {
               id = &fi->image[i];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

            }

            /* fix some DIALECT settings */
            if (intf->DIALECT == MDC_YES) {
              fi->planar = MDC_NO;
              if (dd != NULL) dd->nr_of_slices = intf->images_per_dimension;
            }
          }
        break;

      case MDC_INTF_GATED:
          if (MdcThisKey("study duration (acquired) sec")) {
            gd->study_duration = (float)MdcGetFloatKey() * 1000.;
            continue;
          }

          if (MdcThisKey("number of cardiac cycles (observed)")) {
            /* MARK: for entire energy window */
            gd->cycles_observed = MdcGetFloatKey();
            continue;
          }

          if (MdcThisKey("number of time windows")) {
            intf->time_windows = MdcGetIntKey();
            total = intf->time_windows * intf->energy_windows;
            continue;
          }
          if (MdcThisKey("time window number")) {
            counter = MdcGetIntKey();
          }else{ 
           if (MdcThisKey("number of images in time window")) {
             intf->number_images = MdcGetIntKey();
             intf->images_per_dimension = intf->number_images;
             continue;
           }
           if (gd!=NULL && MdcThisKey("image duration (sec)")) {
             gd->image_duration = (float)MdcGetFloatKey() * 1000.;
             continue;
           }
           if (gd!=NULL && MdcThisKey("time window lower limit (sec)")) {
             gd->window_low = (float)MdcGetFloatKey() * 1000.;
             continue;
           }
           if (gd!=NULL && MdcThisKey("time window upper limit (sec)")) {
             gd->window_high = (float)MdcGetFloatKey() * 1000.;
             continue;
           }
           if (gd!=NULL
               && MdcThisKey("R-R cycles acquired this window")) {
             v = (float)MdcGetFloatKey(); if (v > 100. || v <= 0.) v = 100.;
             intf->procent_cycles_acquired = v;

             /* calculate observed */
             v = (gd->cycles_acquired * 100.) / intf->procent_cycles_acquired;
             uv = (Uint32)v; v = (float)uv; /* simply chop to integer */
             if ((v > gd->cycles_acquired) || (gd->cycles_observed == 0.)) {
               gd->cycles_observed = v;
             }
             continue;
           }
           if (gd!=NULL 
               && MdcThisKey("number of cardiac cycles (acquired)")) { 
             gd->cycles_acquired = (float)MdcGetFloatKey();

             /* calculate observed */
             v = (gd->cycles_acquired * 100.) / intf->procent_cycles_acquired;
             uv = (Uint32)v; v = (float)uv; /* simply chop to integer */
             if ((v > gd->cycles_acquired) || (gd->cycles_observed == 0.)) {
               gd->cycles_observed = v;
             } 
             continue;
           }
          }
          if ( MdcThisKey("time window number")
            || MdcThisKey("end of interfile"))
          if ((counter > 0)  && (counter <= total) && (img < fi->number)) {
            for (i=0; i<intf->number_images; i++, img++) {
               if (i == fi->number) break;
               id = &fi->image[img];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;
            }

            intf->number_images=0;
            /* fill in the rest too */
            for (i=img; i<fi->number; i++) {
               id = &fi->image[i];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;
            }

          }
        break;

      case MDC_INTF_TOMOGRAPH:
          if (MdcThisKey("number of detector heads")) {
            intf->detector_heads = MdcGetIntKey();
            total = intf->detector_heads * intf->energy_windows;
            continue;
          }
          if (MdcThisKey("process status")) {
            intf->process_status = MdcGetProcessStatus();
            switch (intf->process_status) {
              case MDC_INTF_ACQUIRED      : 
                  fi->reconstructed = MDC_NO;
                  acqnr = intf->detector_heads * intf->energy_windows;
                  if (acqnr == 0) {
                    MdcPrntWarn("INTF Requesting zero ACQ_DATA (fixed = 1)");
                    acqnr = 1;
                  }
                  if (!MdcGetStructAD(fi,acqnr))
                    return("INTF Couldn't malloc ACQ_DATA structs");
                  break;
              case MDC_INTF_RECONSTRUCTED : 
                  fi->reconstructed = MDC_YES;
                  break;
              default            : fi->reconstructed = MDC_YES;
            }
            if (fi->reconstructed == MDC_YES) {
              if (total == 0) {
                MdcPrntWarn("INTF Requesting zero DYNAMIC_DATA (fixed = 1)");
                total = 1;
              }
              if (!MdcGetStructDD(fi,total))
                return("INTF Couldn't malloc DYNAMIC_DATA structs");
            }
            continue;
          }
          if (MdcThisKey("number of projections")) {
            intf->number_images = MdcGetIntKey();
            intf->images_per_dimension = intf->number_images;
            continue;
          }
          if (MdcThisKey("extent of rotation")) {
            intf->ext_rot = (float)MdcGetFloatKey();
            continue;
          }
          if (MdcThisKey("study duration (sec)") ||          /* official */
              MdcThisKey("study duration (elapsed) sec")) {  /* dialects */
            intf->study_duration = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          switch (intf->process_status) {
            case MDC_INTF_ACQUIRED:
                if (MdcThisKey("spect study (acquired data)")) {
                  if (fi->acqnr > counter && fi->acqdata != NULL) {
                    acq = &fi->acqdata[counter];
                    acq->scan_arc = intf->ext_rot;
                    if (intf->number_images > 0)
                      acq->angle_step=intf->ext_rot/(float)intf->number_images;
                  }else{
                    acq = NULL;
                  }
                  counter += 1;
                  continue;
                }
                if (MdcThisKey("direction of rotation")) {
                  if (acq != NULL) {
                    acq->rotation_direction = (Int16)MdcGetRotation();
                  }
                  continue;
                }
                if (MdcThisKey("acquisition mode")) {
                  if (acq != NULL) {
                    acq->detector_motion = (Int16)MdcGetMotion();
                  }
                  continue;
                }
                if (MdcThisKey("start angle")) {
                  if (acq != NULL) {
                    acq->angle_start = (float)MdcGetFloatKey();
                  }
                  continue;
                }
                if (MdcThisKey("x_offset")) {
                  if (acq != NULL) {
                    acq->rotation_offset = (float)MdcGetFloatKey();
                  }
                  continue;
                }
                if (MdcThisKey("radius")) {
                  if (acq != NULL) {
                    acq->radial_position = (float)MdcGetFloatKey();
                  } 
                  continue;
                }
                break;
            case MDC_INTF_RECONSTRUCTED:
                if (MdcThisKey("method of reconstruction")) {
                  MdcGetStrKey(fi->recon_method);
                  continue;
                }
                if (MdcThisKey("number of slices")) {
                  intf->number_images = MdcGetIntKey();
                  intf->images_per_dimension = intf->number_images;
                  continue;
                }
                if (MdcThisKey("filter name")) {
                  MdcGetStrKey(fi->filter_type);
                  continue;
                }
                if (MdcThisKey("spect study (reconstructed data)")) {
                  /* fill in dynamic data (time) */
                  if ((counter < fi->dynnr) && (fi->dyndata != NULL)) {
                    dd = &fi->dyndata[counter];
                    dd->nr_of_slices = intf->number_images;
                    dd->time_frame_duration = intf->study_duration;
                  }
                  counter += 1;
                }
                break;
          }
          if ((MdcThisKey("spect study (acquired data)") 
                         && (counter>1))  ||
              (MdcThisKey("spect study (reconstructed data)") 
                         && (counter>1))  ||
              (MdcThisKey("end of interfile")))
          if (counter <= total  && img < fi->number) {
            for (i=0; i<intf->number_images; i++, img++) {
               if (i == fi->number) break;
               id = &fi->image[img];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

               id->slice_width  = ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->slice_thickness;

               fi->pat_slice_orient = MdcGetPatSlOrient(intf);
               id->slice_spacing= ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->centre_centre_separation;


               MdcFillImgPos(fi,i,i,0.0);
               MdcFillImgOrient(fi,i);
            }
            intf->number_images = 0;
            /* fill in the rest too */
            for (i=img; i<fi->number; i++) {
               id = &fi->image[i];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

               id->slice_width = ((id->pixel_xsize + id->pixel_ysize)/2.)
                                 * intf->slice_thickness;

               fi->pat_slice_orient = MdcGetPatSlOrient(intf);
               id->slice_spacing= ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->centre_centre_separation;

               MdcFillImgPos(fi,i,i,0.0);
               MdcFillImgOrient(fi,i);

            }

            /* set number of slices in dynamic data  */
            if (dd != NULL) dd->nr_of_slices = intf->images_per_dimension;

          }
        break;

      case MDC_INTF_GSPECT: /* mixture of GATED and TOMOGRAPH */

          /* GATED related stuff */
          if (gd!=NULL && MdcThisKey("gated spect nesting outer level")) {
            gd->gspect_nesting = MdcGetGSpectNesting();
            continue;
          }
          if (gd!=NULL && MdcThisKey("study duration (acquired) sec")) {
            gd->study_duration = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          if (gd!=NULL && MdcThisKey("study duration (elapsed) sec")) {
            gd->study_duration = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          if (gd!=NULL
              && MdcThisKey("number of cardiac cycles (observed)")) {
            gd->cycles_observed = (float)MdcGetFloatKey();
            continue;
          }
          if (MdcThisKey("number of time windows")) {
            intf->time_windows = MdcGetIntKey();
            continue;
          }
          /* MARK: we don't use because it mostly results in dim confusion
          if (MdcThisKey("time window number")) {
            counter = MdcGetIntKey();
            continue;
          }
          */
          /* only support for SYMMETRIC dimensions    */
          /* note different interpretation than Gated */
          if (MdcThisKey("number of images in time window")) {
             intf->time_slots = MdcGetIntKey();
             continue;
          }
          if (gd!=NULL && MdcThisKey("image duration (sec)")) {
            gd->image_duration = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          if (gd!=NULL && MdcThisKey("time window lower limit (sec)")) {
            gd->window_low = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          if (gd!=NULL && MdcThisKey("time window upper limit (sec)")) {
            gd->window_high = (float)MdcGetFloatKey() * 1000.;
            continue;
          }
          if (gd!=NULL
              && MdcThisKey("R-R cycles acquired this window")) {
            v = (float)MdcGetFloatKey(); if (v > 100. || v <= 0.) v = 100.;
            intf->procent_cycles_acquired = v;

            /* calculate observed */
            v = (gd->cycles_acquired * 100.) / intf->procent_cycles_acquired;
            uv = (Uint32)v; v = (float)uv; /* simply chop to integer */
            if ((v > gd->cycles_acquired) || (gd->cycles_observed == 0.)) {
              gd->cycles_observed = v;
            } 
            continue;
          }
          if (gd!=NULL
              && MdcThisKey("number of cardiac cycles (acquired)")) {
            gd->cycles_acquired = (float)MdcGetFloatKey();

            /* calculate observed */
            v = (gd->cycles_acquired * 100.) / intf->procent_cycles_acquired;
            uv = (Uint32)v; v = (float)uv; /* simply chop to integer */
            if ((v > gd->cycles_acquired) || (gd->cycles_observed == 0.)) {
              gd->cycles_observed = v;
            } 
            continue;
          }
          /* TOMOGRAPH related stuff */
          if (MdcThisKey("number of detector heads")) {
            /* MARK: we don't use because it mostly results in dim confusion
            intf->detector_heads = MdcGetIntKey();
            */
            total = intf->time_slots*intf->detector_heads*intf->energy_windows;
            continue;
          }
          if (MdcThisKey("process status")) {
            intf->process_status = MdcGetProcessStatus();
            switch (intf->process_status) {
              case MDC_INTF_ACQUIRED      : 
                  fi->reconstructed = MDC_NO;
                  acqnr = intf->detector_heads * intf->energy_windows;
                  if (!MdcGetStructAD(fi,acqnr))
                    return("INTF Couldn't malloc ACQ_DATA structs");
                  break;
              case MDC_INTF_RECONSTRUCTED : 
                  fi->reconstructed = MDC_YES;
                  break;
              default            : fi->reconstructed = MDC_YES;
            }
            continue;
          }
          if (MdcThisKey("number of projections")) {
            intf->number_images = MdcGetIntKey();
            intf->images_per_dimension = intf->number_images;
            if (gd != NULL) gd->nr_projections= (float)intf->number_images;
            continue;
          }
          if (MdcThisKey("extent of rotation")) {
            intf->ext_rot = (float)MdcGetFloatKey();
            if (gd != NULL) gd->extent_rotation = intf->ext_rot;
            continue;
          }
          if (MdcThisKey("time per projection (sec)")) {
            gd->time_per_proj = (float)MdcGetFloatKey() * 1000.; 
            continue;
          }
          switch (intf->process_status) {
            case MDC_INTF_ACQUIRED:
                if (MdcThisKey("spect study (acquired data)")) {
                  if (fi->acqnr > counter && fi->acqdata != NULL) {
                    acq = &fi->acqdata[counter];
                    acq->scan_arc = intf->ext_rot;
                    if (intf->number_images > 0)
                      acq->angle_step=intf->ext_rot/(float)intf->number_images;
                  }else{
                    acq = NULL;
                  }
                  counter += 1;
                  continue;
                }
                if (MdcThisKey("direction of rotation")) {
                  if (acq != NULL) {
                    acq->rotation_direction = (Int16)MdcGetRotation();
                  }
                  continue;
                }
                if (MdcThisKey("acquisition mode")) {
                  if (acq != NULL) {
                    acq->detector_motion = (Int16)MdcGetMotion();
                  }
                  continue;
                }
                if (MdcThisKey("start angle")) {
                  if (acq != NULL) {
                    acq->angle_start = (float)MdcGetFloatKey();
                  }
                  continue;
                }
                if (MdcThisKey("x_offset")) {
                  if (acq != NULL) {
                    acq->rotation_offset = (float)MdcGetFloatKey();
                  }
                  continue;
                }
                if (MdcThisKey("radius")) {
                  if (acq != NULL) {
                    acq->radial_position = (float)MdcGetFloatKey();
                  } 
                  continue;
                }
                break;
            case MDC_INTF_RECONSTRUCTED:
                if (MdcThisKey("method of reconstruction")) {
                  MdcGetStrKey(fi->recon_method);
                  continue;
                }
                if (MdcThisKey("number of slices")) {
                  intf->number_images = MdcGetIntKey();
                  intf->images_per_dimension = intf->number_images;
                  continue;
                }
                if (MdcThisKey("filter name")) {
                  MdcGetStrKey(fi->filter_type);
                  continue;
                }
                if (MdcThisKey("spect study (reconstructed data)")) {
                  counter += 1;
                }
                break;
          }
          if ((MdcThisKey("spect study (acquired data)") 
                         && (counter>1))  ||
              (MdcThisKey("spect study (reconstructed data)") 
                         && (counter>1))  ||
              (MdcThisKey("end of interfile")))
          if (counter <= total && img < fi->number) {
            for (i=0; i<intf->number_images; i++, img++) {
               if (i == fi->number) break;
               id = &fi->image[img];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

               id->slice_width  = ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->slice_thickness;

               fi->pat_slice_orient = MdcGetPatSlOrient(intf);
               id->slice_spacing= ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->centre_centre_separation;

               MdcFillImgPos(fi,i,i,0.0);
               MdcFillImgOrient(fi,i);
            }
            intf->number_images = 0;
            /* fill in the rest too */
            for (i=img; i<fi->number; i++) {
               id = &fi->image[i];
               id->type = intf->pixel_type;
               id->bits = MdcType2Bits(id->type);
               id->width = intf->width;
               id->height = intf->height;
               id->pixel_xsize = intf->pixel_xsize;
               id->pixel_ysize = intf->pixel_ysize;

               id->slice_width = ((id->pixel_xsize + id->pixel_ysize)/2.)
                                 * intf->slice_thickness;

               fi->pat_slice_orient = MdcGetPatSlOrient(intf);
               id->slice_spacing= ((id->pixel_xsize + id->pixel_ysize)/2.)
                                  * intf->centre_centre_separation;

               MdcFillImgPos(fi,i,i,0.0);
               MdcFillImgOrient(fi,i);
            } 

          }

          break;

      case MDC_INTF_CURVE:
          MdcCloseFile(fi->ifp);
          return("INTF Curve data not supported");
          break;

    }

    if (MdcThisKey("end of interfile")) break;
  }

  if (MDC_INFO) {
    MdcPrntScrn("\n");
    MdcPrintLine('-',MDC_HALF_LENGTH);
  }

  /* safety check or for dialect without "number of dimensions" key */
  if ((fi->image == NULL) || (fi->number == 0))
    return("INTF Failure to decipher header information");

  return NULL;
}

char *MdcReadIntfImages(FILEINFO *fi, MDC_INTERFILE *intf)
{
  IMG_DATA *id;
  Uint32 i, p, bytes, nbr;
  char *err;

  /* set FILE pointer to begin of data */
  if (intf->data_offset > 0L) 
    fseek(fi->ifp,(signed)intf->data_offset,SEEK_SET);

  for (i=0; i<fi->number; i++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);

     id = &fi->image[i];
     bytes = id->width * id->height * MdcType2Bytes(id->type);
     if ( (id->buf = MdcGetImgBuffer(bytes)) == NULL)
       return("INTF Bad malloc image buffer");
     switch(id->type) {
      case  BIT1: /* convert directly to BIT8_U */
          {
            bytes = MdcPixels2Bytes(id->width * id->height);
            if (fread(id->buf,1,bytes,fi->ifp) != bytes) {
              err=MdcHandleTruncated(fi,i+1,MDC_YES);
              if (err != NULL) return(err);
            }
            MdcMakeBIT8_U(id->buf, fi, i);
            id->type = BIT8_U; 
          }
          break;

      case ASCII: 
          {
            double *pix = (double *)id->buf;

            for (p=0; p<(id->width*id->height); p++) {
               fscanf(fi->ifp,"%le",&pix[p]);
               if (ferror(fi->ifp)) {
                 err=MdcHandleTruncated(fi,i+1,MDC_YES);
                 if (err != NULL) return(err);
                 break;
               } 
            } 
            id->type = FLT64; MDC_FILE_ENDIAN = MDC_HOST_ENDIAN;
          }
          break;

        default: 
          if ((nbr=fread(id->buf,1,bytes,fi->ifp)) != bytes) { 
              if (nbr > 0) err=MdcHandleTruncated(fi,i+1,MDC_YES);
              else err=MdcHandleTruncated(fi,i,MDC_YES);
              if (err != NULL) return(err);
          } 
     } 
     if (fi->truncated) break;
  }

  return NULL;

}


const char *MdcReadINTF(FILEINFO *fi)
{
  MDC_INTERFILE intf;
  IMG_DATA *id;
  const char *err;
  char *origpath=NULL;
  Uint32 i, check=1;
  Int8 WAS_COMPRESSED=MDC_NO;

  /* put in some defaults */
  fi->endian = MDC_FILE_ENDIAN = MDC_BIG_ENDIAN;
  fi->flood_corrected = MDC_YES;
  fi->decay_corrected = MDC_NO;
  fi->reconstructed = MDC_YES;
 
  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Reading InterFile:");

  if (MDC_VERBOSE) MdcPrntMesg("INTF Reading <%s> ...",fi->ifname);

  /* preserve original input path - before header read fills fi->ipath */
  MdcMergePath(fi->ipath,fi->idir,fi->ifname);
  if ((origpath=malloc(strlen(fi->ipath) + 1)) == NULL) {
    return("INTF Couldn't allocate original path");
  }
  strcpy(origpath,fi->ipath);
  MdcSplitPath(fi->ipath,fi->idir,fi->ifname);

  /* initialize intf struct */
  MdcInitIntf(&intf);

  /* read the header */
  err=MdcReadIntfHeader(fi, &intf);
  if (err != NULL) { MdcFree(origpath); return(err); }

  if (MDC_ECHO_ALIAS == MDC_YES) {
    MdcEchoAliasName(fi); return(NULL);
  }

  MdcCloseFile(fi->ifp);

  /* complete FILEINFO stuct */
  fi->type  = intf.pixel_type;
  fi->bits  = MdcType2Bits(fi->type);

  if (intf.DIALECT == MDC_YES) {
    for (i=0; i<fi->number; i++) {
       id = &fi->image[i];
       id->type = intf.pixel_type;
       id->bits = MdcType2Bits(id->type);
       id->width = intf.width;
       id->height= intf.height;
       id->pixel_xsize = intf.pixel_xsize;
       id->pixel_ysize = intf.pixel_ysize;
       id->slice_width   = intf.slice_thickness_mm;
       id->slice_spacing = intf.slice_thickness_mm;
    }
  }else{

    fi->dim[3] = intf.images_per_dimension;
    fi->dim[7] = intf.energy_windows;

    switch (intf.data_type) {
      case MDC_INTF_DYNAMIC:
          fi->dim[4] = intf.frame_groups;
          break;
      case MDC_INTF_TOMOGRAPH:
          fi->dim[6] = intf.detector_heads;
          break;
      case MDC_INTF_GATED:
          fi->dim[5] = intf.time_windows;
          break;
      case MDC_INTF_GSPECT:
          fi->dim[4] = intf.time_slots;
          fi->dim[5] = intf.time_windows;   /* MARK: fishy, don't really know */
          fi->dim[6] = intf.detector_heads; /* MARK: fishy, don't really know */
          break;
      /// Agregado por Martin Belzunce
      case MDC_INTF_DIALECT_PET:
          fi->dim[1] = intf.width;
          fi->dim[2] = intf.height;   
          fi->dim[3] = intf.number_images; 
		  fi->pixdim[1] = intf.pixel_xsize;
		  fi->pixdim[2] = intf.pixel_ysize;
		  fi->pixdim[3] = intf.slice_thickness_mm;
		  for (i=0; i<fi->number; i++) {
			id = &fi->image[i];
			id->type = intf.pixel_type;
			id->bits = MdcType2Bits(id->type);
			id->width = intf.width;
			id->height= intf.height;
			id->pixel_xsize = intf.pixel_xsize;
			id->pixel_ysize = intf.pixel_ysize;
			id->slice_width   = intf.slice_thickness_mm;
			id->slice_spacing = intf.slice_thickness_mm;
		  }
		  break;
	  /// Agregado tambien por MArtin Belzunce, es una chanchada porque el dialect debería ir arriba por lo que entiendo:
      case MDC_INTF_PET:
          fi->dim[1] = intf.width;
          fi->dim[2] = intf.height;   
          fi->dim[3] = intf.number_images; 
		  fi->pixdim[1] = intf.pixel_xsize;
		  fi->pixdim[2] = intf.pixel_ysize;
		  fi->pixdim[3] = intf.slice_thickness_mm;
		  for (i=0; i<fi->number; i++) {
			id = &fi->image[i];
			id->type = intf.pixel_type;
			id->bits = MdcType2Bits(id->type);
			id->width = intf.width;
			id->height= intf.height;
			id->pixel_xsize = intf.pixel_xsize;
			id->pixel_ysize = intf.pixel_ysize;
			id->slice_width   = intf.slice_thickness_mm;
			id->slice_spacing = intf.slice_thickness_mm;
		  }
		  break;
    }
  }

  for (i=(MDC_MAX_DIMS-1); i>3; i--) if (fi->dim[i] > 1) break;
  fi->dim[0] = i;

  /* check fi->dim[] integrity */
  for (i=(MDC_MAX_DIMS-1); i>2; i--) check*=fi->dim[i];
  if ((fi->number > 1)  && (fi->number - 1) == check) {
    /* probably an ugly preview slice included */
    if (MDC_SKIP_PREVIEW == MDC_YES) { 
      intf.data_offset += intf.width*intf.height*MdcType2Bytes(intf.pixel_type);
      fi->number -= 1;
    }else{
      MdcPrntWarn("INTF Probably with confusing preview slice");
    }
  }

  /* make one dimensional when planar or asymmetric tomo */
  if ((check != fi->number) || (fi->planar == MDC_YES)) { 
    if (fi->planar == MDC_NO) {
      if (fi->dim[0] == 3) {
        /* bad total images defined          */
        /* sometimes for reconstructed TOMO  */
        /* the <number of slices> key was not*/
        /* filled in properly                */

        /* we DO issue a warning ...         */
        MdcPrntWarn("INTF Confusing number of images specified");
      }else{
        /* damn, an asymmetric amount of images per     */
        /* dimension which is unsupported for tomo (3D) */
        MdcPrntWarn("INTF Garbled or unsupported images/dimension:\n" \
                    "\t - using one dimensional array\n"       \
                    "\t - image position values might be corrupted");

        intf.data_type = MDC_INTF_TOMOGRAPH; /* disable dynamic etc ... */
 
      }
    }

    /* fix the dimensions */
    fi->dim[0] = 3;
    fi->dim[3] = fi->number;
    for (i=4; i<MDC_MAX_DIMS; i++) fi->dim[i] = 1;

  }

  fi->pixdim[0] = 3.;
  if (fi->image[0].pixel_xsize == 0. ) fi->pixdim[1]=1.;
  else fi->pixdim[1] = fi->image[0].pixel_xsize;
  if (fi->image[0].pixel_ysize == 0. ) fi->pixdim[2]=1.;
  else fi->pixdim[2] = fi->image[0].pixel_ysize;
  /// Modificado por Martin Belzunce
  if ((fi->pixdim[0] == 0)&&(fi->image[0].slice_width == 0))
    fi->pixdim[3]= (fi->pixdim[1] + fi->pixdim[2]) / 2. ;
  else 
    fi->pixdim[3]=fi->image[0].slice_width;

  /* loop final time through all images */
  for (i=0; i<fi->number; i++) {
     id = &fi->image[i];
#if MDC_INTF_SUPPORT_SCALE
     /* set scale factors */
     id->quant_scale = intf.rescale_slope;
     id->intercept   = intf.rescale_intercept;
#endif
#ifndef HAVE_8BYTE_INT
     /* unavailable BIT64 type */
     if (id->type == BIT64_S || id->type == BIT64_U) {
       MdcFree(origpath); return("INTF Unsupported data type BIT64");
     }
#endif
  }

  
  MdcMergePath(fi->ipath,fi->idir,fi->ifname);

  /* check for compression */
  if (MdcWhichCompression(fi->ipath) != MDC_NO) {
    /* "name of data file" with proper .Z or .gz extension */
    if (MdcDecompressFile(fi->ipath) != MDC_OK) {
      MdcFree(origpath); return("INTF Decompression image file failed");
    }

    WAS_COMPRESSED = MDC_YES;

  }else{
    if (MdcFileExists(fi->ipath) == MDC_NO) {
      /* no uncompressed image file found           */
      /* so we look for the compressed image file   */
      /* depending on the compression of the header */
      /* => result from doing `gzip basename.*` ;-) */
      MdcAddCompressionExt(fi->compression, fi->ipath);
   
      if (MdcFileExists(fi->ipath)) { 
        if (MdcDecompressFile(fi->ipath) != MDC_OK) {
          MdcFree(origpath); return("INTF Decompression image file failed");
        }

        /* Yep, you loose if you're using a different */
        /* compression for header and image files     */

        WAS_COMPRESSED = MDC_YES;

      }else{
        /* maybe case sensitivity problem in key     */
        /* "name of data file" (DOS/Unix transition) */

        /* try all UPPER case */
        MdcSplitPath(fi->ipath,fi->idir,fi->ifname);
        MdcUpStr(fi->ifname);
        MdcMergePath(fi->ipath,fi->idir,fi->ifname);
        if (MdcFileExists(fi->ipath) == MDC_NO) {
        /* try all LOWER case */
          MdcSplitPath(fi->ipath,fi->idir,fi->ifname);
          MdcLowStr(fi->ifname);
          MdcMergePath(fi->ipath,fi->idir,fi->ifname);
          if (MdcFileExists(fi->ipath) == MDC_NO)
            return("INTF Couldn't find specified image file");
        }

        MdcPrntWarn("INTF Check upper/lower case of image file");
      }
    }
  }


  /* open the (decompressed) image file */
  if ( (fi->ifp=fopen(fi->ipath,"rb")) == NULL) {
    MdcFree(origpath); return("INTF Couldn't open image file");
  }

  if (WAS_COMPRESSED == MDC_YES) { 
    unlink(fi->ipath); /* delete after use */

    if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Reading InterFile:");
  }

  MdcSplitPath(fi->ipath,fi->idir,fi->ifname);

  err=MdcReadIntfImages(fi, &intf);
  if (err != NULL) { MdcFree(origpath); return(err); }

  MdcCloseFile(fi->ifp);

  /* restore original filename */
  strcpy(fi->ipath,origpath); 
  MdcSplitPath(fi->ipath,fi->idir,fi->ifname);
  MdcFree(origpath);

  if (fi->truncated) return("INTF Truncated image file");

  return NULL; 

}

char *MdcType2Intf(int type)
{
  switch (type) {
   case BIT1   : return("bit"); break;
   case BIT8_U :
   case BIT16_U:
   case BIT32_U:
   case BIT64_U: return("unsigned integer"); break;
   case BIT8_S :
   case BIT16_S:
   case BIT32_S:
   case BIT64_S: return("signed integer"); break; 
   case FLT32  : return("short float"); break;
   case FLT64  : return("long float"); break;
   case ASCII  : return("ASCII"); break;
  }
  return("unsigned integer");
}


char *MdcGetProgramDate(void)
{
  int date, month=0, year;

  sscanf(MDC_DATE,"%2d-%3s-%4d",&date,keystr_check,&year);

  MdcLowStr(keystr_check);

  if (     MdcThisString("jan")) month=1;
  else if (MdcThisString("feb")) month=2;
  else if (MdcThisString("mar")) month=3;
  else if (MdcThisString("apr")) month=4;
  else if (MdcThisString("may")) month=5;
  else if (MdcThisString("jun")) month=6;
  else if (MdcThisString("jul")) month=7;
  else if (MdcThisString("aug")) month=8;
  else if (MdcThisString("sep")) month=9;
  else if (MdcThisString("oct")) month=10;
  else if (MdcThisString("nov")) month=11;
  else if (MdcThisString("dec")) month=12;

  sprintf(keystr,"%04d:%02d:%02d",year,month,date);

  return(keystr);

}

char *MdcSetPatRotation(int patient_slice_orient)
{
  switch (patient_slice_orient) {
    case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
    case MDC_SUPINE_HEADFIRST_SAGITTAL   :
    case MDC_SUPINE_HEADFIRST_CORONAL    :
    case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
    case MDC_SUPINE_FEETFIRST_SAGITTAL   :
    case MDC_SUPINE_FEETFIRST_CORONAL    :
        return("supine"); break;
    case MDC_PRONE_HEADFIRST_TRANSAXIAL :
    case MDC_PRONE_HEADFIRST_SAGITTAL    :
    case MDC_PRONE_HEADFIRST_CORONAL     :
    case MDC_PRONE_FEETFIRST_TRANSAXIAL :
    case MDC_PRONE_FEETFIRST_SAGITTAL    :
    case MDC_PRONE_FEETFIRST_CORONAL     :
        return("prone"); break;
    default                              :
        return("Unknown");
  }
}

char *MdcSetPatOrientation(int patient_slice_orient)
{
  switch (patient_slice_orient) {
    case MDC_SUPINE_HEADFIRST_TRANSAXIAL:
    case MDC_SUPINE_HEADFIRST_SAGITTAL   :
    case MDC_SUPINE_HEADFIRST_CORONAL    :
    case MDC_PRONE_HEADFIRST_TRANSAXIAL :
    case MDC_PRONE_HEADFIRST_SAGITTAL    :
    case MDC_PRONE_HEADFIRST_CORONAL     :
        return("head_in"); break;
    case MDC_SUPINE_FEETFIRST_TRANSAXIAL:
    case MDC_SUPINE_FEETFIRST_SAGITTAL   :
    case MDC_SUPINE_FEETFIRST_CORONAL    :
    case MDC_PRONE_FEETFIRST_TRANSAXIAL :
    case MDC_PRONE_FEETFIRST_SAGITTAL    :
    case MDC_PRONE_FEETFIRST_CORONAL     :
        return("feet_in"); break;
    default                              :
        return("Unknown"); break;
  } 
}

char *MdcCheckIntfDim(FILEINFO *fi)
{
  int DIMENSION_WARNING = MDC_NO;

  switch (fi->acquisition_type) {
    case MDC_ACQUISITION_DYNAMIC:
        /* no support for R-R intervals and detector heads */
        if ( (fi->dim[5] > 1) || (fi->dim[6] > 1) ) {
          strcpy(mdcbufr,"INTF Unsupported dimensions used for DYNAMIC file");
          DIMENSION_WARNING = MDC_YES;
        }
        break;
    case MDC_ACQUISITION_TOMO   :
        /* no support for time slots and R-R intervals */
        if ( (fi->dim[4] > 1) || (fi->dim[5] > 1) ) {
          strcpy(mdcbufr,"INTF Unsupported dimensions used for TOMO file");
          DIMENSION_WARNING = MDC_YES;
        }
        break;
    case MDC_ACQUISITION_GATED  :
        /* no support for time slots and detector heads */
        if ( (fi->dim[4] > 1) || (fi->dim[6] > 1) ) {
          strcpy(mdcbufr,"INTF Unsupported dimensions used for GATED file");
          DIMENSION_WARNING = MDC_YES;
        }
        break;
    case MDC_ACQUISITION_GSPECT : /* uses all dimensions */
        break;
    case MDC_ACQUISITION_UNKNOWN: /* default = Static */
    case MDC_ACQUISITION_STATIC : /* default = Static */
    default                     :
        /* no support for time slots, R-R intervals, detector heads */
        if ( (fi->dim[4] > 1) || (fi->dim[5] > 1) || (fi->dim[6] > 1) ) {
          strcpy(mdcbufr,"INTF Unsupported dimensions used for STATIC file");
          DIMENSION_WARNING = MDC_YES;
        }
  }

  if (DIMENSION_WARNING == MDC_YES) {
    MdcPrntWarn(mdcbufr);
  }
     
  return(NULL);
}

char *MdcWriteGenImgData(FILEINFO *fi)
{
  FILE *fp = fi->ofp;

  fprintf(fp,";\r\n");
  fprintf(fp,"!GENERAL IMAGE DATA :=\r\n");
  fprintf(fp,"!type of data := ");
  switch (fi->acquisition_type) {
   case MDC_ACQUISITION_DYNAMIC: fprintf(fp,"Dynamic\r\n");     break;
   case MDC_ACQUISITION_TOMO   : fprintf(fp,"Tomographic\r\n"); break;
   case MDC_ACQUISITION_GATED  : fprintf(fp,"Gated\r\n");       break;
   case MDC_ACQUISITION_GSPECT : fprintf(fp,"GSPECT\r\n");      break;
   /// Agregado por Martin Belzunce:
   case MDC_ACQUISITION_PET : fprintf(fp,"pet\r\n");      break; /// Reemplazo PET por tomgraphic porque no me reconoce ese campo el matlab.
   case MDC_ACQUISITION_UNKNOWN: /* default = Static */
   case MDC_ACQUISITION_STATIC : /* default = Static */
   default                     : fprintf(fp,"Static\r\n");
  }
  fprintf(fp,"!total number of images := %u\r\n",fi->number);
  fprintf(fp,"study date := %04d:%02d:%02d\r\n",fi->study_date_year
                                             ,fi->study_date_month
                                             ,fi->study_date_day);
  fprintf(fp,"study time := %02d:%02d:%02d\r\n",fi->study_time_hour
                                             ,fi->study_time_minute
                                             ,fi->study_time_second);
  fprintf(fp,"imagedata byte order := ");  
  if (MDC_FILE_ENDIAN == MDC_LITTLE_ENDIAN)
    fprintf(fp,"LITTLEENDIAN\r\n");
  else 
    fprintf(fp,"BIGENDIAN\r\n");

  fprintf(fp,"process label := %s\r\n",fi->study_descr);

#if MDC_INTF_SUPPORT_SCALE
  if (fi->image[0].rescaled) {
    /* write global scales */
    fprintf(fp,";\r\n");
    fprintf(fp,"quantification units := %+e\r\n"
              ,fi->image[0].rescaled_fctr);
    fprintf(fp,"NUD/rescale slope := %+e\r\n"
              ,fi->image[0].rescaled_slope);
    fprintf(fp,"NUD/rescale intercept := %+e\r\n"
              ,fi->image[0].rescaled_intercept);
  }
#endif
  return(NULL);

}

char *MdcWriteMatrixInfo(FILEINFO *fi, Uint32 img)
{
  IMG_DATA *id = &fi->image[img];
  FILE *fp = fi->ofp;

  fprintf(fp,"!matrix size [1] := %u\r\n",id->width);
  fprintf(fp,"!matrix size [2] := %u\r\n",id->height);

  if (MDC_FORCE_INT != MDC_NO) {
    switch (MDC_FORCE_INT) {
      case BIT8_U :
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT8_U));
          fprintf(fp,"!number of bytes per pixel := %u\r\n"
                                               ,MdcType2Bytes(BIT8_U));
          break;
      case BIT16_S:
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT16_S));
          fprintf(fp,"!number of bytes per pixel := %u\r\n"
                                               ,MdcType2Bytes(BIT16_S));
          break;
      default     :
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT16_S));
          fprintf(fp,"!number of bytes per pixel := %u\r\n",
                                                MdcType2Bytes(BIT16_S));
    }
  }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
          fprintf(fp,"!number format := short float\r\n");
          fprintf(fp,"!number of bytes per pixel := 4\r\n");
  }else{
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(id->type));
          fprintf(fp,"!number of bytes per pixel := %u\r\n",
                                             MdcType2Bytes(id->type));
  }

  fprintf(fp,"scaling factor (mm/pixel) [1] := %+e\r\n",id->pixel_xsize);
  fprintf(fp,"scaling factor (mm/pixel) [2] := %+e\r\n",id->pixel_ysize);

  return (NULL);

}


char *MdcWriteWindows(FILEINFO *fi)
{ 
  Uint32 window, total_energy_windows = fi->dim[7];
  FILE *fp = fi->ofp; 
  char *msg = NULL;

  if (total_energy_windows == 0) return("INTF Bad total number of windows");

  fprintf(fp,";\r\n");
  fprintf(fp,"number of energy windows := %u\r\n",total_energy_windows);

  for (window=1; window <= total_energy_windows; window++) {

     fprintf(fp,";\r\n");
     fprintf(fp,"energy window [%u] :=\r\n",window);
     fprintf(fp,"energy window lower level [%u] :=\r\n",window);
     fprintf(fp,"energy window upper level [%u] :=\r\n",window);
     fprintf(fp,"flood corrected := ");
     if (fi->flood_corrected == MDC_YES)
       fprintf(fp,"Y\r\n");
     else
       fprintf(fp,"N\r\n");
     fprintf(fp,"decay corrected := ");
     if (fi->decay_corrected == MDC_YES)
       fprintf(fp,"Y\r\n");
     else
       fprintf(fp,"N\r\n");

     switch (fi->acquisition_type) {
       case MDC_ACQUISITION_DYNAMIC:  msg=MdcWriteIntfDynamic(fi); break;
       case MDC_ACQUISITION_TOMO   :  msg=MdcWriteIntfTomo(fi);    break;
       case MDC_ACQUISITION_GATED  :  msg=MdcWriteIntfGated(fi);   break;
       case MDC_ACQUISITION_GSPECT :  msg=MdcWriteIntfGSPECT(fi);  break;
       /// Agregado por Martin Belzunce:
       case MDC_ACQUISITION_PET :  msg=MdcWriteIntfPET(fi);  break;
       case MDC_ACQUISITION_UNKNOWN:  /* default = Static */
       case MDC_ACQUISITION_STATIC :  /* default = Static */
       default                     :  msg=MdcWriteIntfStatic(fi);
     }

     if (msg != NULL) return(msg);

  } 

  return(NULL);
}


char *MdcWriteIntfStatic(FILEINFO *fi) 
{
  Uint32 i, total_energy_windows=fi->dim[7];
  Uint32 images_per_window = fi->number/total_energy_windows;
  IMG_DATA *id = NULL;
  STATIC_DATA sdata, *sd;
  FILE *fp  = fi->ofp;
  char *msg = NULL;

  fprintf(fp,";\r\n");
  fprintf(fp,"!STATIC STUDY (General) :=\r\n");
  fprintf(fp,"number of images/energy window := %u\r\n",images_per_window);

  for (i=0; i<images_per_window; i++) {
     id = &fi->image[i]; sd = &sdata;
     if (id->sdata != NULL) {
       MdcCopySD(sd,id->sdata);
     }else{
       MdcInitSD(sd);
     }
     fprintf(fp,";\r\n");
     fprintf(fp,"!Static Study (each frame) :=\r\n");
     fprintf(fp,"!image number := %u\r\n",i+1);

     msg = MdcWriteMatrixInfo(fi, i);
     if (msg != NULL) return(msg);

     fprintf(fp,"image duration (sec) := %e\r\n",sd->image_duration / 1000.);
     fprintf(fp,"image start time := %02hd:%02hd:%02hd\r\n"
                                                ,sd->start_time_hour
                                                ,sd->start_time_minute
                                                ,sd->start_time_second);
     fprintf(fp,"label := %s\r\n",sd->label);
     if (id->rescaled) {
       fprintf(fp,"!maximum pixel count := %+e\r\n",id->rescaled_max);
       fprintf(fp,"!minimum pixel count := %+e\r\n",id->rescaled_min);
     }else{
       fprintf(fp,"!maximum pixel count := %+e\r\n",id->max);
       fprintf(fp,"!minimum pixel count := %+e\r\n",id->min);
     }

     fprintf(fp,"total counts := %g\r\n",sd->total_counts);
 
  }

  if (ferror(fp)) return("INTF Error writing Static Header");

  return(NULL);

}

/// Función agregada por Martín Belzunce. Hace lo mismo que static, pero en la info
/// general de la imagen le agrego todas las dimensiones.
char *MdcWriteIntfPET(FILEINFO *fi) 
{
  Uint32 i, total_energy_windows=fi->dim[7];
  Uint32 images_per_window = fi->number/total_energy_windows;
  IMG_DATA *id = NULL;
  STATIC_DATA sdata, *sd;
  FILE *fp  = fi->ofp;
  char *msg = NULL;
  char labels[3] = {'x', 'y', 'z'};
  
  fprintf(fp,";\r\n");
  fprintf(fp,"!PET STUDY (General) :=\r\n");
  fprintf(fp,"!process status := ");
  if (fi->reconstructed == MDC_NO) {
	  fprintf(fp,"Acquired\r\n");
  }else{
	  fprintf(fp,"Reconstructed\r\n");
  }
  fprintf(fp,"number of images/energy window := %u\r\n",images_per_window);
  /// Agregado por Martin Belzunce:
  fprintf(fp,"number of images/energy window := %u\r\n",images_per_window);
  
  for(i = 0; i < fi->dim[0]; i++)
  {
    fprintf(fp,"!matrix size [%d] := %u\r\n", i+1, fi->dim[i+1]);
    fprintf(fp,"matrix axis label [%d] := %c\r\n", i+1, labels[i]);
    fprintf(fp,"scaling factor (mm/pixel) [%d] := %f\r\n", i+1, fi->pixdim[i+1]);
  }

  if (MDC_FORCE_INT != MDC_NO) {
    switch (MDC_FORCE_INT) {
      case BIT8_U :
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT8_U));
          fprintf(fp,"!number of bytes per pixel := %u\r\n"
                                               ,MdcType2Bytes(BIT8_U));
          break;
      case BIT16_S:
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT16_S));
          fprintf(fp,"!number of bytes per pixel := %u\r\n"
                                               ,MdcType2Bytes(BIT16_S));
          break;
      default     :
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(BIT16_S));
          fprintf(fp,"!number of bytes per pixel := %u\r\n",
                                                MdcType2Bytes(BIT16_S));
    }
  }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
          fprintf(fp,"!number format := short float\r\n");
          fprintf(fp,"!number of bytes per pixel := 4\r\n");
  }else{
          fprintf(fp,"!number format := %s\r\n",MdcType2Intf(fi->type));
          fprintf(fp,"!number of bytes per pixel := %u\r\n",
                                             MdcType2Bytes(fi->type));
  }
  if(fi->reconstructed == MDC_YES)
  {
	  fprintf(fp,"!SPECT STUDY (reconstructed data) :=\r\n");
	  fprintf(fp,"method of reconstruction := %s\r\n",fi->recon_method);
	  fprintf(fp,"!number of slices := %u\r\n",fi->dim[3]);
	  fprintf(fp,"number of reference frame := 0\r\n");
	  fprintf(fp,"slice orientation := %s\r\n", MdcGetStrSliceOrient(fi->pat_slice_orient));
	  fprintf(fp,"slice thickness (pixels) := %+e\r\n",1);
	  fprintf(fp,"centre-centre slice separation (pixels) := %+e\r\n", 1);
	  fprintf(fp,"filter name := %s\r\n",fi->filter_type);
	  fprintf(fp,"filter parameters := Cutoff\r\n");
	  /*fprintf(fp,"z-axis filter :=\r\n");*/
	  /*fprintf(fp,"attenuation correction coefficient/cm :=\r\n");*/
	  fprintf(fp,"method of attenuation correction := measured\r\n");
	  fprintf(fp,"scatter corrected := N\r\n"); 
	  /*fprintf(fp,"method of scatter correction :=\r\n");*/
	  fprintf(fp,"oblique reconstruction := N\r\n");
  }
  /* Esto es si quiero toda la data por slice, en principio no sería necesario.
  for (i=0; i<images_per_window; i++) {
     id = &fi->image[i]; sd = &sdata;
     if (id->sdata != NULL) {
       MdcCopySD(sd,id->sdata);
     }else{
       MdcInitSD(sd);
     }
     fprintf(fp,";\r\n");
     fprintf(fp,"!Static Study (each frame) :=\r\n");
     fprintf(fp,"!image number := %u\r\n",i+1);

     msg = MdcWriteMatrixInfo(fi, i);
     if (msg != NULL) return(msg);

     fprintf(fp,"image duration (sec) := %e\r\n",sd->image_duration / 1000.);
     fprintf(fp,"image start time := %02hd:%02hd:%02hd\r\n"
                                                ,sd->start_time_hour
                                                ,sd->start_time_minute
                                                ,sd->start_time_second);
     fprintf(fp,"label := %s\r\n",sd->label);
     if (id->rescaled) {
       fprintf(fp,"!maximum pixel count := %+e\r\n",id->rescaled_max);
       fprintf(fp,"!minimum pixel count := %+e\r\n",id->rescaled_min);
     }else{
       fprintf(fp,"!maximum pixel count := %+e\r\n",id->max);
       fprintf(fp,"!minimum pixel count := %+e\r\n",id->min);
     }

     fprintf(fp,"total counts := %g\r\n",sd->total_counts);
 
  }*/

  if (ferror(fp)) return("INTF Error writing PET Header");

  return(NULL);

}

char *MdcWriteIntfDynamic(FILEINFO *fi)
{ 
  DYNAMIC_DATA *dd;
  Uint32 s, f, s0, img=0;
  Uint32 nrframes=1, nrslices=fi->dim[3];
  double max;
  IMG_DATA *id = NULL; 
  FILE *fp  = fi->ofp;
  char *msg = NULL;

  if ((fi->dynnr == 0) || (fi->dyndata == NULL))
    return("INTF Missing proper DYNAMIC_DATA structs");

  if (fi->diff_size == MDC_YES) 
    return("INTF Dynamic different sizes unsupported");
  if (fi->diff_type == MDC_YES)
    return("INTF Dynamic different types unsupported");

  nrframes = fi->dynnr;

  fprintf(fp,";\r\n");
  fprintf(fp,"!DYNAMIC STUDY (general) :=\r\n");
  fprintf(fp,"!number of frame groups := %u\r\n",nrframes);
  for (s0=0, f=0; f < nrframes; f++) {
     dd = &fi->dyndata[f];
     nrslices = dd->nr_of_slices;
     id = &fi->image[s0]; /* first image of current time frame */
     fprintf(fp,";\r\n");
     fprintf(fp,"!Dynamic Study (each frame group) :=\r\n");
     fprintf(fp,"!frame group number := %u\r\n",f+1);

     msg = MdcWriteMatrixInfo(fi, img);
     if (msg != NULL) return(msg);

     fprintf(fp,"!number of images this frame group := %u\r\n",nrslices);
     fprintf(fp,"!image duration (sec) := %.7g\r\n"
               ,MdcSingleImageDuration(fi,f) / 1000.);
     fprintf(fp,"pause between images (sec) := %.7g\r\n"
               ,dd->delay_slices / 1000. );
     fprintf(fp,"pause between frame groups (sec) := %.7g\r\n"
               ,dd->time_frame_delay / 1000. );

     if (id->rescaled || MDC_CALIBRATE || MDC_QUANTIFY) {
       max = id->rescaled_max;
     }else{
       max = id->max;
     }

     for (s=1; s < nrslices; s++) {
        id = &fi->image[s0 + s];
        if (id->rescaled) {
          if (id->rescaled_max > max) max = id->rescaled_max;
        }else{
          if (id->max          > max) max = id->max;
        }
     }
     fprintf(fp,"!maximum pixel count in group := %+e\r\n",max);

     s0 += dd->nr_of_slices; /* set first slice of next time frame */
  }

  if (ferror(fp)) return("INTF Error writing Dynamic Header");

  if (fi->planar == MDC_NO)
    return("INTF Inappropriate for non-planar dynamic studies");

  return(NULL);

}

char *MdcWriteIntfTomo(FILEINFO *fi)
{
  Uint32 total_energy_windows=fi->dim[7], total_detector_heads=fi->dim[6];
  Uint32 head, img=0, planes = fi->dim[3], images_per_window, fnr;
  float slice_thickness, slice_separation, study_duration=0., proj_duration=0.;
  ACQ_DATA *acq = NULL;
  IMG_DATA *id = &fi->image[0];
  DYNAMIC_DATA *dd = NULL;
  FILE *fp  = fi->ofp;
  char *msg = NULL;

  images_per_window = fi->number / total_energy_windows;

  if (fi->diff_size == MDC_YES)
    return("INTF Tomographic different sizes unsupported");
  if (fi->diff_type == MDC_YES)
    return("INTF Tomographic different types unsupported");

  fnr = id->frame_number;
  if ((fi->dynnr > 0) && (fnr > 0)) {
    dd = &fi->dyndata[fnr - 1];
    study_duration = dd->time_frame_duration;
    proj_duration  = dd->time_frame_duration / dd->nr_of_slices;
  }

  /* in pixels instead of mm */
  slice_thickness=id->slice_width/((id->pixel_xsize+id->pixel_ysize)/2.);
  slice_separation=id->slice_spacing/((id->pixel_xsize+id->pixel_ysize)/2.);

  fprintf(fp,";\r\n");
  fprintf(fp,"!SPECT STUDY (general) :=\r\n");
  fprintf(fp,"number of detector heads := %u\r\n",total_detector_heads);

  for (head=0; head < total_detector_heads; head++, ACQI++) {
     if (ACQI < fi->acqnr && fi->acqdata != NULL) {
       acq = &fi->acqdata[ACQI];
     }else{
       acq = NULL;
     }
     fprintf(fp,";\r\n");
     fprintf(fp,"!number of images/energy window := %u\r\n",images_per_window);
     fprintf(fp,"!process status := ");
     if (fi->reconstructed == MDC_NO) {
       fprintf(fp,"Acquired\r\n");
     }else{
       fprintf(fp,"Reconstructed\r\n");
     }

     msg = MdcWriteMatrixInfo(fi, img);
     if (msg != NULL) return(msg);

     fprintf(fp,"!number of projections := %u\r\n",planes);
     fprintf(fp,"!extent of rotation := ");
     if (acq != NULL) fprintf(fp,"%g",acq->angle_step*(float)planes);
     fprintf(fp,"\r\n");
     fprintf(fp,"!time per projection (sec) := %.7g\r\n",proj_duration / 1000.);
     fprintf(fp,"study duration (sec) := %.7g\r\n",study_duration / 1000.);
     fprintf(fp,"!maximum pixel count := ");
     if (MDC_FORCE_INT != MDC_NO) {
       switch (MDC_FORCE_INT) {
         case BIT8_U: 
          fprintf(fp,"%+e",(float)MDC_MAX_BIT8_U);
          break;
         case BIT16_S:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
          break;
         default:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
       } 
     }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
          fprintf(fp,"%+e",fi->qglmax);
     }else{
          fprintf(fp,"%+e",fi->glmax);
     }
     fprintf(fp,"\r\n");

     fprintf(fp,"patient orientation := %s\r\n"
               ,MdcSetPatOrientation(fi->pat_slice_orient));
     fprintf(fp,"patient rotation := %s\r\n"
               ,MdcSetPatRotation(fi->pat_slice_orient));
     if (fi->reconstructed == MDC_NO) {
          fprintf(fp,";\r\n");
          fprintf(fp,"!SPECT STUDY (acquired data) :=\r\n");
          fprintf(fp,"!direction of rotation := ");
          if (acq != NULL) {
            switch (acq->rotation_direction) {
              case MDC_ROTATION_CW: fprintf(fp,"CW");  break;
              case MDC_ROTATION_CC: fprintf(fp,"CCW"); break;
            }
          }
          fprintf(fp,"\r\n");
          fprintf(fp,"start angle := ");
          if (acq != NULL) {
            fprintf(fp,"%g",acq->angle_start);
          }
          fprintf(fp,"\r\n");
          fprintf(fp,"first projection angle in data set :=\r\n");
          fprintf(fp,"acquisition mode := ");
          if (acq != NULL) {
            switch (acq->detector_motion) {
              case MDC_MOTION_STEP: fprintf(fp,"stepped");    break;
              case MDC_MOTION_CONT: fprintf(fp,"continuous"); break;
              default             : fprintf(fp,"unknown");
            }
            fprintf(fp,"\r\n");
            if (acq->rotation_offset != 0.) {
              fprintf(fp,"Centre_of_rotation := Single_value\r\n");
              fprintf(fp,"!X_offset := %.7g\r\n",acq->rotation_offset);
              fprintf(fp,"Y_offset := 0.\r\n");
              fprintf(fp,"Radius := %.7g\r\n",acq->radial_position);
            }else{
              fprintf(fp,"Centre_of_rotation := Corrected\r\n"); 
            }
          }else{
           fprintf(fp,"\r\n");
          }
          fprintf(fp,"orbit := circular\r\n");
          fprintf(fp,"preprocessed :=\r\n");
     }else{
          fprintf(fp,";\r\n");
          fprintf(fp,"!SPECT STUDY (reconstructed data) :=\r\n");
          fprintf(fp,"method of reconstruction := %s\r\n",fi->recon_method);
          fprintf(fp,"!number of slices := %u\r\n",planes);
          fprintf(fp,"number of reference frame := 0\r\n");
          fprintf(fp,"slice orientation := %s\r\n",
                        MdcGetStrSliceOrient(fi->pat_slice_orient));
          fprintf(fp,"slice thickness (pixels) := %+e\r\n",slice_thickness);
          fprintf(fp,"centre-centre slice separation (pixels) := %+e\r\n",
                                                         slice_separation);
          fprintf(fp,"filter name := %s\r\n",fi->filter_type);
          fprintf(fp,"filter parameters := Cutoff\r\n");
          /*fprintf(fp,"z-axis filter :=\r\n");*/
          /*fprintf(fp,"attenuation correction coefficient/cm :=\r\n");*/
          fprintf(fp,"method of attenuation correction := measured\r\n");
          fprintf(fp,"scatter corrected := N\r\n"); 
          /*fprintf(fp,"method of scatter correction :=\r\n");*/
          fprintf(fp,"oblique reconstruction := N\r\n");
          /*fprintf(fp,"oblique orientation :=\r\n");*/
     }

  } 

  if (ferror(fp)) return("INTF Error writing Tomographic Header");

  return(NULL);

}

char *MdcWriteIntfGated(FILEINFO *fi)
{
  GATED_DATA *gd, tmpgd;
  FILE *fp = fi->ofp;
  IMG_DATA *id = NULL;
  Uint32 time_window;
  char *msg = NULL;
  float v;

  if (fi->gatednr > 0 && fi->gdata != NULL) {
    gd = &fi->gdata[0];
  }else{
    gd = &tmpgd; MdcInitGD(gd);
  }

  fprintf(fp,";\r\n");
  fprintf(fp,"!GATED STUDY (general) :=\r\n");

  msg = MdcWriteMatrixInfo(fi, 0);
  if (msg != NULL) return(msg);

  fprintf(fp,"study duration (elapsed) sec := %.7g\r\n"
            ,gd->study_duration / 1000.);
  fprintf(fp,"number of cardiac cycles (observed) := %.7g\r\n"
            ,gd->cycles_observed);
                        
  fprintf(fp,";\r\n");
  fprintf(fp,"number of time windows := %u\r\n",fi->dim[5]);

  for (time_window=0; time_window<fi->dim[5]; time_window++) {
     id = &fi->image[time_window * fi->dim[3]];
     fprintf(fp,";\r\n");
     fprintf(fp,"!Gated Study (each time window) :=\r\n");
     fprintf(fp,"!time window number := %u\r\n",time_window+1);
     fprintf(fp,"!number of images in time window := %u\r\n",fi->dim[3]);
     fprintf(fp,"!image duration (sec) := %.7g\r\n",gd->image_duration / 1000.);
     fprintf(fp,"framing method := Forward\r\n");
     fprintf(fp,"time window lower limit (sec) := %.7g\r\n"
               ,gd->window_low  / 1000.);
     fprintf(fp,"time window upper limit (sec) := %.7g\r\n"
               ,gd->window_high / 1000.);
     if (gd->cycles_observed > 0.) {
       v = (gd->cycles_acquired * 100.) / gd->cycles_observed;
     }else{
       v = 100.;
     }
     fprintf(fp,"%% R-R cycles acquired this window := %.7g\r\n",v);
     fprintf(fp,"number of cardiac cycles (acquired) := %.7g\r\n"
               ,gd->cycles_acquired);
     fprintf(fp,"study duration (acquired) sec := %.7g\r\n"
               ,gd->study_duration / 1000.);
     fprintf(fp,"!maximum pixel count := ");
     if (MDC_FORCE_INT != MDC_NO) {
       switch (MDC_FORCE_INT) {
         case BIT8_U:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT8_U);
          break;
         case BIT16_S:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
          break;
         default:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
       }
     }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
          fprintf(fp,"%+e",id->qfmax);
     }else{
          fprintf(fp,"%+e",id->fmax);
     }
     fprintf(fp,"\r\n");

     fprintf(fp,"R-R histogram := N\r\n");

  }

  return(NULL);

}

char *MdcWriteIntfGSPECT(FILEINFO *fi)
{
  Uint32 total_energy_windows=fi->dim[7], total_detector_heads=fi->dim[6];
  Uint32 time_window, head, planes = fi->dim[3], images_per_window;
  float slice_thickness, slice_separation, v;
  GATED_DATA *gd = NULL, tmpgd;
  ACQ_DATA *acq = NULL, tmpacq;
  IMG_DATA *id = &fi->image[0];
  FILE *fp = fi->ofp;
  char *msg = NULL;

  if (fi->gatednr > 0 && fi->gdata != NULL) { 
    /* use true struct */
    gd = &fi->gdata[0];
  }else{
    /* use temp struct */
    gd = &tmpgd; MdcInitGD(gd);
  }

  images_per_window = fi->number / total_energy_windows;

  if (fi->diff_size == MDC_YES)
    return("INTF Gated SPECT different sizes unsupported");
  if (fi->diff_type == MDC_YES)
    return("INTF Gated SPECT different types unsupported");

  /* in pixels instead of mm */
  slice_thickness=id->slice_width/((id->pixel_xsize+id->pixel_ysize)/2.);
  slice_separation=id->slice_spacing/((id->pixel_xsize+id->pixel_ysize)/2.);

  fprintf(fp,";\r\n");
  fprintf(fp,"!GATED SPECT STUDY (general) :=\r\n");

  msg = MdcWriteMatrixInfo(fi, 0);
  if (msg != NULL) return(msg);

  fprintf(fp,"!gated SPECT nesting outer level := %s\r\n"
            ,MdcGetStrGSpectNesting(gd->gspect_nesting));

  fprintf(fp,"study duration (elapsed) sec := %.7g\r\n"
            ,gd->study_duration / 1000.);
  fprintf(fp,"number of cardiac cycles (observed) := %.7g\r\n"
            ,gd->cycles_observed);

  fprintf(fp,";\r\n");
  fprintf(fp,"number of time windows := %u\r\n",fi->dim[5]);

  for (time_window=0; time_window<fi->dim[5]; time_window++) {
     id = &fi->image[time_window * fi->dim[3]];
     fprintf(fp,";\r\n");
     fprintf(fp,"!Gated Study (each time window) :=\r\n");
     fprintf(fp,"!time window number := %u\r\n",time_window+1);
     fprintf(fp,"!number of images in time window := %u\r\n",fi->dim[4]);
     fprintf(fp,"!image duration (sec) := %.7g\r\n",gd->image_duration / 1000.);
     fprintf(fp,"framing method := Forward\r\n");
     fprintf(fp,"time window lower limit (sec) := %.7g\r\n"
               ,gd->window_low  / 1000.);
     fprintf(fp,"time window upper limit (sec) := %.7g\r\n"
               ,gd->window_high / 1000.);
     if (gd->cycles_observed > 0.) {
       v = (gd->cycles_acquired * 100.) / gd->cycles_observed;
     }else{
       v = 100.;
     }
     fprintf(fp,"%% R-R cycles acquired this window := %.7g\r\n",v);
     fprintf(fp,"number of cardiac cycles (acquired) := %.7g\r\n"
               ,gd->cycles_acquired);
     fprintf(fp,"study duration (acquired) sec := %.7g\r\n"
               ,gd->study_duration / 1000.);
     fprintf(fp,"!maximum pixel count := ");
     if (MDC_FORCE_INT != MDC_NO) {
       switch (MDC_FORCE_INT) {
         case BIT8_U:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT8_U);
          break;
         case BIT16_S:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
          break;
         default:
          fprintf(fp,"%+e",(float)MDC_MAX_BIT16_S);
       }
     }else if (MDC_QUANTIFY || MDC_CALIBRATE) {
          fprintf(fp,"%+e",id->qfmax);
     }else{
          fprintf(fp,"%+e",id->fmax);
     }
     fprintf(fp,"\r\n");

     fprintf(fp,"R-R histogram := N\r\n");

  }
  
  fprintf(fp,";\r\n");
  fprintf(fp,"number of detector heads := %u\r\n",fi->dim[6]);

  for (head=0; head<total_detector_heads; head++, ACQI++) {
     if (ACQI < fi->acqnr && fi->acqdata != NULL) {
       acq = &fi->acqdata[ACQI];
     }else{
       acq = &tmpacq; MdcInitAD(acq);
     }
     fprintf(fp,";\r\n");
     fprintf(fp,"!number of images/energy window := %u\r\n",images_per_window);
     fprintf(fp,"!process status := ");
     if (fi->reconstructed == MDC_NO) {
       fprintf(fp,"Acquired\r\n");
     }else{
       fprintf(fp,"Reconstructed\r\n");
     }

     fprintf(fp,"!number of projections := %g\r\n",gd->nr_projections);
     fprintf(fp,"!extent of rotation := %g\r\n",gd->extent_rotation);
     fprintf(fp,"!time per projection (sec) := %.7g\r\n"
               ,gd->time_per_proj / 1000.0);
     fprintf(fp,"patient orientation := %s\r\n"
               ,MdcSetPatOrientation(fi->pat_slice_orient));
     fprintf(fp,"patient rotation := %s\r\n"
               ,MdcSetPatRotation(fi->pat_slice_orient));
     if (fi->reconstructed == MDC_NO) {
          fprintf(fp,";\r\n");
          fprintf(fp,"!SPECT STUDY (acquired data) :=\r\n");
          fprintf(fp,"!direction of rotation := ");
          switch (acq->rotation_direction) {
            case MDC_ROTATION_CW: fprintf(fp,"CW");  break;
            case MDC_ROTATION_CC: fprintf(fp,"CCW"); break;
          }
          fprintf(fp,"\r\n");
          fprintf(fp,"start angle := %g",acq->angle_start);
          fprintf(fp,"\r\n");
          fprintf(fp,"first projection angle in data set :=\r\n");
          fprintf(fp,"acquisition mode := ");
          if (acq != NULL) {
            switch (acq->detector_motion) {
              case MDC_MOTION_STEP: fprintf(fp,"stepped");    break;
              case MDC_MOTION_CONT: fprintf(fp,"continuous"); break;
              default             : fprintf(fp,"unknown");
            }
            fprintf(fp,"\r\n");
            if (acq->rotation_offset != 0.) {
              fprintf(fp,"Centre_of_rotation := Single_value\r\n");
              fprintf(fp,"!X_offset := %.7g\r\n",acq->rotation_offset);
              fprintf(fp,"Y_offset := 0.\r\n");
              fprintf(fp,"Radius := %.7g\r\n",acq->radial_position);
            }else{
              fprintf(fp,"Centre_of_rotation := Corrected\r\n");
            }
          }else{
            fprintf(fp,"\r\n");
          }
          fprintf(fp,"orbit := circular\r\n");
          fprintf(fp,"preprocessed :=\r\n");
     }else{
          fprintf(fp,";\r\n");
          fprintf(fp,"!SPECT STUDY (reconstructed data) :=\r\n");
          fprintf(fp,"method of reconstruction := %s\r\n",fi->recon_method);
          fprintf(fp,"!number of slices := %u\r\n",planes);
          fprintf(fp,"number of reference frame := 0\r\n");
          fprintf(fp,"slice orientation := %s\r\n",
                        MdcGetStrSliceOrient(fi->pat_slice_orient));
          fprintf(fp,"slice thickness (pixels) := %+e\r\n",slice_thickness);
          fprintf(fp,"centre-centre slice separation (pixels) := %+e\r\n",
                                                         slice_separation);
          fprintf(fp,"filter name := %s\r\n",fi->filter_type);
          fprintf(fp,"filter parameters := Cutoff\r\n");
          /*fprintf(fp,"z-axis filter :=\r\n");*/
          /*fprintf(fp,"attenuation correction coefficient/cm :=\r\n");*/
          fprintf(fp,"method of attenuation correction := measured\r\n");
          fprintf(fp,"scatter corrected := N\r\n"); 
          /*fprintf(fp,"method of scatter correction :=\r\n");*/
          fprintf(fp,"oblique reconstruction := N\r\n");
          /*fprintf(fp,"oblique orientation :=\r\n");*/
     }

  }

  return(NULL);

}

/// Lo modifico para que tenga mis datos.
char *MdcWriteIntfHeader(FILEINFO *fi)
{
  FILE *fp = fi->ofp;
  char *msg=NULL;
  int i, t, offset=0;

  if (MDC_SINGLE_FILE == MDC_YES) fseek(fp,0,SEEK_SET); /* at begin of file */

  fprintf(fp,"!INTERFILE :=\r\n");
  fprintf(fp,"!imaging modality := nucmed\r\n");
  fprintf(fp,"!originating system := %s\r\n",fi->manufacturer);
  fprintf(fp,"!version of keys := %s\r\n",MDC_INTF_SUPP_VERS);
  fprintf(fp,"date of keys := %s\r\n",MDC_INTF_SUPP_DATE);
  fprintf(fp,"conversion program := %s\r\n",MDC_PRGR);
  fprintf(fp,"program author := Martin Belzunce (Modulo interfile de Erik Nolf(Medcon))\r\n");
  fprintf(fp,"program version := medcon %s\r\n",MDC_VERSION);
  fprintf(fp,"program date := %s\r\n",MdcGetProgramDate());
  fprintf(fp,";\r\n");
  fprintf(fp,"!GENERAL DATA :=\r\n");
  fprintf(fp,"original institution := %s\r\n",fi->institution);
  if (MDC_SINGLE_FILE == MDC_YES) offset = MDC_INTF_DATA_OFFSET;
  fprintf(fp,"!data offset in bytes := %d\r\n",offset);
  if (XMDC_GUI == MDC_YES) MdcSplitPath(fi->opath,fi->odir,fi->ofname);
  MdcNewExt(fi->ofname,'\0',"i33");
  fprintf(fp,"!name of data file := %s\r\n",fi->ofname);
  MdcNewExt(fi->ofname,'\0',FrmtExt[MDC_FRMT_INTF]);
  if (XMDC_GUI == MDC_YES) MdcMergePath(fi->opath,fi->odir,fi->ofname);
  fprintf(fp,"patient name := %s\r\n",fi->patient_name);
  fprintf(fp,"!patient ID := %s\r\n",fi->patient_id);
 
  i=0; t=0;
  while (i < MDC_MAXSTR && i < strlen(fi->patient_dob)) { 
     if (i==4 || i==6) {
       mdcbufr[t++]=':';
     }
     mdcbufr[t++]=fi->patient_dob[i++];
  }
  mdcbufr[t]='\0';
 
  fprintf(fp,"patient dob := %s\r\n",mdcbufr);
  fprintf(fp,"patient sex := %s\r\n",fi->patient_sex);
  fprintf(fp,"!study ID := %s\r\n",fi->study_id);
  fprintf(fp,"exam type := %s\r\n",fi->series_descr);
  fprintf(fp,"data compression := none\r\n");
  fprintf(fp,"data encode := none\r\n");
  fprintf(fp,"organ := %s\r\n",fi->organ_code);
  if (strcmp(fi->radiopharma,"Unknown") == 0) {
    fprintf(fp,"isotope := %s\r\n",fi->isotope_code);
  }else{
    fprintf(fp,"isotope := %s/%s\r\n",fi->isotope_code,fi->radiopharma);
  }
  fprintf(fp,"dose := %g\r\n",fi->injected_dose);

#if MDC_INTF_SUPPORT_NUD
  fprintf(fp,"NUD/Patient Weight [kg] := %.2f\r\n",fi->patient_weight);
  fprintf(fp,"NUD/imaging modality := %s\r\n",MdcGetStrModality(fi->modality));
  fprintf(fp,"NUD/activity := %g\r\n",fi->injected_dose);
  fprintf(fp,"NUD/activity start time := %02d:%02d:%02d\r\n"
                                         ,fi->dose_time_hour
                                         ,fi->dose_time_minute
                                         ,fi->dose_time_second);
  fprintf(fp,"NUD/isotope half life [hours] := %f\r\n"
                                         ,fi->isotope_halflife / 3600.);
#endif

  msg = MdcWriteGenImgData(fi);
  if (msg != NULL) return(msg);

  msg = MdcWriteWindows(fi);
  if (msg != NULL) return(msg);

  fprintf(fp,"!END OF INTERFILE :=\r\n%c",MDC_CNTRL_Z);

  if (ferror(fp)) return("INTF Bad write header file");

  if (MDC_SINGLE_FILE && (ftell(fp) >= offset))
    return("INTF Predefined data offset in bytes too small");

  return(NULL);

}


char *MdcWriteIntfImages(FILEINFO *fi)
{
  IMG_DATA *id;
  FILE *fp = fi->ofp;
  Uint32 i, size;
  Uint8 *buf;

  if (MDC_SINGLE_FILE == MDC_YES) fseek(fp,MDC_INTF_DATA_OFFSET,SEEK_SET);
 
  for (i=0; i<fi->number; i++) {

     if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_INCR,1./(float)fi->number,NULL);

     id = &fi->image[i];
     size = id->width * id->height;
     if (MDC_FORCE_INT != MDC_NO) {
       switch (MDC_FORCE_INT) {
         case BIT8_U: 
             buf = MdcGetImgBIT8_U(fi, i);
             if (buf == NULL) return("INTF Bad malloc Uint8 buffer"); 
             /* no endian swap necessary */
             if (fwrite(buf,(unsigned)MdcType2Bytes(BIT8_U),size,fp) != size) {
               MdcFree(buf);
               return("INTF Bad write Uint8 image");
             }
             break;
         case BIT16_S: 
             buf = MdcGetImgBIT16_S(fi, i);
             if (buf == NULL) return("INTF Bad malloc Int16 buffer");
             if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN)
               MdcMakeImgSwapped(buf,fi,i,id->width,id->height,BIT16_S);
             if (fwrite(buf,(unsigned)MdcType2Bytes(BIT16_S),size,fp) != size) {
               MdcFree(buf);
               return("INTF Bad write Int16 image");
             }
             break;
         default:
             buf = MdcGetImgBIT16_S(fi, i);
             if (buf == NULL) return("INTF Bad malloc Int16 buffer");
             if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN)
               MdcMakeImgSwapped(buf,fi,i,id->width,id->height,BIT16_S);
             if (fwrite(buf,(unsigned)MdcType2Bytes(BIT16_S),size,fp) != size) {
               MdcFree(buf);
               return("INTF Bad write Int16 image");
             }
             
       }
       MdcFree(buf); 
     }else if (!(MDC_QUANTIFY || MDC_CALIBRATE)) {
       switch ( id->type ) {
         case  BIT1: return("INTF 1-Bit format unsupported"); break;
         case ASCII: return("INTF Ascii format unsupported"); break;
         default:
             if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN && 
                  id->type != BIT8_U && id->type != BIT8_S) {
               buf = MdcGetImgSwapped(fi,i);
               if (buf == NULL) return("INTF Couldn't malloc swapped image");
               if (fwrite(buf,(unsigned)MdcType2Bytes(id->type),size,fp)
                   !=size) {
                 MdcFree(buf);
                 return("INTF Bad write swapped image"); 
               }
               MdcFree(buf);
             }else{
               if (fwrite(id->buf,(unsigned)MdcType2Bytes(id->type),size,fp)
                   !=size) {
                 return("INTF Bad write image");
               }
             }
       }
     }else{
       buf = MdcGetImgFLT32( fi, i);
       if (buf == NULL) return("INTF Bad malloc buf");
       if (MDC_FILE_ENDIAN != MDC_HOST_ENDIAN) 
         MdcMakeImgSwapped(buf,fi,i,id->width,id->height,FLT32);
       if (fwrite(buf,(unsigned)MdcType2Bytes(FLT32),size,fp) != size) {
         MdcFree(buf);
         return("INTF Bad write quantified image");
       }
       MdcFree(buf);
     } 
  }

  return NULL;

}


const char *MdcWriteINTF(FILEINFO *fi)
{
  const char *err;
  char tmpfname[MDC_MAX_PATH + 1];

  MDC_FILE_ENDIAN = MDC_WRITE_ENDIAN;

  /* get filename */
  if (XMDC_GUI == MDC_YES) {
    strcpy(tmpfname,fi->opath);
  }else{
    if (MDC_ALIAS_NAME == MDC_YES) {
      MdcAliasName(fi,tmpfname);
    }else{
      strcpy(tmpfname,fi->ifname);
    }
    MdcDefaultName(fi,MDC_FRMT_INTF,fi->ofname,tmpfname);
  }

  if (MDC_PROGRESS) MdcProgress(MDC_PROGRESS_BEGIN,0.,"Writing InterFile:");

  if (MDC_VERBOSE) MdcPrntMesg("INTF Writing <%s> & <.i33> ...",fi->ofname);

  /* check for colored files */
  if (fi->map == MDC_MAP_PRESENT)
    return("INTF Colored files unsupported");

  /* first we write the image file */
  if (XMDC_GUI == MDC_YES) {
    fi->ofname[0]='\0'; MdcNewExt(fi->ofname,tmpfname,"i33");
  }else{ 
    MdcNewName(fi->ofname,tmpfname,"i33");  
  }

  if (MDC_FILE_STDOUT == MDC_YES) {
    /* send image data to stdout (1>stdout) */
    fi->ofp = stdout;
  }else{
    if (MdcKeepFile(fi->ofname))
      return("INTF Image file exists!!");
    if ( (fi->ofp=fopen(fi->ofname,"wb")) == NULL)
      return("INTF Couldn't open image file");
  }
  err = MdcWriteIntfImages(fi);
  if (err != NULL) return(err);

  /* write header, now we got rescale info */

  if (MDC_SINGLE_FILE == MDC_NO) {

     MdcCloseFile(fi->ofp);

     if (XMDC_GUI == MDC_YES) {
       strcpy(fi->ofname,tmpfname);
     }else{
        MdcDefaultName(fi,MDC_FRMT_INTF,fi->ofname,tmpfname);
     }
  }

  if (MDC_FILE_STDOUT == MDC_YES) {
    /* send header to stderr (2>stderr) */
    fi->ofp = stderr;
  }else if (MDC_SINGLE_FILE == MDC_NO) {
    if (MdcKeepFile(fi->ofname))
      return("INTF Header file exists!!");
    if ( (fi->ofp=fopen(fi->ofname,"wb")) == NULL)
      return("INTF Couldn't open header file");
  }

  MdcCheckIntfDim(fi);

  err = MdcWriteIntfHeader(fi);
  if (err != NULL) return(err);

  MdcCloseFile(fi->ofp);

  return NULL;

}
